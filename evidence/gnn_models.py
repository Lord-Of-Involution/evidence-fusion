import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, global_mean_pool, global_max_pool

# ==============================================================================
# 原生 Tensor Core 距离图，彻底抛弃 torch-cluster 编译黑洞
# ==============================================================================
def native_radius_graph(pos, r, global_batch):
    edge_indices =[]
    unique_batches = torch.unique(global_batch)
    
    for b_idx in unique_batches:
        node_indices = torch.where(global_batch == b_idx)[0]
        pos_b = pos[node_indices]
        
        # GPU 上光速完成 O(N^2) 距离矩阵
        dist_mat = torch.cdist(pos_b, pos_b)
        
        # 半径内，且不是自环
        mask = (dist_mat <= r) & (dist_mat > 1e-5)
        row_local, col_local = torch.where(mask)
        
        row_global = node_indices[row_local]
        col_global = node_indices[col_local]
        
        edge_indices.append(torch.stack([row_global, col_global], dim=0))
        
    if len(edge_indices) == 0:
        return torch.empty((2, 0), dtype=torch.long, device=pos.device)
        
    return torch.cat(edge_indices, dim=1)

# ==============================================================================
# Anisotropic GNN (保持两层，防止 Oversmoothing)
# ==============================================================================
class AnisotropicGNN(MessagePassing):
    def __init__(self, in_channels, out_channels, r_link=10.0):
        super().__init__(aggr='mean') 
        self.r_link = r_link
        
        # 物理护城河：4 个几何特征并行输入
        self.edge_mlp = nn.Sequential(
            nn.Linear(4, 32),
            nn.GELU(),
            nn.Linear(32, out_channels)
        )
        
        self.node_mlp = nn.Sequential(
            nn.Linear(in_channels + out_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Linear(out_channels, out_channels)
        )

    def forward(self, x, pos, global_batch):
        edge_index = native_radius_graph(pos, r=self.r_link, global_batch=global_batch)
        
        row, col = edge_index
        diff = pos[row] - pos[col]
        
        r_parallel = torch.abs(diff[:, 2]) 
        r_perp = torch.sqrt(diff[:, 0]**2 + diff[:, 1]**2) 
        
        r_diff = torch.abs(r_parallel - r_perp)
        r_ratio = r_parallel / (r_perp + 1e-5)
        
        edge_attr = torch.stack([r_parallel, r_perp, r_diff, r_ratio], dim=1)
        
        x = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=(x.size(0), x.size(0)))
        return x

    def message(self, edge_attr):
        return self.edge_mlp(edge_attr)

    def update(self, aggr_out, x):
        combined = torch.cat([x, aggr_out], dim=1)
        return self.node_mlp(combined)

# ==============================================================================
# Bayesian MIL Evidence GNN (全阶统计矩版本)
# ==============================================================================
class EvidenceGNN(nn.Module):
    def __init__(self, r_link=10.0, hidden_dim=64):
        super().__init__()
        self.conv1 = AnisotropicGNN(in_channels=1, out_channels=hidden_dim, r_link=r_link)
        self.conv2 = AnisotropicGNN(in_channels=hidden_dim, out_channels=hidden_dim, r_link=r_link)
        
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 4, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1) 
        )
        
        # 【护城河：强制零初始化】
        # 让网络出生的那一刻处于绝对无偏状态，Loss = exp(0) = 1.0，彻底告别 2 亿！
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, data):
        x, pos, batch, sub_batch = data.x, data.pos, data.batch, data.sub_batch
        M = sub_batch.max().item() + 1
        global_batch = batch * M + sub_batch
        
        # 1. 两次物理感受野传导
        x = self.conv1(x, pos, global_batch)
        x = self.conv2(x, pos, global_batch)
        
        # 2. 全阶物理统计矩提取
        x_mean = global_mean_pool(x, global_batch)               # 均值
        x_max = global_max_pool(x, global_batch)                 # 最大星系团核心
        x_min = -global_max_pool(-x, global_batch)               # 空洞探测器
        
        # 【护城河：用 Variance 替代 Std】
        # E[X^2] - E[X]^2，去掉 sqrt 防止导数在 0 处爆炸！
        x_sq_mean = global_mean_pool(x**2, global_batch)
        x_var = torch.relu(x_sq_mean - x_mean**2) 
        
        # 四神兽合体
        global_feat = torch.cat([x_mean, x_max, x_min, x_var], dim=1)
        
        # 3. 贝叶斯证据池化
        local_evidences = self.head(global_feat)
        local_evidences = local_evidences.view(batch.max().item() + 1, M)
        global_evidence = local_evidences.sum(dim=1, keepdim=True) 
        
        return global_evidence