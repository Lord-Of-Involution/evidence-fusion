# 修改 evidence/gnn_models.py

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, global_mean_pool, global_max_pool

from torch_geometric.utils import to_dense_batch

@torch.no_grad()
def native_radius_graph(pos, r, global_batch):

    edge_indices =[]
    unique_batches = torch.unique(global_batch)
    
    for b_idx in unique_batches:
        node_indices = torch.where(global_batch == b_idx)[0]
        pos_b = pos[node_indices]
        
        # 逐框计算距离矩阵，算完即毁，完美控制显存
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

class AnisotropicGNN(MessagePassing):
    # 【默认半径扩大到 20 Mpc】
    def __init__(self, in_channels, out_channels, r_link=20.0):
        # 【极其关键】将 aggr='mean' 改为 aggr='add'。
        # 这样网络会自动对邻居的信息求和，完美等效于计算暗物质晕周围的“局部密度”！
        super().__init__(aggr='add') 
        self.r_link = r_link
        
        # 【物理升级：输入维度变为 5】
        self.edge_mlp = nn.Sequential(
            nn.Linear(5, 32),
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
        
        # 【物理升级：加入带符号的相空间信息 signed_z】
        signed_z = diff[:, 2] 
        r_parallel = torch.abs(diff[:, 2]) 
        r_perp = torch.sqrt(diff[:, 0]**2 + diff[:, 1]**2) 
        
        r_diff = torch.abs(r_parallel - r_perp)
        r_ratio = r_parallel / (r_perp + 1e-5)
        
        # 5维无死角物理边特征
        edge_attr = torch.stack([signed_z, r_parallel, r_perp, r_diff, r_ratio], dim=1)
        
        x = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=(x.size(0), x.size(0)))
        return x

    def message(self, edge_attr):
        return self.edge_mlp(edge_attr)

    def update(self, aggr_out, x):
        combined = torch.cat([x, aggr_out], dim=1)
        return self.node_mlp(combined)

class EvidenceGNN(nn.Module):
    # 默认 r_link 提升为 20.0
    def __init__(self, r_link=20.0, hidden_dim=64):
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
        
        # 护城河：强制零初始化
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        
        # 【防线补充】：加入 MIL Temperature 控制求和导致的爆炸
        self.mil_temp = nn.Parameter(torch.tensor(1.0))

    def forward(self, data):
        x, pos, batch, sub_batch = data.x, data.pos, data.batch, data.sub_batch
        M = sub_batch.max().item() + 1
        global_batch = batch * M + sub_batch
        
        x = self.conv1(x, pos, global_batch)
        x = self.conv2(x, pos, global_batch)
        
        x_mean = global_mean_pool(x, global_batch)
        x_max = global_max_pool(x, global_batch)
        x_min = -global_max_pool(-x, global_batch)
        x_sq_mean = global_mean_pool(x**2, global_batch)
        x_var = torch.relu(x_sq_mean - x_mean**2) 
        
        global_feat = torch.cat([x_mean, x_max, x_min, x_var], dim=1)
        
        local_evidences = self.head(global_feat)
        local_evidences = local_evidences.view(batch.max().item() + 1, M)
        
        # 【物理升级】用 mean() 替代 sum()，并除以可学习的 Temperature，防止校准崩塌
        safe_temp = torch.clamp(self.mil_temp, min=0.1)
        global_evidence = local_evidences.mean(dim=1, keepdim=True) / safe_temp
        
        return global_evidence