import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, global_mean_pool, global_max_pool

@torch.no_grad()
def native_radius_graph(pos, r, global_batch):
    edge_indices = []
    unique_batches = torch.unique(global_batch)
    
    for b_idx in unique_batches:
        node_indices = torch.where(global_batch == b_idx)[0]
        pos_b = pos[node_indices]
        
        dist_mat = torch.cdist(pos_b, pos_b)
        mask = (dist_mat <= r) & (dist_mat > 1e-5)
        row_local, col_local = torch.where(mask)
        
        row_global = node_indices[row_local]
        col_global = node_indices[col_local]
        
        edge_indices.append(torch.stack([row_global, col_global], dim=0))
        
    if len(edge_indices) == 0:
        return torch.empty((2, 0), dtype=torch.long, device=pos.device)
        
    return torch.cat(edge_indices, dim=1)


class AnisotropicGNN(MessagePassing):
    def __init__(self, in_channels, out_channels, r_link=20.0, geometry="lightcone"):
        # [ULTIMATE PHYSICS UPGRADE: Shape, Dispersion, and Explicit Density]
        super().__init__(aggr=['mean', 'std']) 
        assert geometry in ["plane_parallel", "lightcone"], "FATAL: Invalid geometry mode"
        
        self.r_link = r_link
        self.geometry = geometry
        self.edge_out_dim = 32
        
        self.edge_mlp = nn.Sequential(
            nn.Linear(in_channels + 5, 32),
            nn.GELU(),
            nn.Linear(32, self.edge_out_dim)
        )
        
        # Aggr out: 32 (mean) + 32 (std) + 1 (log-degree/local density) = 65
        aggr_dim = (self.edge_out_dim * 2) + 1
        
        self.node_mlp = nn.Sequential(
            nn.Linear(in_channels + aggr_dim, out_channels),
            nn.BatchNorm1d(out_channels), 
            nn.GELU(),
            nn.Linear(out_channels, out_channels)
        )

    def forward(self, x, pos, global_batch, los=None):
        edge_index = native_radius_graph(pos, r=self.r_link, global_batch=global_batch)
        row, col = edge_index
        
        # [CRITICAL ARMOR: Decoupled Density]
        # Calculate explicit local degree for each receiver node.
        # minlength ensures the tensor matches the exact number of nodes even if some are isolated.
        deg = torch.bincount(row, minlength=pos.size(0)).view(-1, 1).float()
        # Log-compression tames [0, 800] into a safe neural scale [0, 6.6]
        deg_norm = torch.log1p(deg)
        
        pos_row = pos[row] 
        pos_col = pos[col] 
        diff = pos_row - pos_col
        
        if self.geometry == "plane_parallel":
            signed_z = diff[:, 2] 
            r_parallel = torch.abs(signed_z) 
            r_perp = torch.sqrt(diff[:, 0]**2 + diff[:, 1]**2)
            
        elif self.geometry == "lightcone":
            n_hat = los[row]
            signed_z = (diff * n_hat).sum(dim=1)
            r_parallel = torch.abs(signed_z)
            
            diff_norm_sq = torch.sum(diff**2, dim=1)
            r_perp = torch.sqrt(torch.clamp(diff_norm_sq - r_parallel**2, min=1e-8))
        
        norm_factor = self.r_link + 1e-5
        r_norm = torch.sqrt(r_parallel**2 + r_perp**2 + 1e-8)
        cos_theta = r_parallel / r_norm  
        
        z_scaled = signed_z / norm_factor
        para_scaled = r_parallel / norm_factor
        perp_scaled = r_perp / norm_factor
        norm_scaled = r_norm / norm_factor
        
        edge_attr = torch.stack([z_scaled, para_scaled, perp_scaled, cos_theta, norm_scaled], dim=1)
        
        # Pass deg_norm through the propagate mechanism to reach the update function
        x = self.propagate(edge_index, x=x, edge_attr=edge_attr, deg_norm=deg_norm, size=(x.size(0), x.size(0)))
        return x

    def message(self, x_j, edge_attr):
        combined = torch.cat([x_j, edge_attr], dim=-1)
        raw_message = self.edge_mlp(combined)
        
        # Spatial envelope enforcing gravity's inductive bias
        spatial_envelope = torch.exp(-3.0 * edge_attr[:, 4]).unsqueeze(1)
        
        return raw_message * spatial_envelope

    def update(self, aggr_out, x, deg_norm):
        # aggr_out: [N, 64] | deg_norm: [N, 1]
        # The node explicitly "sees" its decoupled absolute density alongside its spatial shape!
        combined = torch.cat([x, aggr_out, deg_norm], dim=1)
        out = self.node_mlp(combined)
        
        if x.size(1) == out.size(1):
            out = out + x
            
        return out


class EvidenceGNN(nn.Module):
    def __init__(self, r_link=20.0, hidden_dim=64, geometry="lightcone"):
        super().__init__()
        self.conv1 = AnisotropicGNN(in_channels=1, out_channels=hidden_dim, r_link=r_link, geometry=geometry)
        self.conv2 = AnisotropicGNN(in_channels=hidden_dim, out_channels=hidden_dim, r_link=r_link, geometry=geometry)
        
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 4, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1) 
        )
        
        self.mil_temp = nn.Parameter(torch.tensor(1.0))

    def forward(self, data):
        x, pos, batch, sub_batch = data.x, data.pos, data.batch, data.sub_batch
        M = sub_batch.max().item() + 1
        global_batch = batch * M + sub_batch
        
        los = getattr(data, 'los', None)
        
        x = self.conv1(x, pos, global_batch, los=los)
        x = self.conv2(x, pos, global_batch, los=los)
        
        x_mean = global_mean_pool(x, global_batch)
        x_max = global_max_pool(x, global_batch)
        x_min = -global_max_pool(-x, global_batch)
        x_sq_mean = global_mean_pool(x**2, global_batch)
        x_var = torch.relu(x_sq_mean - x_mean**2) 
        
        global_feat = torch.cat([x_mean, x_max, x_min, x_var], dim=1)
        
        local_evidences = self.head(global_feat)
        local_evidences = local_evidences.view(batch.max().item() + 1, M)
        
        safe_temp = torch.clamp(self.mil_temp, min=0.1)
        global_evidence = local_evidences.mean(dim=1, keepdim=True) / safe_temp
        
        return global_evidence