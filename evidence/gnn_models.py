import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, global_mean_pool, global_max_pool, radius_graph

class AnisotropicGNN(MessagePassing):
    def __init__(self, in_channels, out_channels, r_link=20.0, geometry="lightcone"):
        # CRITICAL FIX: 'sum' is mandatory to preserve absolute local density (delta).
        super().__init__(aggr=['mean', 'std', 'sum']) 
        assert geometry in ["plane_parallel", "lightcone"], "FATAL: Invalid geometry mode"
        
        self.r_link = r_link
        self.geometry = geometry
        self.edge_out_dim = 32
        
        self.edge_mlp = nn.Sequential(
            nn.Linear(in_channels + 5, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Linear(32, self.edge_out_dim)
        )
        
        # 3 aggregators (mean, std, sum) * edge_out_dim
        aggr_dim = (self.edge_out_dim * 3) 
        
        self.node_mlp = nn.Sequential(
            nn.Linear(in_channels + aggr_dim, out_channels),
            nn.LayerNorm(out_channels), 
            nn.GELU(),
            nn.Linear(out_channels, out_channels)
        )

    def forward(self, x, pos, global_batch, los=None):
        edge_index = radius_graph(
            pos, 
            r=self.r_link, 
            batch=global_batch, 
            max_num_neighbors=64,
            loop=False
        )
        row, col = edge_index
        
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
        
        x = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return x

    def message(self, x_j, edge_attr):
        combined = torch.cat([x_j, edge_attr], dim=-1)
        raw_message = self.edge_mlp(combined)
        
        spatial_envelope = torch.exp(-3.0 * edge_attr[:, 4]).unsqueeze(1)
        return raw_message * spatial_envelope

    def update(self, aggr_out, x):
        combined = torch.cat([x, aggr_out], dim=1)
        out = self.node_mlp(combined)
        
        if x.size(1) == out.size(1):
            out = out + x
            
        return out

class MicroTopologyGNN(nn.Module):
    """
    PURE GNN BASELINE:
    Stripped of all macroscopic vector logic. 
    Outputs a highly non-linear feature manifold of the microscopic universe.
    """
    def __init__(self, r_link=20.0, hidden_dim=64, geometry="lightcone"):
        super().__init__()
        self.conv1 = AnisotropicGNN(in_channels=1, out_channels=hidden_dim, r_link=r_link, geometry=geometry)
        self.conv2 = AnisotropicGNN(in_channels=hidden_dim, out_channels=hidden_dim, r_link=r_link, geometry=geometry)
        
        # 4 pooling features (mean, max, min, var)
        self.gnn_feat_dim = hidden_dim * 4 
        base_dim = self.gnn_feat_dim * 2 # concat mean and max of subboxes
        
        self.output_dim = 128
        
        self.micro_encoder = nn.Sequential(
            nn.Linear(base_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, self.output_dim)
        )

    def forward(self, x, pos, batch, sub_batch, los=None):
        B = batch.max().item() + 1
        M = sub_batch.max().item() + 1
        global_batch = batch * M + sub_batch
        
        x = self.conv1(x, pos, global_batch, los=los)
        x = self.conv2(x, pos, global_batch, los=los)
        
        x_mean = global_mean_pool(x, global_batch)
        x_max = global_max_pool(x, global_batch)
        x_min = -global_max_pool(-x, global_batch)
        
        x_sq_mean = global_mean_pool(x**2, global_batch)
        x_var = torch.relu(x_sq_mean - x_mean**2) 
        
        subbox_feat = torch.cat([x_mean, x_max, x_min, x_var], dim=1)
        subbox_feat = subbox_feat.view(B, M, -1)
        cone_mean = subbox_feat.mean(dim=1)     
        cone_max = subbox_feat.max(dim=1)[0]    
        
        micro_raw = torch.cat([cone_mean, cone_max], dim=1)
        return self.micro_encoder(micro_raw)