import torch
import torch.nn as nn
import torch.nn.functional as F
from evidence.core import l_pop_transform

# ==============================================================================
# 0. Shared Components (Residual Block)
# ==============================================================================
class ResidualBlock(nn.Module):
    """
    Generic Residual Block.
    Handles 3D data (for CNN) and 1D vector data (for Fusion Head).
    """
    def __init__(self, in_c, out_c, stride=1, is_3d=True):
        super().__init__()
        if is_3d:
            # 3D Convolution for Voxel Grids
            self.net = nn.Sequential(
                nn.Conv3d(in_c, out_c, 3, padding=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_c),
                nn.GELU(),
                nn.Conv3d(out_c, out_c, 3, padding=1, bias=False),
                nn.BatchNorm3d(out_c)
            )
            # Shortcut for 3D
            self.shortcut = nn.Sequential()
            if stride != 1 or in_c != out_c:
                self.shortcut = nn.Sequential(
                    nn.Conv3d(in_c, out_c, 1, stride=stride, bias=False),
                    nn.BatchNorm3d(out_c)
                )
        else:
            # 1D Linear for Feature Vectors (Fusion Head)
            self.net = nn.Sequential(
                nn.Linear(in_c, out_c),
                nn.BatchNorm1d(out_c),
                nn.GELU(),
                nn.Linear(out_c, out_c),
                nn.BatchNorm1d(out_c)
            )
            # Shortcut for 1D
            self.shortcut = nn.Sequential()
            if in_c != out_c:
                self.shortcut = nn.Sequential(
                    nn.Linear(in_c, out_c),
                    nn.BatchNorm1d(out_c)
                )

    def forward(self, x):
        out = self.net(x)
        out += self.shortcut(x)
        return F.gelu(out)

# ==============================================================================
# 1. The Evidence MLP (Baseline & Teacher)
# ==============================================================================
class EvidenceMLP(nn.Module):
    """
    Standard MLP for vector data (Pk, Bk).
    [Restored to user's original definition to ensure weight compatibility]
    """
    def __init__(self, input_size, hidden_sizes=[512, 512, 128], activation="gelu", dropout=0.0, batchnorm=True):
        super().__init__()
        layers = []
        in_dim = input_size
        
        # Select Activation
        act_fn = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}.get(activation.lower(), nn.GELU)
        
        # Build Backbone
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            if batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(act_fn())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
            
        self.backbone = nn.Sequential(*layers)
        self.output_dim = in_dim
        
        # The Evidence Head: Projects features to scalar f(x)
        self.head = nn.Linear(in_dim, 1) 

    def forward(self, x, return_features=False):
        # x shape: [Batch, Input_Dim]
        features = self.backbone(x)
        if return_features:
            return features
        return self.head(features)
        
    def predict_posterior(self, x, alpha=2.0):
        """Helper for inference"""
        with torch.no_grad():
            f_x = self.forward(x)
            J_val = l_pop_transform(f_x, alpha)
            return torch.sigmoid(J_val)



class EarlyFusionNetwork(nn.Module):
    """
    RIGOROUS EARLY FUSION:
    Takes an explicitly instantiated EvidenceMLP and MicroTopologyGNN.
    Concatenates their representation manifolds, and maps to J-Space.
    """
    def __init__(self, mlp_backbone, gnn_backbone, freeze_mlp=False):
        super().__init__()
        
        self.mlp = mlp_backbone
        self.gnn = gnn_backbone
        
        if freeze_mlp:
            for param in self.mlp.parameters():
                param.requires_grad = False
            self.mlp.eval()
            
        mlp_dim = self.mlp.output_dim
        gnn_dim = self.gnn.output_dim
        fusion_dim = mlp_dim + gnn_dim

        # ORTHOGONAL FUSION HEAD
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.BatchNorm1d(256),  # Kept consistent with MLP logic
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Linear(64, 1)
        )
        
        self.mil_temp = nn.Parameter(torch.tensor(1.0))

    def forward(self, data):
        # 1. Micro-Topology Extraction
        feat_gnn = self.gnn(
            x=data.x, 
            pos=data.pos, 
            batch=data.batch, 
            sub_batch=data.sub_batch, 
            los=getattr(data, 'los', None)
        )
        
        # 2. Macro-Spectra Extraction
        # data.vec must be reshaped to match batch size B
        B = data.batch.max().item() + 1
        macro_vec = data.vec.view(B, -1)
        
        if not self.mlp.training:
            with torch.no_grad():
                feat_mlp = self.mlp(macro_vec, return_features=True)
        else:
            feat_mlp = self.mlp(macro_vec, return_features=True)
            
        # 3. Fusion Representation
        combined_feat = torch.cat([feat_gnn, feat_mlp], dim=1)
        
        # 4. Map to Evidence
        global_evidence = self.fusion_head(combined_feat)
        
        safe_temp = torch.clamp(self.mil_temp, min=0.1)
        return global_evidence / safe_temp