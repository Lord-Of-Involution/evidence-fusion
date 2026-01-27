import torch
import torch.nn as nn
import torch.nn.functional as F
from evidence.core import l_pop_transform

# ==============================================================================
# 1. The Evidence MLP (Baseline & Teacher)
# ==============================================================================
class EvidenceMLP(nn.Module):
    """
    Standard MLP for vector data (Pk, Bk).
    Designed to output scalar f(x) for Evidence Loss.
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

# ==============================================================================
# 2. The 3D CNN (Feature Extractor)
# ==============================================================================
class Small3DCNN(nn.Module):
    """
    Lightweight 3D CNN for Voxel Grids.
    Structure: Conv3D -> BN -> Act -> MaxPool
    """
    def __init__(self, input_channels=1, feature_dim=64): 
        super().__init__()
        base = 16 
        
        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv3d(in_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm3d(out_c),
                nn.GELU(),
                nn.MaxPool3d(kernel_size=2, stride=2) 
            )
            
        # Downsampling: 3 blocks means input size / 8
        # If input is 32^3 -> 4^3
        self.features = nn.Sequential(
            conv_block(input_channels, base),       # -> [base, D/2, H/2, W/2]
            conv_block(base, base*2),               # -> [base*2, D/4, H/4, W/4]
            conv_block(base*2, base*4),             # -> [base*4, D/8, H/8, W/8]
        )
        
        # Assuming input allows flattening. 
        # AdaptiveAvgPool3d ensures fixed output size regardless of input grid size.
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base*4, feature_dim),
            nn.GELU(),
            nn.BatchNorm1d(feature_dim)
        )
        self.output_dim = feature_dim

    def forward(self, x):
        # x shape: [Batch, 1, D, H, W]
        x = self.features(x)
        x = self.pool(x) # Global Average Pooling
        return self.projection(x)

# ==============================================================================
# 3. The Fusion Network (Residual Correction)
# ==============================================================================
class FusionEvidenceNetwork(nn.Module):
    """
    Combines a Pre-trained MLP and a fresh CNN.
    Uses 'Smart Initialization' to start exactly at the MLP's performance level.
    """
    def __init__(self, mlp_model, cnn_model, freeze_mlp=True):
        super().__init__()
        self.mlp = mlp_model
        self.cnn = cnn_model
        
        if freeze_mlp:
            for param in self.mlp.parameters():
                param.requires_grad = False
            self.mlp.eval() # Ensure BN stats don't update
            
        # Joint Head
        # Inputs: [MLP_Features (Teacher), CNN_Features (Student)]
        joint_in = self.mlp.output_dim + self.cnn.output_dim
        self.joint_head = nn.Linear(joint_in, 1)

        self._smart_init()

    def _smart_init(self):
        """
        CRITICAL: Initialize the joint head so that:
        f_fusion(x) = f_mlp(x) + 0 * f_cnn(x)
        
        This allows the model to start training from the MLP's high accuracy,
        rather than from random noise.
        """
        with torch.no_grad():
            # 1. Zero out everything first
            self.joint_head.weight.fill_(0.0)
            self.joint_head.bias.fill_(0.0)
            
            # 2. Copy MLP Bias
            self.joint_head.bias.copy_(self.mlp.head.bias)
            
            # 3. Copy MLP Weights into the first part of the matrix
            # joint_weights: [1, mlp_dim + cnn_dim]
            # mlp_weights:   [1, mlp_dim]
            mlp_dim = self.mlp.output_dim
            self.joint_head.weight[:, :mlp_dim] = self.mlp.head.weight
            
            # The second part (corresponding to CNN) remains 0.0

    def forward(self, x_grid, x_vectors):
        # 1. Get MLP Features
        # Note: If frozen, we don't need gradients for this part
        if not next(self.mlp.parameters()).requires_grad:
            with torch.no_grad():
                feat_mlp = self.mlp(x_vectors, return_features=True)
        else:
            feat_mlp = self.mlp(x_vectors, return_features=True)
        
        # 2. Get CNN Features
        feat_cnn = self.cnn(x_grid)
        
        # 3. Concat and Predict
        combined = torch.cat([feat_mlp, feat_cnn], dim=1)
        return self.joint_head(combined)