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

# ==============================================================================
# 2. The 3D CNN (ResNet-10 Feature Extractor)
# ==============================================================================
class Small3DCNN(nn.Module):
    """
    ResNet-10 (3D) for Voxel Grids.
    """
    def __init__(self, input_channels=1, feature_dim=128):
        super().__init__()
        
        self.stem = nn.Sequential(
            nn.InstanceNorm3d(input_channels, affine=True),
            nn.Conv3d(input_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(64),
            nn.GELU()
        )
        
        self.layer1 = ResidualBlock(64, 64, stride=2, is_3d=True)
        self.layer2 = ResidualBlock(64, 128, stride=2, is_3d=True)
        self.layer3 = ResidualBlock(128, 256, stride=2, is_3d=True)
        self.layer4 = ResidualBlock(256, 512, stride=2, is_3d=True)
        
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, feature_dim),
            nn.GELU(),
            nn.BatchNorm1d(feature_dim)
        )

        self.output_dim = feature_dim 

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        return self.projection(x)

# ==============================================================================
# 3. The Fusion Network (Linear Shortcut + Deep Residual)
# ==============================================================================
class FusionEvidenceNetwork(nn.Module):
    """
    Combines Pre-trained MLP and CNN using 'Linear Shortcut' + 'Deep Residual' strategy.
    
    1. Shortcut: Copies MLP weights exactly. Ensures Initial Acc = MLP Acc.
    2. Residual: A deep ResNet that learns to correct the MLP using CNN data.
    """
    def __init__(self, mlp_model, cnn_model, freeze_mlp=True):
        super().__init__()
        self.mlp = mlp_model
        self.cnn = cnn_model
        
        if freeze_mlp:
            for param in self.mlp.parameters():
                param.requires_grad = False
            self.mlp.eval()
            
        mlp_dim = self.mlp.output_dim
        cnn_dim = self.cnn.output_dim
        fusion_dim = mlp_dim + cnn_dim

        # --- A. Linear Shortcut (The "Smart Init" Path) ---
        # Mimics the original MLP head exactly.
        # f_shortcut(mlp_feat) = f_mlp(raw_input)
        self.mlp_shortcut = nn.Linear(mlp_dim, 1)

        # --- B. Deep Residual Head (The "Correction" Path) ---
        # Takes [MLP_Features, CNN_Features] and learns a correction term.
        # Initialized to output 0.
        self.fusion_residual = nn.Sequential(
            nn.BatchNorm1d(fusion_dim),
            nn.Linear(fusion_dim, 256),
            nn.GELU(),
            # 1D Residual Block for non-linear fusion
            ResidualBlock(256, 256, is_3d=False), 
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )

        self._smart_init()

    def _smart_init(self):
        """
        Initialize weights to guarantee:
        Output = MLP_Prediction + 0
        """
        with torch.no_grad():
            # 1. Copy MLP Head weights to Shortcut
            # Note: We use self.mlp.head because we restored your MLP class structure
            self.mlp_shortcut.weight.copy_(self.mlp.head.weight)
            self.mlp_shortcut.bias.copy_(self.mlp.head.bias)
            
            # 2. Zero-init the Residual Head (Last layer only)
            self.fusion_residual[-1].weight.fill_(0.0)
            self.fusion_residual[-1].bias.fill_(0.0)

    def forward(self, x_grid, x_vectors):
            # 1. Get MLP Features (Thread-safe)
            with torch.no_grad():
                feat_mlp = self.mlp(x_vectors, return_features=True)
            
            # 2. Get CNN Features
            feat_cnn = self.cnn(x_grid)
            
            # 3. Path A: Base Prediction (Teacher)
            base_pred = self.mlp_shortcut(feat_mlp)
            
            # 4. Path B: Residual Correction (Deep Fusion)
            combined = torch.cat([feat_mlp, feat_cnn], dim=1)
            correction = self.fusion_residual(combined)
            
            # ✅ 正确写法：
            return base_pred + correction