#!/usr/bin/env python3
"""
[Step 6] Evaluate MLP: Validation & Calibration (Updated)
=========================================================
Features:
1. ROC Curve
2. Posterior Hist (Annotated with Loss & Acc)
3. Log-Bayes Factor Hist
4. Calibration Plot (Scatter + Binomial Error Bars)
"""

import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from torch.utils.data import Dataset, DataLoader

# ==============================================================================
# A. 模型与 Loss 定义
# ==============================================================================
class EvidenceNetworkPaper(nn.Module):
    def __init__(self, input_size, hidden_sizes, activation="gelu", dropout=0.0, batchnorm=False):
        super().__init__()
        layers = []
        in_dim = input_size
        act_fn = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}.get(activation.lower(), nn.GELU)
        
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            if batchnorm: layers.append(nn.BatchNorm1d(h))
            layers.append(act_fn())
            if dropout > 0.0: layers.append(nn.Dropout(dropout))
            in_dim = h
            
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(in_dim, 1) 

    def forward(self, x):
        return self.head(self.backbone(x))

def l_pop_transform(x, alpha=2.0):
    return x + x * torch.abs(x).pow(alpha - 1)

def one_pop_exponential_loss(f_x, targets, alpha=2.0):
    J_val = l_pop_transform(f_x, alpha)
    term = (0.5 - targets) * J_val
    loss = torch.exp(term)
    return loss  # Return raw tensor for mean calculation later

# ==============================================================================
# B. 数据集
# ==============================================================================
class VectorDataset(Dataset):
    def __init__(self, path):
        print(f"Loading {path}...")
        payload = torch.load(path, map_location="cpu")
        
        if 'data' in payload: self.x = payload['data']
        elif 'stats' in payload: self.x = payload['stats']
        else: self.x = payload['data_stats']
            
        self.y = payload['labels']
        self.x = self.x.float()
        self.y = self.y.float().view(-1, 1)
        
        if self.x.shape[0] < self.x.shape[1] and self.x.shape[0] <= 2000:
             self.x = self.x.T

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# ==============================================================================
# C. 核心逻辑
# ==============================================================================
def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Config
    config_path = os.path.join(args.model_dir, "mlp_config.json")
    if not os.path.exists(config_path):
        config_path = os.path.join(args.model_dir, "config.json")
    with open(config_path, "r") as f:
        conf = json.load(f)

    # 2. Data
    val_path = os.path.join(args.data_dir, "val_data.pt")
    val_ds = VectorDataset(val_path)
    val_dl = DataLoader(val_ds, batch_size=1024, shuffle=False)

    # 3. Model
    input_dim = val_ds.x.shape[1]
    model = EvidenceNetworkPaper(
        input_size=input_dim,
        hidden_sizes=conf["hidden_sizes"],
        dropout=conf.get("dropout", 0.0),
        batchnorm=conf.get("batchnorm", False)
    ).to(device)

    model_path = os.path.join(args.model_dir, "mlp_best.pt")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 4. Inference & Metrics
    all_preds = []   # P(H1|x)
    all_targets = [] # y_true
    all_logits = []  # J scores
    total_loss = 0.0
    num_samples = 0

    print("Running inference...")
    with torch.no_grad():
        for x, y in val_dl:
            x, y = x.to(device), y.to(device)
            f_x = model(x)
            
            # Loss Calculation
            batch_loss = one_pop_exponential_loss(f_x, y, alpha=2.0)
            total_loss += batch_loss.sum().item()
            num_samples += y.size(0)
            
            # Metrics
            J = l_pop_transform(f_x, alpha=2.0)
            prob = torch.sigmoid(J)
            
            all_logits.extend(J.cpu().numpy())
            all_preds.extend(prob.cpu().numpy())
            all_targets.extend(y.cpu().numpy())

    y_true = np.array(all_targets).flatten()
    y_prob = np.array(all_preds).flatten()
    y_logits = np.array(all_logits).flatten()
    y_log10_K = y_logits / np.log(10)

    # Global Metrics
    avg_loss = total_loss / num_samples
    accuracy = np.mean((y_prob > 0.5) == y_true)
    print(f"Validation Metrics -> Loss: {avg_loss:.4f} | Accuracy: {accuracy:.4f}")

    # ==========================================================================
    # 5. Plotting
    # ==========================================================================
    save_path = os.path.join(args.model_dir, "plots")
    os.makedirs(save_path, exist_ok=True)
    try: plt.style.use('seaborn-v0_8-paper')
    except: plt.style.use('ggplot')

    # --- 1. ROC Curve ---
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.4f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_path, "1_roc_curve.png"), dpi=300)
    plt.close()

    # --- 2. Posterior Histogram (Annotated) ---
    plt.figure(figsize=(8, 6))
    plt.hist(y_prob[y_true==0], bins=50, alpha=0.6, color='C0', label='Class 0 (Fiducial)', density=True)
    plt.hist(y_prob[y_true==1], bins=50, alpha=0.6, color='C1', label='Class 1 (Biased)', density=True)
    plt.axvline(0.5, color='k', linestyle='--', alpha=0.5)
    
    # Add Text Box for Metrics
    text_str = f'Val Loss: {avg_loss:.4f}\nAccuracy: {accuracy:.4f}'
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    plt.gca().text(0.05, 0.95, text_str, transform=plt.gca().transAxes, fontsize=12,
                   verticalalignment='top', bbox=props)
    
    plt.xlabel('Predicted P(H1|x)')
    plt.title('Posterior Distribution')
    plt.legend(loc='upper center')
    plt.savefig(os.path.join(save_path, "2_posterior_hist.png"), dpi=300)
    plt.close()

    # --- 3. Log-Bayes Factor ---
    plt.figure(figsize=(8, 6))
    plt.hist(y_log10_K[y_true==0], bins=50, alpha=0.6, color='C0', density=True, label='Class 0')
    plt.hist(y_log10_K[y_true==1], bins=50, alpha=0.6, color='C1', density=True, label='Class 1')
    plt.axvline(0, color='k', linestyle='--', alpha=0.5)
    plt.xlabel(r'$\log_{10} \mathcal{K}$')
    plt.title('Evidence Strength')
    plt.legend()
    plt.savefig(os.path.join(save_path, "3_bayes_factor_hist.png"), dpi=300)
    plt.close()

    # --- 4. Calibration Plot with Error Bars ---
    # Manual binning for error bars
    n_bins = 50
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bins) - 1
    
    bin_means = []
    bin_true = []
    bin_errors = []
    
    for i in range(n_bins):
        mask = bin_indices == i
        if np.any(mask):
            n_samples = np.sum(mask)
            p_mean = np.mean(y_prob[mask])
            fraction = np.mean(y_true[mask])
            
            # Binomial Standard Error: sqrt(p(1-p)/n)
            # Use max(fraction, 1e-6) to avoid zero error if p=0/1 purely by chance in small bins
            std_err = np.sqrt(fraction * (1 - fraction) / n_samples)
            
            bin_means.append(p_mean)
            bin_true.append(fraction)
            bin_errors.append(std_err)
    
    plt.figure(figsize=(7, 7))
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Ideal")
    
    # Scatter plot with Error Bars
    plt.errorbar(bin_means, bin_true, yerr=bin_errors, fmt='o', color='C2', 
                 ecolor='gray', elinewidth=1.5, capsize=3, alpha=0.8, label=f'MLP ({n_bins} bins)')

    # Add a small histogram at the bottom to show density
    plt.hist(y_prob, bins=n_bins, density=False, weights=np.ones_like(y_prob)*0.2/len(y_prob), 
             alpha=0.2, color='gray', bottom=0, label="Density")

    plt.xlabel("Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Reliability Diagram (with Binomial Error)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_path, "4_calibration_scatter.png"), dpi=300)
    plt.close()

    print(f"All plots saved to: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    args = parser.parse_args()
    
    evaluate(args)