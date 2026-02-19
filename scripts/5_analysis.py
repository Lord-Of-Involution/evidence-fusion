#!/usr/bin/env python3
"""
[Stage 5] Analysis & Visualization
==================================
Compares MLP vs. Fusion models on the Validation Set.
Generates:
1. Log Bayes Factor Distributions (Histogram)
2. ROC Curves (Performance)
3. Calibration Curves (Reliability)
"""

import os
import json 
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import roc_curve, auc, calibration_curve
from tqdm import tqdm

from evidence.models import EvidenceMLP, Small3DCNN, FusionEvidenceNetwork
from evidence.data import FusionDataset
from evidence.core import l_pop_transform, compute_posterior

# Beautify plots
sns.set_theme(style="whitegrid", font_scale=1.2)

def load_mlp(model_dir, device):
    print(f"Loading MLP from {model_dir}...")
    with open(os.path.join(model_dir, "mlp_config.json"), "r") as f:
        conf = json.load(f)
    model = EvidenceMLP(
        input_size=conf["input_size"],
        hidden_sizes=conf["hidden_sizes"],
        dropout=conf["dropout"]
    ).to(device)
    model.load_state_dict(torch.load(os.path.join(model_dir, "mlp_best.pt"), map_location=device))
    model.eval()
    return model

def load_fusion(model_dir, mlp_model, device):
    print(f"Loading Fusion from {model_dir}...")
    # Rebuild architecture
    cnn = Small3DCNN(input_channels=1, feature_dim=64)
    model = FusionEvidenceNetwork(mlp_model, cnn).to(device)
    model.load_state_dict(torch.load(os.path.join(model_dir, "fusion_best.pt"), map_location=device))
    model.eval()
    return model

def run_inference(model, loader, device, model_type="fusion"):
    """
    Returns:
        log_k: The Log Bayes Factor (ln K)
        probs: The posterior probability P(M1|x)
        labels: True labels
    """
    log_k_list = []
    probs_list = []
    labels_list = []
    
    with torch.no_grad():
        for (grid, vec), label in tqdm(loader, desc=f"Testing {model_type}"):
            grid, vec = grid.to(device), vec.to(device)
            
            if model_type == "mlp":
                f_x = model(vec)
            else:
                f_x = model(grid, vec)
            
            # 1. Compute Log Bayes Factor (ln K)
            # using alpha=2.0 as in training
            ln_k = l_pop_transform(f_x, alpha=2.0)
            
            # 2. Compute Posterior P(M1|x)
            post = compute_posterior(f_x)
            
            log_k_list.append(ln_k.cpu().numpy())
            probs_list.append(post.cpu().numpy())
            labels_list.append(label.numpy())
            
    return np.concatenate(log_k_list), np.concatenate(probs_list), np.concatenate(labels_list)

def plot_histograms(res_mlp, res_fusion, out_dir):
    """
    Replicates Figure 3 from Jeffrey & Wandelt.
    Shows distribution of ln(K) for M0 and M1 cases.
    """
    plt.figure(figsize=(12, 6))
    
    # Unpack
    k_mlp, _, y_mlp = res_mlp
    k_fus, _, y_fus = res_fusion
    
    # Plot M0 (Expect negative ln K) and M1 (Expect positive ln K)
    # MLP
    plt.hist(k_mlp[y_mlp==0], bins=50, alpha=0.3, color='blue', label='MLP (True M0)', density=True, range=(-20, 20))
    plt.hist(k_mlp[y_mlp==1], bins=50, alpha=0.3, color='orange', label='MLP (True M1)', density=True, range=(-20, 20))
    
    # Fusion (Step/Line plot to distinguish)
    plt.hist(k_fus[y_fus==0], bins=50, histtype='step', linewidth=2, color='darkblue', label='Fusion (True M0)', density=True, range=(-20, 20))
    plt.hist(k_fus[y_fus==1], bins=50, histtype='step', linewidth=2, color='darkorange', label='Fusion (True M1)', density=True, range=(-20, 20))
    
    plt.xlabel(r"Log Bayes Factor $\ln K$")
    plt.ylabel("Density")
    plt.title("Distribution of Evidence")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    path = os.path.join(out_dir, "analysis_histograms.png")
    plt.savefig(path, dpi=150)
    print(f"Saved {path}")
    plt.close()

def plot_roc(res_mlp, res_fusion, out_dir):
    """Standard ROC Curve comparison."""
    plt.figure(figsize=(8, 8))
    
    # MLP
    fpr, tpr, _ = roc_curve(res_mlp[2], res_mlp[1]) # Use Probabilities
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'MLP (AUC = {roc_auc:.3f})')
    
    # Fusion
    fpr_f, tpr_f, _ = roc_curve(res_fusion[2], res_fusion[1])
    roc_auc_f = auc(fpr_f, tpr_f)
    plt.plot(fpr_f, tpr_f, color='red', lw=2, label=f'Fusion (AUC = {roc_auc_f:.3f})')
    
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison')
    plt.legend(loc="lower right")
    
    path = os.path.join(out_dir, "analysis_roc.png")
    plt.savefig(path, dpi=150)
    print(f"Saved {path}")
    plt.close()

def plot_calibration(res_fusion, out_dir):
    """
    Checks if the predicted probability matches the true frequency.
    Paper claims Evidence Networks are well-calibrated.
    """
    plt.figure(figsize=(8, 8))
    
    prob_true, prob_pred = calibration_curve(res_fusion[2], res_fusion[1], n_bins=10)
    
    plt.plot(prob_pred, prob_true, marker='o', label='Fusion Model', color='red')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
    
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Plot (Reliability Diagram)')
    plt.legend()
    
    path = os.path.join(out_dir, "analysis_calibration.png")
    plt.savefig(path, dpi=150)
    print(f"Saved {path}")
    plt.close()

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    
    # 1. Load Data (Validation Set)
    print("Loading Validation Data...")
    val_ds = FusionDataset(
        vector_path=os.path.join(args.vector_dir, "val_data.pt"),
        grid_path=os.path.join(args.grid_dir, "val_grids.npy")
    )
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    
    # 2. Load Models
    mlp_model = load_mlp(args.mlp_dir, device)
    fusion_model = load_fusion(args.fusion_dir, mlp_model, device) # Re-uses MLP inside
    
    # 3. Inference
    res_mlp = run_inference(mlp_model, val_loader, device, "mlp")
    res_fusion = run_inference(fusion_model, val_loader, device, "fusion")
    
    # 4. Plotting
    print("\nGenerating Plots...")
    plot_histograms(res_mlp, res_fusion, args.out_dir)
    plot_roc(res_mlp, res_fusion, args.out_dir)
    plot_calibration(res_fusion, args.out_dir)
    
    print("\n Analysis Complete! Check your plots.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vector_dir", type=str, default="./data/vectors")
    parser.add_argument("--grid_dir", type=str, default="./data/grids")
    parser.add_argument("--mlp_dir", type=str, default="./models/mlp")
    parser.add_argument("--fusion_dir", type=str, default="./models/fusion")
    parser.add_argument("--out_dir", type=str, default="./plots")
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()
    main(args)