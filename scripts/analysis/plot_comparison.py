#!/usr/bin/env python3
"""
Ultimate Bayesian Synergy Plotter
=================================
Visualizes the Pure Conditional Evidence J_Cond.
Enforces 1:1 geometric aspect ratio and asymmetric 1% tail trimming.
"""

import os
import argparse
import math
import numpy as np
import matplotlib.pyplot as plt
import torch

try:
    import seaborn as sns
    sns.set_theme(style="whitegrid", font_scale=1.2)
except ImportError:
    pass

def l_pop_transform(x, alpha=2.0):
    return x + x * torch.abs(x).pow(alpha - 1)

def compute_probability(J_val, c_prior=0.0):
    return torch.sigmoid(J_val + c_prior)

def get_asymmetric_bounds(data_array, trim_percent=1.0):
    """Trims the extreme top and bottom tails exactly where they lie."""
    return np.percentile(data_array, [trim_percent, 100.0 - trim_percent])

def main(args):
    print(">>> Engaging Synergistic Geometry Reconstruction...")
    
    mlp_path = os.path.join(args.mlp_dir, "mlp_evidence.pt")
    fus_path = os.path.join(args.fus_dir, "gnn_evidence.pt")
    
    mlp_cache = torch.load(mlp_path, map_location='cpu', weights_only=False)
    fus_cache = torch.load(fus_path, map_location='cpu', weights_only=False)

    val_mlp_raw = mlp_cache["val"]["f_mlp"]
    val_fus_raw = fus_cache["val"]["f_gnn"]  
    val_y = fus_cache["val"]["y"]

    assert torch.all(mlp_cache["val"]["y"].flatten() == val_y.flatten()), "Tags misaligned!"

    tr_y = fus_cache["train"]["y"]
    n1 = tr_y.sum().item()
    n0 = len(tr_y) - n1
    c_prior = math.log(n1 / max(1, n0))

    J_mlp = l_pop_transform(val_mlp_raw, alpha=2.0)
    J_fus = l_pop_transform(val_fus_raw, alpha=2.0)
    J_cond = J_fus - J_mlp

    P_mlp = compute_probability(J_mlp, c_prior).numpy().flatten()
    P_fus = compute_probability(J_fus, c_prior).numpy().flatten()
    Y_np  = val_y.numpy().flatten()

    acc_mlp = np.mean((P_mlp > 0.5) == Y_np)
    acc_fus = np.mean((P_fus > 0.5) == Y_np)

    os.makedirs(args.out_dir, exist_ok=True)
    mask0 = (Y_np == 0) 
    mask1 = (Y_np == 1) 

    # ==========================================================================
    # Plot 1: Evidence Network Posterior Distribution (P-Space)
    # ==========================================================================
    plt.figure(figsize=(10, 6), dpi=150)
    bins_p = np.linspace(0, 1, 50)
    
    plt.hist(P_mlp[mask0], bins=bins_p, density=True, alpha=0.3, color='dodgerblue', label='Spectra-MLP (True 0)')
    plt.hist(P_mlp[mask1], bins=bins_p, density=True, alpha=0.3, color='crimson', label='Spectra-MLP (True 1)')
    plt.hist(P_fus[mask0], bins=bins_p, density=True, histtype='step', linewidth=2.5, color='blue', label='Fusion GNN (True 0)')
    plt.hist(P_fus[mask1], bins=bins_p, density=True, histtype='step', linewidth=2.5, color='darkred', label='Fusion GNN (True 1)')

    plt.title(f"Evidence Network Posterior Distribution\nBaseline Acc: {acc_mlp:.3f} $\\rightarrow$ Fusion Acc: {acc_fus:.3f}", fontweight='bold', pad=15)
    plt.xlabel("Probability of Biased Universe $P(M_1 \mid x)$", labelpad=10)
    plt.ylabel("Density", labelpad=10)
    plt.legend(loc='upper center')
    
    plt.savefig(os.path.join(args.out_dir, "fig_1_posterior_histogram.png"), bbox_inches='tight')
    plt.close()

    # ==========================================================================
    # Plot 2: Conditional Evidence Separation Histogram (Asymmetric Trim)
    # ==========================================================================
    plt.figure(figsize=(10, 6), dpi=150)
    
    # Asymmetric 1% trimming
    J_cond_np = J_cond.numpy().flatten()
    c_min, c_max = get_asymmetric_bounds(J_cond_np, trim_percent=1.0)
    safe_bins = np.linspace(c_min, c_max, 60)
    
    plt.hist(J_cond_np[mask0], bins=safe_bins, density=True, alpha=0.6, color='dodgerblue', label='True 0 (Fiducial)')
    plt.hist(J_cond_np[mask1], bins=safe_bins, density=True, alpha=0.6, color='crimson', label='True 1 (Biased)')
    plt.axvline(0, color='black', linestyle='--', linewidth=2, alpha=0.7, label='Zero Conditional Info')

    plt.title(f"Isolated Conditional Evidence: $\mathcal{{J}}_{{Cond}}(\mathrm{{dots}} \mid P_k)$\n(Tails trimmed 1%)", fontweight='bold', pad=15)
    plt.xlabel("Conditional Log-Bayes Factor", labelpad=10)
    plt.ylabel("Density", labelpad=10)
    plt.legend()
    
    plt.savefig(os.path.join(args.out_dir, "fig_2_conditional_histogram.png"), bbox_inches='tight')
    plt.close()

    # ==========================================================================
    # Plot 3: J-Space Orthogonality Scatter (Strict 1:1 Aspect Ratio)
    # ==========================================================================
    total_samples = len(Y_np)
    limit_plot = min(1500, total_samples)
    np.random.seed(42)
    random_idx = np.random.choice(total_samples, size=limit_plot, replace=False)
    
    J_mlp_sample = J_mlp.numpy().flatten()[random_idx]
    J_cond_sample = J_cond.numpy().flatten()[random_idx]
    Y_sample = Y_np[random_idx]
    
    m0_samp = (Y_sample == 0)
    m1_samp = (Y_sample == 1)

    plt.figure(figsize=(9, 9), dpi=150)
    
    plt.scatter(J_mlp_sample[m0_samp], J_cond_sample[m0_samp], color='dodgerblue', alpha=0.6, edgecolor='white', linewidth=0.3, label='True 0 (Fiducial)')
    plt.scatter(J_mlp_sample[m1_samp], J_cond_sample[m1_samp], color='crimson', alpha=0.6, edgecolor='white', linewidth=0.3, label='True 1 (Biased)')
    
    plt.axvline(0, color='grey', linestyle='-', linewidth=1.0, alpha=0.5)
    plt.axhline(0, color='grey', linestyle='-', linewidth=1.0, alpha=0.5)

    # Calculate absolute global limits to force a perfect square bounding box
    min_x, max_x = get_asymmetric_bounds(J_mlp_sample, trim_percent=1.0)
    min_y, max_y = get_asymmetric_bounds(J_cond_sample, trim_percent=1.0)
    
    global_min = min(min_x, min_y) - 1.0
    global_max = max(max_x, max_y) + 1.0

    x_line = np.linspace(global_min, global_max, 500)
    y_bound = -x_line - c_prior
    plt.plot(x_line, y_bound, color='black', linestyle='--', linewidth=3.0, alpha=0.9, label=f'Decision Boundary ($P=0.5$)')

    # Force strict limits and aspect ratio
    plt.xlim(global_min, global_max)
    plt.ylim(global_min, global_max)
    plt.gca().set_aspect('equal', adjustable='box')

    plt.xlabel(r"Macroscopic Evidence $\mathcal{J}_{MLP}(P_k)$", labelpad=10, fontsize=13)
    plt.ylabel(r"Conditional Micro-Topology Evidence $\mathcal{J}_{Cond}(\mathrm{dots} \mid P_k)$", labelpad=10, fontsize=13)
    plt.title(f"Information Orthogonality in Bayesian $\mathcal{{J}}$-Space\n(Strict 1:1 Geometrical Scale)", fontweight='bold', pad=15)
    plt.legend(loc='upper right', frameon=True, shadow=True)
    
    plt.savefig(os.path.join(args.out_dir, "fig_3_j_space_orthogonality.png"), bbox_inches='tight')
    plt.close()
    
    print(f">>> Publication plots saved to {args.out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlp_dir", type=str, default="/work/hdd/bdne/jdong8/fusion_models/abacus_mlp_solo/")
    parser.add_argument("--fus_dir", type=str, default="/work/hdd/bdne/jdong8/fusion_models/abacus_early_fusion/")
    parser.add_argument("--out_dir", type=str, default="/projects/bdne/jdong8/src/evidence-fusion/plots")
    args = parser.parse_args()
    main(args)