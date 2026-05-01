#!/usr/bin/env python3
import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch

# 美化绘图
try:
    import seaborn as sns
    sns.set_theme(style="whitegrid", font_scale=1.2)
except ImportError:
    pass

def l_pop_transform(x, alpha=2.0):
    """J(x) = x + x * |x|^(alpha-1)"""
    return x + x * torch.abs(x).pow(alpha - 1)

def compute_probability(f_x, alpha=2.0):
    """p = Sigmoid( J(f(x)) )"""
    J_val = l_pop_transform(f_x, alpha)
    return torch.sigmoid(J_val)

def main(args):
    print(f"Loading cached evidence from {args.fusion_dir}...")
    
    # 1. 读取 100% 对齐的证据缓存
    cache_path = os.path.join(args.fusion_dir, "sync_evidence.pt")
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Cannot find {cache_path}! Did 10_extract_sync.py run successfully?")
    
    cache = torch.load(cache_path, map_location='cpu')
    val_mlp = cache["val"]["f_mlp"]
    val_gnn = cache["val"]["f_gnn"]
    val_y = cache["val"]["y"]

    # 2. 读取最优融合参数
    params_path = os.path.join(args.fusion_dir, "optimal_fusion_params.json")
    if os.path.exists(params_path):
        with open(params_path, "r") as f:
            params = json.load(f)
        temp = params.get("temp", 1.0)
        gamma = params.get("gamma", 0.0)
    else:
        print("Warning: optimal_fusion_params.json not found, using Temp=1.0, Gamma=0.0")
        temp, gamma = 1.0, 0.0

    print(f"Using Parameters -> Temp: {temp:.4f}, Gamma: {gamma:.4f}")

    # 3. 计算最终的 Scalar Evidence
    f_fus = (val_mlp / temp) + (gamma * val_gnn)

    # 4. 转化为概率
    P_mlp = compute_probability(val_mlp).numpy().flatten()
    P_gnn = compute_probability(val_gnn).numpy().flatten() 
    P_fus = compute_probability(f_fus).numpy().flatten()
    Y_np  = val_y.numpy().flatten()

    # 5. 计算指标
    acc_mlp = np.mean((P_mlp > 0.5) == Y_np)
    acc_gnn = np.mean((P_gnn > 0.5) == Y_np)
    acc_fus = np.mean((P_fus > 0.5) == Y_np)

    print(f"\nRESULTS ON VALIDATION SET:")
    print(f"  GNN Solo Acc : {acc_gnn:.4f}")
    print(f"  MLP Solo Acc : {acc_mlp:.4f}")
    print(f"  Fusion Acc   : {acc_fus:.4f}")

    os.makedirs(args.out_dir, exist_ok=True)
# ==========================================
    # 5.5 新增图: MLP-Only Histogram (纯宏观频谱的证据分布 - 铺垫用)
    # ==========================================
    os.makedirs(args.out_dir, exist_ok=True)
    print("\nGenerating MLP-Only Histogram Plot...")
    
    plt.figure(figsize=(10, 6), dpi=150)
    mask0 = (Y_np == 0) # True Class 0 (Unbiased)
    mask1 = (Y_np == 1) # True Class 1 (Biased)
    bins = np.linspace(0, 1, 50)
    
    # 稍微调高了透明度 (alpha=0.5) 让单图看起来更饱满
    plt.hist(P_mlp[mask0], bins=bins, density=True, alpha=0.5, color='blue', label='Spectra-MLP (True 0: Fiducial)')
    plt.hist(P_mlp[mask1], bins=bins, density=True, alpha=0.5, color='orange', label='Spectra-MLP (True 1: Biased)')

    plt.title(f"Macroscopic Spectra Evidence Distribution\nAccuracy: {acc_mlp:.3f}")
    plt.xlabel("Probability $P(M_1 | x)$")
    plt.ylabel("Density")
    plt.legend(loc='upper center')
    plt.grid(True, alpha=0.3)
    
    out_mlp_hist = os.path.join(args.out_dir, "mlp_only_hist.png")
    plt.savefig(out_mlp_hist, bbox_inches='tight')
    plt.close()
    print(f"Saved MLP-Only Histogram to {out_mlp_hist}")
    # ==========================================
    # 6. 图 1: Histogram (直方图)
    # ==========================================
    print("\nGenerating Histogram Plot...")
    plt.figure(figsize=(10, 6), dpi=150)
    mask0 = (Y_np == 0) # True Class 0 (Unbiased)
    mask1 = (Y_np == 1) # True Class 1 (Biased)
    bins = np.linspace(0, 1, 50)
    
    plt.hist(P_mlp[mask0], bins=bins, density=True, alpha=0.3, color='blue', label='MLP (True 0)')
    plt.hist(P_mlp[mask1], bins=bins, density=True, alpha=0.3, color='orange', label='MLP (True 1)')
    plt.hist(P_fus[mask0], bins=bins, density=True, histtype='step', linewidth=2.5, color='darkblue', label='Fusion (True 0)')
    plt.hist(P_fus[mask1], bins=bins, density=True, histtype='step', linewidth=2.5, color='darkorange', label='Fusion (True 1)')

    plt.title(f"Evidence Network Distribution\nMLP Acc: {acc_mlp:.3f} | Fusion Acc: {acc_fus:.3f}")
    plt.xlabel("Probability $P(M_1 | x)$")
    plt.ylabel("Density")
    plt.legend(loc='upper center')
    plt.grid(True, alpha=0.3)
    out_hist = os.path.join(args.out_dir, "bayesian_fusion_hist.png")
    plt.savefig(out_hist, bbox_inches='tight')
    plt.close()

    # ==========================================
    # 准备随机抽样索引 (共用，确保两张散点图点的位置严格一一对应)
    # ==========================================
    total_samples = len(Y_np)
    limit = min(1000, total_samples)
    np.random.seed(42)
    random_idx = np.random.choice(total_samples, size=limit, replace=False)
    Y_sample = Y_np[random_idx]
    mask0_sample = (Y_sample == 0)
    mask1_sample = (Y_sample == 1)

    # ==========================================
    # 7. 图 2: 概率空间散点图 (Probability Space)
    # ==========================================
    print("Generating Probability Space Scatter Plot...")
    plt.figure(figsize=(8, 8), dpi=150)
    
    P_mlp_sample = P_mlp[random_idx]
    P_gnn_sample = P_gnn[random_idx]

    plt.scatter(P_mlp_sample[mask0_sample], P_gnn_sample[mask0_sample], color='blue', alpha=0.5, edgecolor='none', label='True 0 (Fiducial)')
    plt.scatter(P_mlp_sample[mask1_sample], P_gnn_sample[mask1_sample], color='orange', alpha=0.5, edgecolor='none', label='True 1 (Biased)')
    
    plt.axvline(0.5, color='black', linestyle='-', linewidth=1.2, alpha=0.4)
    plt.axhline(0.5, color='black', linestyle='-', linewidth=1.2, alpha=0.4)
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)

    if abs(gamma) > 1e-5:
        f_grid = np.linspace(-10, 10, 1000) # 缩小网格范围防止平移横线
        FX, FY = np.meshgrid(f_grid, f_grid)
        F_FUS = (FX / temp) + (gamma * FY)
        p_axis = compute_probability(torch.tensor(f_grid)).numpy()
        plt.contour(p_axis, p_axis, F_FUS, levels=[0.0], colors=['red'], linestyles=['--'], linewidths=[2.5])
        plt.plot([],[], color='red', linestyle='--', linewidth=2.5, label='Fusion Decision Boundary')
    else:
        plt.axvline(0.5, color='red', linestyle='--', linewidth=2.5, label='Decision Boundary (Pure MLP)')

    plt.xlabel(r"MLP Probability $P_{MLP}(M_1 | x)$")
    plt.ylabel(r"GNN Probability $P_{GNN}(M_1 | x)$")
    plt.title("Bayesian Fusion in Probability Space")
    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    out_prob_scatter = os.path.join(args.out_dir, "bayesian_geometry_prob_scatter.png")
    plt.savefig(out_prob_scatter, bbox_inches='tight')
    plt.close()

    # ==========================================
    # 8. 图 3: 对数证据空间散点图 (Log-Evidence Space)
    # ==========================================
    print("Generating Log-Evidence Space Scatter Plot...")
    plt.figure(figsize=(8, 8), dpi=150)
    
    f_mlp_sample = val_mlp.numpy().flatten()[random_idx]
    f_gnn_sample = val_gnn.numpy().flatten()[random_idx]

    plt.scatter(f_mlp_sample[mask0_sample], f_gnn_sample[mask0_sample], color='blue', alpha=0.5, edgecolor='none', label='True 0 (Fiducial)')
    plt.scatter(f_mlp_sample[mask1_sample], f_gnn_sample[mask1_sample], color='orange', alpha=0.5, edgecolor='none', label='True 1 (Biased)')
    
    plt.axvline(0, color='black', linestyle='-', linewidth=1.2, alpha=0.4)
    plt.axhline(0, color='black', linestyle='-', linewidth=1.2, alpha=0.4)

    # 强行画一个正方形的绝对坐标系
    global_min = min(f_mlp_sample.min(), f_gnn_sample.min()) - 0.5
    global_max = max(f_mlp_sample.max(), f_gnn_sample.max()) + 0.5
    x_line = np.linspace(global_min, global_max, 100)

    if abs(gamma) > 1e-5:
        # 【加回来的灵魂】：融合投影轴 (Fusion Axis)，它完美垂直于决策边界！
        m_axis = gamma * temp
        y_axis = m_axis * x_line
        mask_axis = (y_axis >= global_min) & (y_axis <= global_max)
        plt.plot(x_line[mask_axis], y_axis[mask_axis], color='red', linestyle='-', linewidth=2, alpha=0.8, label=f'Fusion Axis (Slope: {m_axis:.3f})')
        
        # 决策边界 (Decision Boundary): 垂直于投影轴的零点分割线
        m_bound = -1.0 / (gamma * temp)
        y_bound = m_bound * x_line
        mask_bound = (y_bound >= global_min) & (y_bound <= global_max)
        plt.plot(x_line[mask_bound], y_bound[mask_bound], color='red', linestyle='--', linewidth=2.5, label=f'Decision Boundary ($f_{{fus}}=0$)')
    else:
        plt.axvline(0, color='red', linestyle='--', linewidth=2.5, label='Decision Boundary (Pure MLP)')
    plt.xlim(global_min, global_max)
    #plt.ylim(global_min, global_max)
    plt.ylim(-2, 4)
    #plt.gca().set_aspect('equal', adjustable='box')

    plt.xlabel(r"MLP Log-Evidence $f_{MLP}$")
    plt.ylabel(r"GNN Log-Evidence $f_{GNN}$")
    plt.title("Orthogonality Check: Log-Evidence Space (1:1 Scale)")
    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    
    out_log_scatter = os.path.join(args.out_dir, "bayesian_geometry_log_scatter.png")
    plt.savefig(out_log_scatter, bbox_inches='tight')
    plt.close()
    
    print("All plots generated successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fusion_dir", type=str, default="/work/hdd/bdne/jdong8/fusion_models/ultimate_fusion")
    parser.add_argument("--out_dir", type=str, default="/work/hdd/bdne/jdong8/fusion_models/ultimate_fusion/plots")
    args = parser.parse_args()
    main(args)