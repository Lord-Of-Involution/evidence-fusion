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
    P_gnn = compute_probability(val_gnn).numpy().flatten() # 顺便看看 GNN 单独的表现
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

    # ==========================================
    # 6. 开始画图
    # ==========================================
    os.makedirs(args.out_dir, exist_ok=True)
    print("\nGenerating Histogram Plot...")
    
    plt.figure(figsize=(10, 6), dpi=150)
    
    mask0 = (Y_np == 0) # True Class 0 (Unbiased)
    mask1 = (Y_np == 1) # True Class 1 (Biased)
    
    bins = np.linspace(0, 1, 50)
    
    # 绘制 MLP 的阴影图
    plt.hist(P_mlp[mask0], bins=bins, density=True, alpha=0.3, color='blue', label='MLP (True 0)')
    plt.hist(P_mlp[mask1], bins=bins, density=True, alpha=0.3, color='orange', label='MLP (True 1)')
    
    # 绘制 Fusion 的阶梯图
    plt.hist(P_fus[mask0], bins=bins, density=True, histtype='step', linewidth=2.5, color='darkblue', label='Fusion (True 0)')
    plt.hist(P_fus[mask1], bins=bins, density=True, histtype='step', linewidth=2.5, color='darkorange', label='Fusion (True 1)')

    plt.title(f"Evidence Network Distribution\nMLP Acc: {acc_mlp:.3f} | Fusion Acc: {acc_fus:.3f}")
    plt.xlabel("Probability $P(M_1 | x)$")
    plt.ylabel("Density")
    plt.legend(loc='upper center')
    plt.grid(True, alpha=0.3)
    
    out_path = os.path.join(args.out_dir, "bayesian_fusion_hist.png")
    plt.savefig(out_path, bbox_inches='tight')
    print(f"Saved plot to {out_path}")

# ==========================================
    # 额外赠送：散点图 (横纵 1:1 绝对同比例 + 决策边界的几何证明)
    # ==========================================
    plt.figure(figsize=(8, 8), dpi=150)
    
    # 取前 1500 个样本画散点
    limit = min(1500, len(Y_np))
    x_val_0 = val_mlp.numpy().flatten()[:limit][mask0[:limit]]
    y_val_0 = val_gnn.numpy().flatten()[:limit][mask0[:limit]]
    x_val_1 = val_mlp.numpy().flatten()[:limit][mask1[:limit]]
    y_val_1 = val_gnn.numpy().flatten()[:limit][mask1[:limit]]

    plt.scatter(x_val_0, y_val_0, color='blue', alpha=0.5, edgecolor='none', label='True 0 (Unbiased)')
    plt.scatter(x_val_1, y_val_1, color='orange', alpha=0.5, edgecolor='none', label='True 1 (Biased)')
    
    # 画出原始的 MLP/GNN 零点准星
    plt.axvline(0, color='black', linestyle='-', linewidth=1.2, alpha=0.4)
    plt.axhline(0, color='black', linestyle='-', linewidth=1.2, alpha=0.4)

    # 找到全局的最大最小值，强行画一个正方形的绝对坐标系
    all_x = np.concatenate([x_val_0, x_val_1])
    all_y = np.concatenate([y_val_0, y_val_1])
    global_min = min(all_x.min(), all_y.min()) - 0.5
    global_max = max(all_x.max(), all_y.max()) + 0.5

    x_line = np.linspace(global_min, global_max, 100)

# ==========================================
    # 【高能预警】：用数学绘制贝叶斯融合的几何真理 (完美支持负 Gamma！)
    # ==========================================
    # 用绝对值判断，防止把负 gamma 漏掉！
    if abs(gamma) > 1e-5:
        # 1. 融合投影轴 (Fusion Axis)
        m_axis = gamma * temp
        plt.plot(x_line, m_axis * x_line, color='red', linestyle='-', linewidth=2, alpha=0.8, 
                 label=f'Fusion Axis (Slope: {m_axis:.3f})')
        
        # 2. 决策边界 (Decision Boundary): 垂直于投影轴的零点分割线
        m_bound = -1.0 / (gamma * temp)
        
        # 掩码控制，防止斜线画出正方形画布，导致坐标系被暴力拉扯变形
        y_bound = m_bound * x_line
        mask_bound = (y_bound >= global_min) & (y_bound <= global_max)
        
        plt.plot(x_line[mask_bound], y_bound[mask_bound], color='red', linestyle='--', linewidth=2.5, 
                 label=f'Decision Boundary ($f_{{fus}}=0$)')
    else:
        # 只有在 Gamma 严格为 0 时，才退化为纯 MLP
        plt.axvline(0, color='red', linestyle='--', linewidth=2.5, label='Decision Boundary (Pure MLP)')

    # 强制让 X 轴和 Y 轴的视野完全一样，避免视觉拉伸！
    plt.xlim(global_min, global_max)
    plt.ylim(global_min, global_max)
    plt.gca().set_aspect('equal', adjustable='box')

    plt.xlabel(r"MLP Log-Evidence $f_{MLP}$")
    plt.ylabel(r"GNN Log-Evidence $f_{GNN}$")
    plt.title("Orthogonality Check: Bayesian Fusion Geometry\n(1:1 True Physical Scale)")
    
    # 将图例放在外面防止遮挡散点
    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    
    out_scatter = os.path.join(args.out_dir, "bayesian_geometry_scatter.png")
    plt.savefig(out_scatter, bbox_inches='tight')
    print(f"Saved Bayesian Geometry plot to {out_scatter}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fusion_dir", type=str, default="/work/hdd/bdne/jdong8/fusion_models/ultimate_fusion")
    parser.add_argument("--out_dir", type=str, default="/work/hdd/bdne/jdong8/fusion_models/ultimate_fusion/plots")
    args = parser.parse_args()
    main(args)