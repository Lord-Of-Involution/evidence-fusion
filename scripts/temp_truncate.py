#!/usr/bin/env python3
"""
临时急救脚本：从已有的 train_data.pt 和 val_data.pt 中截断高 k 噪声
维度变化：800 -> 400
结构保留：P0(0-99), P2(200-299), P4(400-499), Qk(600-699)
"""

import os
import torch
import shutil

def truncate_file(in_path, out_path):
    print(f"\nProcessing: {in_path}")
    if not os.path.exists(in_path):
        print(f"  [ERROR] File not found: {in_path}")
        return

    # 1. 载入原始数据
    payload = torch.load(in_path, map_location='cpu')
    
    # 兼容各种历史 key
    data_key = 'data' if 'data' in payload else ('stats' if 'stats' in payload else 'data_stats')
    x = payload[data_key]
    
    # 探测是否是转置形状 [800, N]，并统一为 [N, Features]
    is_transposed = False
    if x.shape[0] == 800 and x.shape[1] > 800:
        x = x.T
        is_transposed = True
        
    print(f"  Original Data Shape: {x.shape}")

    # 2. 核心操作：构建截断索引！
    # 原始 800 维: 0-199(P0), 200-399(P2), 400-599(P4), 600-799(Qk)
    # 截断后 400 维: 只保留每个谱的前 100 个低 k 信号
    idx_p0 = list(range(0, 100))
    idx_p2 = list(range(200, 300))
    idx_p4 = list(range(400, 500))
    idx_qk = list(range(600, 700))
    
    keep_indices = torch.tensor(idx_p0 + idx_p2 + idx_p4 + idx_qk, dtype=torch.long)

    # 3. 执行切片
    x_new = x[:, keep_indices]

    # 如果原数据是转置的，还原回去
    if is_transposed:
        x_new = x_new.T
        
    payload[data_key] = x_new

    # 4. 同步更新 Metadata 里的 Normalizer 参数 (如果有的话)
    for scaler_key in ['scaler_mean', 'scaler_std']:
        if scaler_key in payload:
            scaler_val = payload[scaler_key]
            # 可能是 [1, 800] 也可能是 [800]
            if scaler_val.ndim == 2:
                payload[scaler_key] = scaler_val[:, keep_indices]
            else:
                payload[scaler_key] = scaler_val[keep_indices]

    # 5. 保存
    torch.save(payload, out_path)
    print(f"  [SUCCESS] Saved to {out_path} | New Shape: {x_new.shape}")

if __name__ == "__main__":
    # 我们用一个新的文件夹存，免得把原来的搞坏了
    base_dir = "/work/hdd/bdne/jdong8/fusion_data"
    old_vector_dir = os.path.join(base_dir, "vectors")
    new_vector_dir = os.path.join(base_dir, "vectors_400")
    
    os.makedirs(new_vector_dir, exist_ok=True)
    
    train_in = os.path.join(old_vector_dir, "train_data.pt")
    val_in = os.path.join(old_vector_dir, "val_data.pt")
    
    train_out = os.path.join(new_vector_dir, "train_data.pt")
    val_out = os.path.join(new_vector_dir, "val_data.pt")
    
    truncate_file(train_in, train_out)
    truncate_file(val_in, val_out)
    
    print("\n>>> All done! Please point your training scripts to the new directory:")
    print(f">>> {new_vector_dir}")