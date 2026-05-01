#!/usr/bin/env python3
import numpy as np
import torch
import os
from tqdm import tqdm

def precompute_centers(grid_npy, out_pt, M=4, num_candidates=50, subbox_size=60.0):
    print(f"Opening {grid_npy} (Sequential Read)...")
    grids = np.load(grid_npy, mmap_mode='r')
    N = grids.shape[0]
    grid_res = 1000.0 / 128.0
    
    all_centers =[]
    # 顺序读取，机械硬盘的速度可以拉满到 150MB/s，完全不卡
    for i in tqdm(range(N), desc="Precomputing Centers"):
        grid = grids[i, 0]
        prob = grid - grid.min() + 1e-5
        prob_1d = (prob / prob.sum()).flatten()
        
        candidates = np.random.choice(len(prob_1d), size=num_candidates, p=prob_1d)
        
        centers =[]
        for idx_1d in candidates:
            z = idx_1d % 128
            y = (idx_1d // 128) % 128
            x = (idx_1d // (128*128)) % 128
            c = np.array([x, y, z]) * grid_res + (grid_res/2.0)
            
            if len(centers) > 0:
                dists = np.linalg.norm(np.array(centers) - c, axis=1)
                if np.min(dists) < subbox_size / 2.0:
                    continue
            centers.append(c)
            if len(centers) == M: break
            
        while len(centers) < M:
            centers.append(np.random.uniform(0, 1000.0, size=3))
            
        all_centers.append(np.array(centers, dtype=np.float32))
        
    torch.save(torch.tensor(np.array(all_centers)), out_pt)
    print(f"Saved to {out_pt} (Size: {os.path.getsize(out_pt)/1e6:.2f} MB)")

if __name__ == "__main__":
    base_dir = "/work/hdd/bdne/jdong8/fusion_data"
    precompute_centers(
        f"{base_dir}/grids/train_grids.npy", 
        f"{base_dir}/catalogs/train_centers.pt", 
        M=16, num_candidates=150, subbox_size=100.0
    )
    precompute_centers(
        f"{base_dir}/grids/val_grids.npy", 
        f"{base_dir}/catalogs/val_centers.pt", 
        M=16, num_candidates=150
    )