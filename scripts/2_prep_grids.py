#!/usr/bin/env python3
"""
[Step 2] Data Prep: 3D Grids (Corrected Version)
================================================
Fix: Uses np.lib.format.open_memmap to create valid .npy files
     while maintaining Zero-RAM usage.
"""

import os
import glob
import re
import numpy as np
import h5py
import torch
import argparse
from tqdm import tqdm
# 引入这个库来生成带标准头的 npy
from numpy.lib.format import open_memmap 

# ================================
# Config
# ================================
BOX_SIZE = 1000.0
GRID_SIZE = 128
SNAP_KEY = "0.666667"
HOD_RE = re.compile(r"hod(\d{5})", re.IGNORECASE)

# =====================================
# Helpers (保持不变)
# =====================================
def parse_lhid_seed(diag_path):
    m = HOD_RE.search(os.path.basename(diag_path))
    if not m: return -1, -1
    seed = int(m.group(1))
    
    parts = diag_path.split(os.sep)
    lhid = -1
    for i, p in enumerate(parts):
        if p == "diag" and i > 0:
            try: lhid = int(parts[i-1])
            except: pass
            break
    return lhid, seed

def get_snapshot_group(f, path):
    if SNAP_KEY in f: return f[SNAP_KEY]
    groups = [k for k in f.keys() if isinstance(f[k], h5py.Group)]
    return f[groups[0]] if len(groups) == 1 else None

def voxelize_positions(pos, box_size, grid_size):
    bins = np.linspace(0, box_size, grid_size + 1)
    hist, _ = np.histogramdd(pos, bins=(bins, bins, bins))
    
    mean_density = np.mean(hist)
    if mean_density > 0:
        delta = (hist / mean_density) - 1.0
    else:
        delta = hist
    return delta.astype(np.float32)

def build_lookup_table(dir_0, dir_1):
    print("Scanning raw files to build lookup table...")
    lookup = {}
    files = glob.glob(os.path.join(dir_0, "**", "galaxies", "hod*.h5"), recursive=True)
    files += glob.glob(os.path.join(dir_1, "**", "galaxies", "hod*.h5"), recursive=True)
    
    for fpath in tqdm(files, desc="Indexing"):
        l, s = parse_lhid_seed(fpath)
        if l != -1:
            lookup[(l, s)] = fpath
            
    print(f"Indexed {len(lookup)} files.")
    return lookup

# =====================================
# Core Logic (Modified)
# =====================================
def process_split(vector_pt_path, out_npy_path, lookup_table):
    if not os.path.exists(vector_pt_path):
        print(f"Skipping {vector_pt_path} (Not found)")
        return

    print(f"\nProcessing {os.path.basename(vector_pt_path)}...")
    
    # 1. Load Leader Data
    payload = torch.load(vector_pt_path, map_location="cpu")
    lhids = payload['lhid'].numpy()
    seeds = payload['seed'].numpy()
    N = len(lhids)
    
    print(f"  Target: {N} samples.")
    
    # 2. Create Valid .npy File directly on Disk
    # Shape: [N, 1, 64, 64, 64]
    shape = (N, 1, GRID_SIZE, GRID_SIZE, GRID_SIZE)
    
    # --- 关键修改 ---
    # 使用 open_memmap 替代 memmap
    # 这会创建一个带有正确 Header 的 .npy 文件，同时允许我们像数组一样写入
    fp = open_memmap(out_npy_path, mode='w+', dtype='float32', shape=shape)
    
    # 3. Fill it up
    missing_count = 0
    
    for i in tqdm(range(N), desc="Voxelizing"):
        key = (lhids[i], seeds[i])
        
        if key not in lookup_table:
            missing_count += 1
            continue
            
        diag_path = lookup_table[key]
        
        # Determine galaxy path
        if "/diag/galaxies/" in diag_path:
            gal_path = diag_path.replace("/diag/galaxies/", "/galaxies/")
        else:
            gal_path = diag_path.replace("/diag/", "/")
            
        try:
            with h5py.File(gal_path, "r") as f:
                g = get_snapshot_group(f, gal_path)
                pos = g["pos"][:] 
                
            delta = voxelize_positions(pos, BOX_SIZE, GRID_SIZE)
            
            # Write directly to disk
            fp[i, 0] = delta
            
        except Exception as e:
            # print(f"Error reading {gal_path}: {e}")
            missing_count += 1
            
    # Flush changes to disk
    fp.flush()
    print(f"  Saved {out_npy_path}. (Size: {os.path.getsize(out_npy_path)/1e9:.2f} GB)")
    if missing_count > 0:
        print(f"  WARNING: {missing_count} samples were missing raw files!")

# =====================================
# Main
# =====================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_0", type=str, required=True)
    parser.add_argument("--dir_1", type=str, required=True)
    parser.add_argument("--vector_dir", type=str, default="./data/vectors")
    parser.add_argument("--out_dir", type=str, default="./data/grids")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    lookup = build_lookup_table(args.dir_0, args.dir_1)
    
    process_split(
        os.path.join(args.vector_dir, "train_data.pt"),
        os.path.join(args.out_dir, "train_grids.npy"),
        lookup
    )
    
    process_split(
        os.path.join(args.vector_dir, "val_data.pt"),
        os.path.join(args.out_dir, "val_grids.npy"),
        lookup
    )
    
    print("\nDone! Grids are aligned with Vectors.")

if __name__ == "__main__":
    main()