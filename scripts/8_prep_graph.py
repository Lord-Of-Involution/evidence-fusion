#!/usr/bin/env python3
"""
[Step 2.1] Data Prep: Point Cloud Catalogs (GNN Ready)
======================================================
This script replaces the old Voxelization process.
It extracts exact 3D coordinates, applies Redshift Space Distortion (RSD) 
to the Z-axis, and saves the point clouds into an efficient HDF5 file.

Output: 
    train_catalogs.h5 and val_catalogs.h5
    Structure:
    /0/pos_rsd  (N_gal, 3)
    /1/pos_rsd  (N_gal, 3)
    ...
    Indices directly match the vectors in train_data.pt!
"""

import os
import glob
import re
import numpy as np
import h5py
import torch
import argparse
from tqdm import tqdm
from astropy.cosmology import FlatLambdaCDM

# ================================
# Config & Physics
# ================================
BOX_SIZE = 1000.0
REDSHIFT = 0.5
SNAP_KEY = "0.666667"  # a = 1 / (1 + z) = 1 / 1.5 = 0.666667
HOD_RE = re.compile(r"hod(\d{5})", re.IGNORECASE)

# Quijote Fiducial Cosmology
QUIJOTE_COSMO = FlatLambdaCDM(H0=67.11, Om0=0.3175)

# =====================================
# Physics Helper: Apply RSD
# =====================================
def apply_rsd(pos, vel, redshift, cosmo, box_size=1000.0, axis=2):
    """
    Manually displaces galaxy positions along the line-of-sight (Z-axis) 
    using their peculiar velocities, mimicking Redshift Space.
    """
    a = 1.0 / (1.0 + redshift)
    H_a = cosmo.H(redshift).value  # in km/s/Mpc
    
    # dz = v_z / (a * H(a))
    dz = vel[:, axis] / (a * H_a)
    
    pos_rsd = pos.copy()
    pos_rsd[:, axis] += dz
    
    # Apply periodic boundary conditions
    pos_rsd[:, axis] %= box_size
    
    return pos_rsd.astype(np.float32)

# =====================================
# Helpers
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
    groups =[k for k in f.keys() if isinstance(f[k], h5py.Group)]
    return f[groups[0]] if len(groups) == 1 else None

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
# Core Logic
# =====================================
def process_split(vector_pt_path, out_h5_path, lookup_table):
    if not os.path.exists(vector_pt_path):
        print(f"Skipping {vector_pt_path} (Not found)")
        return

    print(f"\nProcessing {os.path.basename(vector_pt_path)}...")
    
    # 1. Load Vector Data (To ensure absolute perfect alignment)
    payload = torch.load(vector_pt_path, map_location="cpu")
    lhids = payload['lhid'].numpy()
    seeds = payload['seed'].numpy()
    N = len(lhids)
    
    print(f"  Target: {N} samples.")
    
    missing_count = 0
    
    # 2. Open Output HDF5
    with h5py.File(out_h5_path, 'w') as out_f:
        for i in tqdm(range(N), desc="Extracting Catalogs"):
            key = (lhids[i], seeds[i])
            
            if key not in lookup_table:
                missing_count += 1
                continue
                
            diag_path = lookup_table[key]
            
            # Navigate to galaxy path (Real Space data)
            if "/diag/galaxies/" in diag_path:
                gal_path = diag_path.replace("/diag/galaxies/", "/galaxies/")
            else:
                gal_path = diag_path.replace("/diag/", "/")
                
            try:
                with h5py.File(gal_path, "r") as f:
                    g = get_snapshot_group(f, gal_path)
                    pos = g["pos"][:]
                    vel = g["vel"][:]
                    # 如果有 gal_type (central/satellite)，也可以一并提取
                    # gal_type = g["gal_type"][:]
                    
                # --- 核心物理操作：还原红移空间畸变 (RSD) ---
                pos_rsd = apply_rsd(pos, vel, redshift=REDSHIFT, cosmo=QUIJOTE_COSMO)
                
                # 3. Save to HDF5 under group matching the vector index `i`
                grp = out_f.create_group(str(i))
                grp.create_dataset("pos_rsd", data=pos_rsd, compression="lzf")
                
                # 可选：把原始速度也存下来备查
                # grp.create_dataset("vel", data=vel, compression="lzf")
                
            except Exception as e:
                # print(f"Error reading {gal_path}: {e}")
                missing_count += 1
                
    print(f"  Saved {out_h5_path}. (Size: {os.path.getsize(out_h5_path)/1e6:.2f} MB)")
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
    parser.add_argument("--out_dir", type=str, default="./data/catalogs")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    lookup = build_lookup_table(args.dir_0, args.dir_1)
    
    process_split(
        os.path.join(args.vector_dir, "train_data.pt"),
        os.path.join(args.out_dir, "train_catalogs.h5"),
        lookup
    )
    
    process_split(
        os.path.join(args.vector_dir, "val_data.pt"),
        os.path.join(args.out_dir, "val_catalogs.h5"),
        lookup
    )
    
    print("\nDone! Catalogs (Redshift Space) are now aligned with Vectors.")

if __name__ == "__main__":
    main()