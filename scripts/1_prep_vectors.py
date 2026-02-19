#!/usr/bin/env python3
"""
[Step 1] Data Prep: Vector Summaries (Pk + Qk)
==============================================
Strategy: Self-Normalization + Global Standardization

Workflow:
1. Load Raw H5 (P0, P2, P4, Qk/Bk).
2. PHYSICS NORM (Row-wise):
   - Calculate scale S_i = mean(P0_i) for each universe.
   - Divide ALL components (P0, P2, P4, Qk) by S_i.
   - This removes the amplitude (Linear Bias) information while preserving
     the shape and the relative ratios (RSD) between multipoles.
3. Split Train/Val (Stratified).
4. STATS NORM (Column-wise):
   - Compute Mean/Std on TRAIN set only.
   - Apply (X - Mu) / Sigma to both Train and Val.
5. Save .pt files ready for the neural network.
"""

import os
import glob
import re
import numpy as np
import h5py
import torch
import argparse
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# ================================
# Config
# ================================
TARGET_K = 200          # 200 bins per summary
SNAP_KEY = "0.666667"
HOD_RE = re.compile(r"hod(\d{5})", re.IGNORECASE)

# Parameters to extract for downstream analysis
BIAS_PARAM_KEYS = [
    "mean_occupation_centrals_assembias_param1",
    "mean_occupation_satellites_assembias_param1",
    "eta_vb_centrals",
    "eta_vb_satellites",
    "conc_gal_bias_satellites"
]

# =====================================
# Helpers
# =====================================
def parse_lhid_seed(diag_path):
    """Extract LHID and Seed from filename/path."""
    m = HOD_RE.search(os.path.basename(diag_path))
    if not m: return -1, -1
    seed = int(m.group(1))
    
    # Try to parse LHID from path structure
    parts = diag_path.split(os.sep)
    lhid = -1
    for i, p in enumerate(parts):
        if p == "diag" and i > 0:
            try:
                lhid = int(parts[i-1])
            except: pass
            break
    return lhid, seed

def _get_snapshot_group(f, path):
    if SNAP_KEY in f: return f[SNAP_KEY]
    groups = [k for k in f.keys() if isinstance(f[k], h5py.Group)]
    return f[groups[0]] if len(groups) == 1 else None

def _load_spectra(g):
    """
    Loads P0, P2, P4 (Monopole, Quadrupole, Hexadecapole)
    and Qk (Reduced Bispectrum or Bispectrum).
    """
    # 1. Load Pk (Multipoles)
    if "zPk" not in g: return None, None, None, None
    zpk = np.asarray(g["zPk"][:])
    P = zpk.T if zpk.shape[1] == 3 else zpk # Ensure shape (3, Nk)
    
    # Safety check for bin count
    if P.shape[1] < TARGET_K: return None, None, None, None
    
    P0 = P[0, :TARGET_K].astype(np.float32)
    P2 = P[1, :TARGET_K].astype(np.float32)
    P4 = P[2, :TARGET_K].astype(np.float32)
    
    # 2. Load Qk (Bispectrum)
    # Note: Check if your key is 'zQk' or 'zBk' or 'Bk'
    key_cand = ["zQk", "zBk", "Qk", "Bk"]
    zqk = None
    for k in key_cand:
        if k in g:
            zqk = np.asarray(g[k][:]).reshape(-1)
            break
            
    if zqk is None or len(zqk) < TARGET_K: return None, None, None, None
    Qk = zqk[:TARGET_K].astype(np.float32)
    
    return P0, P2, P4, Qk

def _get_metadata(diag_path):
    """Extract N_gal and Bias Parameters."""
    if "/diag/galaxies/" in diag_path:
        gal_path = diag_path.replace("/diag/galaxies/", "/galaxies/")
    else:
        gal_path = diag_path.replace("/diag/", "/")

    if not os.path.exists(gal_path): return -1, np.zeros(len(BIAS_PARAM_KEYS))

    with h5py.File(gal_path, "r") as f:
        g = _get_snapshot_group(f, gal_path)
        ngal = g["gal_type"].shape[0] if g and "gal_type" in g else -1
        bias_vals = [float(f.attrs.get(k, 0.0)) for k in BIAS_PARAM_KEYS]
            
    return ngal, np.array(bias_vals, dtype=np.float32)

# =====================================
# Main Pipeline
# =====================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_0", type=str, required=True, help="Path to Class 0 (Unbiased)")
    parser.add_argument("--dir_1", type=str, required=True, help="Path to Class 1 (Biased)")
    parser.add_argument("--out_dir", type=str, default="./data/vectors")
    parser.add_argument("--test_size", type=float, default=0.1)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # --- 1. Collect Files ---
    files, labels = [], []
    print("Scanning directories...")
    
    # Recursive glob to find files
    f0 = glob.glob(os.path.join(args.dir_0, "**", "galaxies", "hod*.h5"), recursive=True)
    files += f0; labels += [0]*len(f0)
    
    f1 = glob.glob(os.path.join(args.dir_1, "**", "galaxies", "hod*.h5"), recursive=True)
    files += f1; labels += [1]*len(f1)
    
    if len(files) == 0:
        print("Error: No files found. Check your paths.")
        return
        
    print(f"Found {len(f0)} Class 0 and {len(f1)} Class 1 files.")
    
    # --- 2. Load & Physics Normalize ---
    X_list = []
    meta_list = []
    
    print("\n[Phase 1] Loading and applying Self-Normalization (P / mean(P0))...")
    for path, lbl in tqdm(zip(files, labels), total=len(files)):
        try:
            with h5py.File(path, "r") as f:
                g = _get_snapshot_group(f, path)
                if g is None: continue
                P0, P2, P4, Qk = _load_spectra(g)
                if P0 is None: continue

            # === STRATEGY: Self-Normalization ===
            # 1. Calculate the 'volume' of the signal from the Monopole (P0)
            #    We use P0 because it's strictly positive and robust.
            amplitude_scale = np.mean(P0) + 1e-10
            
            # 2. Normalize EVERYTHING by this scale
            #    P_new = P_old / Scale
            #    This removes the linear bias amplitude factor b^2.
            P0_norm = P0 / amplitude_scale
            P2_norm = P2 / amplitude_scale
            P4_norm = P4 / amplitude_scale
            Qk_norm = Qk / amplitude_scale
            
            # 3. Concatenate Features
            #    Shape: [800]
            feat = np.concatenate([P0_norm, P2_norm, P4_norm, Qk_norm])
            
            # 4. Get Metadata
            ngal, bias_vec = _get_metadata(path)
            lhid, seed = parse_lhid_seed(path)
            
            X_list.append(feat)
            meta_list.append({
                "label": lbl, "lhid": lhid, "seed": seed, 
                "ngal": ngal, "bias": bias_vec,
                "scale_factor": amplitude_scale # Store scale factor just in case
            })
            
        except Exception as e:
            # print(f"Skipping {path}: {e}")
            continue

    X = np.stack(X_list).astype(np.float32)
    y = np.array([m["label"] for m in meta_list])
    print(f"Loaded Data Shape: {X.shape}")
    
    # --- 3. Split Train/Val ---
    # We split INDICES first to keep everything aligned
    idxs = np.arange(len(X))
    train_idx, val_idx = train_test_split(idxs, test_size=args.test_size, stratify=y, random_state=42)
    
    X_train = X[train_idx]
    X_val = X[val_idx]
    
    # --- 4. Global Statistics Normalization (Z-Score) ---
    print("\n[Phase 2] Computing Global Statistics (Standard Scaling)...")
    
    # CRITICAL: Compute Mean/Std ONLY on Training Data to avoid leakage
    mu = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True) + 1e-6 # Stability
    
    print(f"  Mean range: [{mu.min():.3f}, {mu.max():.3f}]")
    print(f"  Std  range: [{std.min():.3f}, {std.max():.3f}]")
    
    # Apply to Train and Val
    X_train_norm = (X_train - mu) / std
    X_val_norm = (X_val - mu) / std
    
    # --- 5. Save ---
    def pack_and_save(indices, x_norm, fname):
        subset_meta = [meta_list[i] for i in indices]
        
        # Build Metadata Tensors
        lhid_t = torch.tensor([m["lhid"] for m in subset_meta])
        seed_t = torch.tensor([m["seed"] for m in subset_meta])
        ngal_t = torch.tensor([m["ngal"] for m in subset_meta])
        bias_t = torch.tensor(np.stack([m["bias"] for m in subset_meta]))
        scale_t = torch.tensor([m["scale_factor"] for m in subset_meta])
        
        payload = {
            "data": torch.from_numpy(x_norm),           # [N, 800] Normalized
            "labels": torch.from_numpy(y[indices]).float().view(-1, 1),
            
            # Physics Metadata
            "lhid": lhid_t,
            "seed": seed_t,
            "ngal": ngal_t,
            "bias_params": bias_t,
            "amplitude_scale": scale_t, # Original amplitude, useful for analysis
            
            # Normalization Metadata (Saved in case we need to reverse it)
            "scaler_mean": torch.from_numpy(mu),
            "scaler_std": torch.from_numpy(std)
        }
        
        out_path = os.path.join(args.out_dir, fname)
        torch.save(payload, out_path)
        print(f"Saved {fname}: {x_norm.shape}")

    pack_and_save(train_idx, X_train_norm, "train_data.pt")
    pack_and_save(val_idx, X_val_norm, "val_data.pt")

    print("\nDone! Data is ready for evidence/data.py")

if __name__ == "__main__":
    main() 