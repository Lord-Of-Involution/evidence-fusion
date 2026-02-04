#!/usr/bin/env python3
"""
[Diagnostic Step] Train CNN Solo (No MLP)
=========================================
Purpose: 
1. Check if 128^3 grids contain extractable information.
2. Debug input statistics (Range, Non-zero fraction).
3. Establish a baseline for pure visual performance.
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Import your existing modules
from evidence.models import Small3DCNN
from evidence.data import FusionDataset, get_balanced_loader
from evidence.core import one_pop_exponential_loss, compute_posterior

# ==============================================================================
# 1. Wrapper Model for Solo Training
# ==============================================================================
class SoloCNN(nn.Module):
    def __init__(self, feature_dim=128):
        super().__init__()
        # Load your definition from models.py
        # Make sure models.py has the AdaptiveAvgPool fix we discussed!
        self.cnn = Small3DCNN(input_channels=1, feature_dim=feature_dim)
        
        # Simple Linear Head for Classification (Scalar Output)
        self.head = nn.Linear(feature_dim, 1) 

    def forward(self, x):
        # --- CRITICAL: Log Transform ---
        # N-body density can range from -1 to 1000+. 
        # CNNs hate this. We squash it.
        # log(x + 2) ensures inputs are positive and compressed.
        #x = torch.log1p(x + 1.0) 
        
        features = self.cnn(x)
        return self.head(features)

# ==============================================================================
# 2. Training Logic
# ==============================================================================
def train_solo(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data Loading ---
    print(f"Loading Grids from {args.grid_dir}...")
    
    # We reuse FusionDataset but we will ignore the 'vec' part in the loop
    train_ds = FusionDataset(
        vector_path=os.path.join(args.vector_dir, "train_data.pt"),
        grid_path=os.path.join(args.grid_dir, "train_grids.npy")
    )
    val_ds = FusionDataset(
        vector_path=os.path.join(args.vector_dir, "val_data.pt"),
        grid_path=os.path.join(args.grid_dir, "val_grids.npy")
    )
    
    # Use robust num_workers
    train_loader = get_balanced_loader(train_ds, batch_size=args.batch_size, num_workers=16, prefetch_factor=2)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=16, prefetch_factor=2)
    
    print(f"Data Ready. Train: {len(train_ds)}, Val: {len(val_ds)}")

    # --- Model Setup ---
    model = SoloCNN(feature_dim=128).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # --- Training Loop ---
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        batches = 0
        
        pbar = tqdm(train_loader, desc=f"CNN Solo Ep {epoch}")
        
        for i, ((grid, _), label) in enumerate(pbar):
            # Move to GPU
            grid, label = grid.to(device), label.to(device)
            
            # --- [DIAGNOSTIC] Check Data Stats on First Batch ---
            if epoch == 0 and i == 0:
                print(f"\n[DEBUG] Input Grid Stats:")
                print(f"  Shape: {grid.shape}")
                print(f"  Raw Min: {grid.min().item():.4f} | Max: {grid.max().item():.4f}")
                print(f"  Raw Mean: {grid.mean().item():.4f} | Std: {grid.std().item():.4f}")
                print(f"  Non-Zero Fraction: {(grid != 0).float().mean().item():.6f}")
                print("  (Note: Log transform is applied inside model forward)\n")
            # ----------------------------------------------------

            optimizer.zero_grad()
            f_x = model(grid) # Only using Grid!
            
            loss = one_pop_exponential_loss(f_x, label)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            batches += 1
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        avg_train_loss = train_loss / batches
        
        # --- Validation ---
        model.eval()
        val_loss = 0.0
        correct = 0; total = 0
        
        with torch.no_grad():
            for (grid, _), label in val_loader:
                grid, label = grid.to(device), label.to(device)
                f_x = model(grid)
                
                loss = one_pop_exponential_loss(f_x, label)
                val_loss += loss.item()
                
                post = compute_posterior(f_x)
                pred = (post > 0.5).float()
                correct += (pred == label).sum().item()
                total += label.size(0)

        avg_val_loss = val_loss / len(val_loader)
        acc = correct / total
        
        print(f"CNN Solo Ep {epoch} | Tr Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Acc: {acc:.4f}")
        
        scheduler.step(avg_val_loss)
        
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(args.out_dir, "cnn_solo_best.pt"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vector_dir", type=str, default="./data/vectors")
    parser.add_argument("--grid_dir", type=str, default="./data/grids")
    parser.add_argument("--out_dir", type=str, default="./models/cnn_solo")
    parser.add_argument("--lr", type=float, default=1e-4) # Slightly higher LR for solo training
    parser.add_argument("--epochs", type=int, default=30)
    # Important: Default batch size for 128^3
    parser.add_argument("--batch_size", type=int, default=16) 
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    train_solo(args)