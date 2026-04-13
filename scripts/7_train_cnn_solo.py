#!/usr/bin/env python3
"""
[Diagnostic Step] Train CNN Solo with Loss Switch (Shape Fix)
=============================================================
Purpose: 
Compare stable feature extraction (BCE) vs. Bayes Factor 
estimation (I-POP) on 3D grid data.
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from evidence.models import Small3DCNN
from evidence.data import FusionDataset, get_balanced_loader
from evidence.core import one_pop_exponential_loss, compute_posterior

# ==============================================================================
# 1. Wrapper Model for Solo Training
# ==============================================================================
class SoloCNN(nn.Module):
    def __init__(self, feature_dim=128):
        super().__init__()
        self.cnn = Small3DCNN(input_channels=1, feature_dim=feature_dim)
        self.head = nn.Linear(feature_dim, 1) 

    def forward(self, x):
        # 强制对数压缩：拯救方差
        x_safe = torch.log1p(torch.relu(x + 1.0))
        features = self.cnn(x_safe)
        return self.head(features) # Output shape: [Batch, 1]

# ==============================================================================
# 2. Training Logic
# ==============================================================================
def train_solo(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Selected Loss Type: {args.loss_type.upper()}")

    print(f"Loading Grids from {args.grid_dir}...")
    
    train_ds = FusionDataset(
        vector_path=os.path.join(args.vector_dir, "train_data.pt"),
        grid_path=os.path.join(args.grid_dir, "train_grids.npy")
    )
    val_ds = FusionDataset(
        vector_path=os.path.join(args.vector_dir, "val_data.pt"),
        grid_path=os.path.join(args.grid_dir, "val_grids.npy")
    )
    
    train_loader = get_balanced_loader(train_ds, batch_size=args.batch_size, num_workers=16, prefetch_factor=2)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=16, prefetch_factor=2)
    
    print(f"Data Ready. Train: {len(train_ds)}, Val: {len(val_ds)}")

    # --- Model Setup ---
    model = SoloCNN(feature_dim=128)
    
    if torch.cuda.device_count() > 1:
        print(f"Detected {torch.cuda.device_count()} GPUs! Using DataParallel.")
        model = nn.DataParallel(model)
        
    model = model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    patience = 3 if args.loss_type == 'ipop' else 4
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=patience)

    bce_criterion = nn.BCEWithLogitsLoss() if args.loss_type == 'bce' else None
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        batches = 0
        
        pbar = tqdm(train_loader, desc=f"CNN Solo Ep {epoch}")
        
        for i, ((grid, _), label) in enumerate(pbar):
            grid, label = grid.to(device), label.to(device)

            optimizer.zero_grad()
            

            f_x = model(grid) 
            

            if args.loss_type == 'bce':
                loss = bce_criterion(f_x, label.float().view_as(f_x))
            else:
                loss = one_pop_exponential_loss(f_x, label.view_as(f_x))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
                f_x = model(grid) # 保持 [Batch, 1]
                
                if args.loss_type == 'bce':
                    loss = bce_criterion(f_x, label.float().view_as(f_x))
                    val_loss += loss.item()
                    pred = (f_x > 0.0).float()
                else:
                    loss = one_pop_exponential_loss(f_x, label.view_as(f_x))
                    val_loss += loss.item()
                    post = compute_posterior(f_x)
                    pred = (post > 0.5).float()
                
                # 对齐 pred 和 label 的形状再求准确率
                correct += (pred.view_as(label) == label).sum().item()
                total += label.size(0)

        avg_val_loss = val_loss / len(val_loader)
        acc = correct / total
        
        print(f"CNN Solo ({args.loss_type.upper()}) Ep {epoch} | Tr Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Acc: {acc:.4f}")
        
        scheduler.step(avg_val_loss)
        
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            state_to_save = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            save_name = f"cnn_solo_{args.loss_type}_best.pt"
            torch.save(state_to_save, os.path.join(args.out_dir, save_name))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vector_dir", type=str, default="./data/vectors")
    parser.add_argument("--grid_dir", type=str, default="./data/grids")
    parser.add_argument("--out_dir", type=str, default="./models/cnn_solo")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16) 
    parser.add_argument("--loss_type", type=str, choices=['bce', 'ipop'], default='bce', 
                        help="Choose loss function: 'bce' for robust feature extraction, 'ipop' for evidence estimation.")
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    train_solo(args)