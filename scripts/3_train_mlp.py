#!/usr/bin/env python3
"""
[Step 3] Train MLP (The Teacher) - Final Version
================================================
Standardized script for Project 2.0.
Reads 'train_data.pt' and 'val_data.pt'.
Implements Jeffrey & Wandelt (2023) Evidence Network (Scalar Output + 1-POP Loss).
"""

import os
import json
import argparse
import numpy as np
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ==============================================================================
# 1. Evidence Network Architecture (Scalar Output)
# ==============================================================================
class EvidenceNetworkPaper(nn.Module):
    def __init__(self, input_size, hidden_sizes, activation="gelu", dropout=0.0, batchnorm=False):
        super().__init__()
        layers =[]
        in_dim = input_size
        act_fn = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}.get(activation.lower(), nn.GELU)
        
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            if batchnorm: layers.append(nn.BatchNorm1d(h))
            layers.append(act_fn())
            if dropout > 0.0: layers.append(nn.Dropout(dropout))
            in_dim = h
            
        self.backbone = nn.Sequential(*layers)
        self.output_dim = in_dim
        self.head = nn.Linear(in_dim, 1) 

    def forward(self, x):
        return self.head(self.backbone(x))

# ==============================================================================
# 2. Loss & Transform
# ==============================================================================
def l_pop_transform(x, alpha=2.0):
    return x + x * torch.abs(x).pow(alpha - 1)

def one_pop_exponential_loss(f_x, targets, alpha=2.0, c=0.0):
    J_val = l_pop_transform(f_x, alpha)
    term = (0.5 - targets) * (J_val + c)
    # [CRITICAL ARMOR] Prevent Gradient Hijacking by an isolated outlier.
    # max=7.0 bounds the maximum single-sample penalty to e^7 ≈ 1096.
    term = torch.clamp(term, min=-20.0, max=7.0)
    loss = torch.exp(term)
    return torch.mean(loss)

# ==============================================================================
# 3. Standard Dataset Loader
# ==============================================================================
class VectorDataset(Dataset):
    def __init__(self, path):
        print(f"Loading vectors from {path}...")
        try:
            payload = torch.load(path, map_location="cpu")
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {path}\nDid you run 1_prep_vectors.py?")

        if 'data' in payload: self.x = payload['data']
        elif 'stats' in payload: self.x = payload['stats']
        elif 'data_stats' in payload: self.x = payload['data_stats']
        else: raise KeyError(f"Key 'data' not found in {path}")
            
        self.y = payload['labels']
        self.x = self.x.float()
        # [CRITICAL ARMOR] Clamp input features to 10-sigma.
        # Eradicates shot-noise outliers from the lightcone summary statistics.
        self.x = torch.clamp(self.x, min=-10.0, max=10.0)
        self.y = self.y.float().view(-1, 1)
        
        if self.x.shape[0] < self.x.shape[1] and self.x.shape[0] <= 2000:
             print(f"  -> Auto-transposing from {self.x.shape}")
             self.x = self.x.T
        
        print(f"  -> Final Shape: {self.x.shape}")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# ==============================================================================
# 4. Main Training Loop
# ==============================================================================
def train(args):
    if torch.cuda.is_available(): device = torch.device("cuda")
    elif torch.backends.mps.is_available(): device = torch.device("mps")
    else: device = torch.device("cpu")
    print(f"Using device: {device}")
    
    train_path = os.path.join(args.data_dir, "train_data.pt")
    val_path = os.path.join(args.data_dir, "val_data.pt")
    
    train_ds = VectorDataset(train_path)
    val_ds = VectorDataset(val_path)
    
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    input_dim = train_ds.x.shape[1]
    print(f"Input Dimension: {input_dim}")
    
    # ======== 【贝叶斯先验偏置计算】 ========
    # 动态获取训练集中的正负样本比例，绝不 Hardcode
    n1 = train_ds.y.sum().item()
    n0 = len(train_ds.y) - n1
    c_prior = math.log(n1 / max(1, n0))
    print("="*60)
    print(f">>> Dataset Statistics: N0 = {n0}, N1 = {n1}")
    print(f">>> Bayesian Prior Bias (c) = {c_prior:.6f}")
    print("="*60)

    input_dim = train_ds.x.shape[1]
    print(f"Input Dimension: {input_dim}")
    
    model = EvidenceNetworkPaper(
        input_size=input_dim, 
        hidden_sizes=args.hidden_sizes,
        dropout=args.dropout,
        batchnorm=True
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # Patience 拉长到 10，防止前期波动把学习率降光
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    print(f"Starting Training for {args.epochs} epochs...")
    os.makedirs(args.out_dir, exist_ok=True)
    
    config = vars(args)
    config["input_dim"] = input_dim
    config["batchnorm"] = True
    
    with open(os.path.join(args.out_dir, "mlp_config.json"), "w") as f:
        json.dump(config, f, indent=4)

    best_val_loss = math.inf
    
    for epoch in range(args.epochs):

        if epoch == args.bce_epochs:
            print("\n" + "="*60)
            print(f">>> [Phase Shift] Switching from BCE to 1-POP Loss at Epoch {epoch}!")
            print(">>> Resetting best_val_loss and LR Scheduler...")
            print("="*60 + "\n")
            best_val_loss = math.inf
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=10
            )

        model.train()
        total_loss = 0; batches = 0
        
        pbar = tqdm(train_dl, desc=f"Ep {epoch} [Train]")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            f_x = model(x)
            
            # ======== 【BCE 预热】 ========
            if epoch < args.bce_epochs:
                # BCE 吃的是 Logits (后验)，所以必须用 J(x) + c
                logits = l_pop_transform(f_x, alpha=2.0) + c_prior
                loss = torch.nn.BCEWithLogitsLoss()(logits, y)
                loss_type = "BCE"
            else:
                # 1-POP 传入 c_prior
                loss = one_pop_exponential_loss(f_x, y, alpha=2.0, c=c_prior)
                loss_type = "1POP"
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item(); batches += 1
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'type': loss_type})

        avg_train_loss = total_loss / max(1, batches)

        # Validation
        model.eval()
        val_loss_sum = 0; correct = 0; total = 0; val_batches = 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                f_x = model(x)
                
                if epoch < args.bce_epochs:
                    # BCE 吃的是 Logits (后验)，所以必须用 J(x) + c
                    logits = l_pop_transform(f_x, alpha=2.0) + c_prior
                    loss = torch.nn.BCEWithLogitsLoss()(logits, y)
                    loss_type = "BCE"
                else:
                    # 1-POP 传入 c_prior
                    loss = one_pop_exponential_loss(f_x, y, alpha=2.0, c=c_prior)
                    loss_type = "1POP"
                    
                val_loss_sum += loss.item()
                val_batches += 1
                
                # Acc
                J_val = l_pop_transform(f_x, alpha=2.0)
                probs = torch.sigmoid(J_val+ c_prior)
                preds = (probs > 0.5).float()
                correct += (preds == y).sum().item(); total += y.size(0)

        avg_val_loss = val_loss_sum / max(1, val_batches)
        val_acc = correct / max(1, total)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Ep {epoch} | Tr Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f} | LR: {current_lr:.2e}")
        
        scheduler.step(avg_val_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(args.out_dir, "mlp_best.pt"))
            print(f"  --> New Best Model Saved (Loss: {best_val_loss:.4f})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data/vectors")
    parser.add_argument("--out_dir", type=str, default="./models/mlp_paper")
    parser.add_argument("--hidden_sizes", type=int, nargs='+', default=[512, 512, 128])
    
    parser.add_argument("--epochs", type=int, default=60) # 拉长一点配合20轮的warmup
    parser.add_argument("--bce_epochs", type=int, default=20) 
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    args = parser.parse_args()
    train(args)