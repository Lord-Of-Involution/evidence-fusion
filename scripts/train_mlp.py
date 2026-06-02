#!/usr/bin/env python3
"""
Unified MLP Engine: Train (BCE -> 1POP) & Extract
=================================================
Strictly adheres to Bayesian probability axioms.
Outputs pre-computed J-space likelihood ratios for downstream fusion.
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
# 1. Evidence Network Architecture
# ==============================================================================
class EvidenceNetworkPaper(nn.Module):
    def __init__(self, input_size, hidden_sizes, activation="gelu", dropout=0.0, batchnorm=False):
        super().__init__()
        layers = []
        in_dim = input_size
        act_fn = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}.get(activation.lower(), nn.GELU)
        
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            if batchnorm: 
                layers.append(nn.BatchNorm1d(h))
            layers.append(act_fn())
            if dropout > 0.0: 
                layers.append(nn.Dropout(dropout))
            in_dim = h
            
        self.backbone = nn.Sequential(*layers)
        self.output_dim = in_dim
        self.head = nn.Linear(in_dim, 1) 

    def forward(self, x):
        return self.head(self.backbone(x))

# ==============================================================================
# 2. Loss & Transform Math
# ==============================================================================
def l_pop_transform(x, alpha=2.0):
    return x + x * torch.abs(x).pow(alpha - 1)

def one_pop_exponential_loss(f_x, targets, alpha=2.0, c=0.0):
    J_val = l_pop_transform(f_x, alpha)
    targets = targets.view_as(f_x)
    term = (0.5 - targets) * (J_val + c)
    # Prevent infinite gradients from extreme local outliers
    term = torch.clamp(term, min=-20.0, max=7.0)
    loss = torch.exp(term)
    return torch.mean(loss)

# ==============================================================================
# 3. Deterministic Dataset Loader
# ==============================================================================
class VectorDataset(Dataset):
    def __init__(self, path):
        print(f">>> Loading macroscopic vectors from {path}...")
        if not os.path.exists(path):
            raise FileNotFoundError(f"[FATAL] File not found: {path}")

        payload = torch.load(path, map_location="cpu", weights_only=False)

        if 'data' in payload: 
            self.x = payload['data']
        elif 'stats' in payload: 
            self.x = payload['stats']
        else: 
            raise KeyError(f"[FATAL] Valid feature matrix not found in {path}")
            
        self.y = payload['labels']
        self.x = self.x.float()
        
        # 10-sigma anomaly truncation
        self.x = torch.clamp(self.x, min=-10.0, max=10.0)
        self.y = self.y.float().view(-1, 1)
        
        # Legacy auto-transpose safeguard
        if self.x.shape[0] < self.x.shape[1] and self.x.shape[0] <= 2000:
             print(f"    -> Executing transpose patch from {self.x.shape}")
             self.x = self.x.T
        
        print(f"    -> Verified Shape: {self.x.shape}")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# ==============================================================================
# 4. Evidence Extraction Engine
# ==============================================================================
def extract_evidence(model, loader, device, desc):
    all_f, all_y = [], []
    model.eval()
    with torch.no_grad():
        for x, y in tqdm(loader, desc=desc, leave=False):
            x = x.to(device)
            f_x = model(x)
            all_f.append(f_x.cpu())
            all_y.append(y.cpu())
    return torch.cat(all_f), torch.cat(all_y)

# ==============================================================================
# 5. Pipeline Execution
# ==============================================================================
def execute_pipeline(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n>>> Waking up Silicon on {device}...")
    
    train_path = os.path.join(args.vector_dir, "train_data.pt")
    val_path = os.path.join(args.vector_dir, "val_data.pt")
    
    train_ds = VectorDataset(train_path)
    val_ds = VectorDataset(val_path)
    
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # Bayesian Prior Calculation
    n1 = train_ds.y.sum().item()
    n0 = len(train_ds.y) - n1
    c_prior = math.log(n1 / max(1, n0))
    print("\n" + "="*60)
    print(f">>> Manifold Statistics : N0 = {n0}, N1 = {n1}")
    print(f">>> Bayesian Prior (c)  : {c_prior:.6f}")
    print("="*60 + "\n")

    input_dim = train_ds.x.shape[1]
    
    model = EvidenceNetworkPaper(
        input_size=input_dim, 
        hidden_sizes=args.hidden_sizes,
        dropout=args.dropout,
        batchnorm=True
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    os.makedirs(args.out_dir, exist_ok=True)
    
    config = vars(args)
    config["input_dim"] = input_dim
    config["batchnorm"] = True
    
    with open(os.path.join(args.out_dir, "mlp_config.json"), "w") as f:
        json.dump(config, f, indent=4)

    best_val_loss = float('inf')
    model_save_path = os.path.join(args.out_dir, "mlp_best.pt")
    
    # --- TRAINING PHASE ---
    for epoch in range(args.epochs):
        # Phase Shift Logic
        if epoch == args.bce_epochs:
            print("\n" + "="*60)
            print(f">>> [PHASE SHIFT] BCE Pre-heating complete. Engaging 1-POP Exponential Loss.")
            print(">>> Resetting validation baseline and scheduler memory.")
            print("="*60 + "\n")
            best_val_loss = float('inf')
            # Flush optimizer momentum safely
            optimizer.param_groups[0]['lr'] = args.lr * 0.5 
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

        model.train()
        total_loss = 0.0
        batches = 0
        
        pbar = tqdm(train_dl, desc=f"Ep {epoch:03d} [Tr]", dynamic_ncols=True)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            f_x = model(x)
            
            if epoch < args.bce_epochs:
                logits = l_pop_transform(f_x, alpha=2.0) + c_prior
                loss = torch.nn.BCEWithLogitsLoss()(logits, y)
                loss_type = "BCE"
            else:
                loss = one_pop_exponential_loss(f_x, y, alpha=2.0, c=c_prior)
                loss_type = "1POP"
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            batches += 1
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'type': loss_type})

        avg_train_loss = total_loss / max(1, batches)

        # Validation Phase
        model.eval()
        val_loss_sum = 0.0
        correct = 0
        total = 0
        val_batches = 0
        
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                f_x = model(x)
                
                if epoch < args.bce_epochs:
                    logits = l_pop_transform(f_x, alpha=2.0) + c_prior
                    loss = torch.nn.BCEWithLogitsLoss()(logits, y)
                else:
                    loss = one_pop_exponential_loss(f_x, y, alpha=2.0, c=c_prior)
                    
                val_loss_sum += loss.item()
                val_batches += 1
                
                # Bayesian accuracy calculation
                J_val = l_pop_transform(f_x, alpha=2.0)
                probs = torch.sigmoid(J_val + c_prior)
                preds = (probs > 0.5).float()
                correct += (preds == y).sum().item()
                total += y.size(0)

        avg_val_loss = val_loss_sum / max(1, val_batches)
        val_acc = correct / max(1, total)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Ep {epoch:03d} | LR: {current_lr:.2e} | Tr Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Acc: {val_acc:.4f}")
        
        scheduler.step(avg_val_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_save_path)


    if args.epochs > 0:
        for epoch in range(args.epochs):
            # ... (这里放你所有的训练循环、Phase Shift 和 Validation 代码) ...
            # ... scheduler.step(avg_val_loss) ...
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), model_save_path)
    else:
        print("\n[WARNING] args.epochs is 0. Bypassing training loop entirely!")
        if not os.path.exists(model_save_path):
            raise FileNotFoundError(f"Cannot extract! No pre-trained model found at {model_save_path}")


    # --- EXTRACTION PHASE ---
    print("\n" + "="*60)
    print(">>> Training Finalized. Engaging Deep Extraction Protocol...")
    print("="*60)
    
    # Reload the absolute best weights
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    
    # Re-initialize loaders with absolute determinism (shuffle=False)
    ext_train_dl = DataLoader(train_ds, batch_size=1024, shuffle=False, num_workers=4, pin_memory=True)
    ext_val_dl = DataLoader(val_ds, batch_size=1024, shuffle=False, num_workers=4, pin_memory=True)
    
    tr_f, tr_y = extract_evidence(model, ext_train_dl, device, "Extract Train")
    val_f, val_y = extract_evidence(model, ext_val_dl, device, "Extract Val")

    cache_path = os.path.join(args.out_dir, "mlp_evidence.pt")
    torch.save({
        "train": {"f_mlp": tr_f, "y": tr_y},
        "val": {"f_mlp": val_f, "y": val_y}
    }, cache_path)
    
    print(f"\n>>> Checkmate. Pure Evidence successfully harvested and cached at:")
    print(f"    -> {cache_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Unified naming convention: Everything is vector_dir for MLP inputs
    parser.add_argument("--vector_dir", type=str, default="/work/hdd/bdne/jdong8/fusion_data_purified/vectors")
    parser.add_argument("--out_dir", type=str, default="/work/hdd/bdne/jdong8/fusion_models/abacus_mlp_solo")
    
    parser.add_argument("--hidden_sizes", type=int, nargs='+', default=[512, 512, 128])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--bce_epochs", type=int, default=30) 
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    
    args = parser.parse_args()
    execute_pipeline(args)