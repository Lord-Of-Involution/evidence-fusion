#!/usr/bin/env python3
import os
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from evidence.models import EvidenceMLP, Small3DCNN
from evidence.data import FusionDataset, get_balanced_loader
from evidence.core import one_pop_exponential_loss, compute_posterior
from tqdm import tqdm
import sys 

class SmartFusionNetwork(nn.Module):
    def __init__(self, mlp, cnn, cnn_feat_dim=128):
        super().__init__()
        self.mlp = mlp
        self.cnn = cnn
        
        self.cnn_proj = nn.Linear(cnn_feat_dim, 1)
        

        nn.init.zeros_(self.cnn_proj.weight)
        nn.init.zeros_(self.cnn_proj.bias)
        

        self.gamma = nn.Parameter(torch.zeros(1))
        

        self.mlp_temp = nn.Parameter(torch.tensor(2.0, dtype=torch.float32))

    def forward(self, grid, vec):
        with torch.no_grad():
            f_mlp = self.mlp(vec)         # [Batch, 1]
            
        cnn_feat = self.cnn(grid)         # [Batch, 128]
        f_cnn = self.cnn_proj(cnn_feat)   # [Batch, 1] 
        

        f_final = (f_mlp / self.mlp_temp) + (self.gamma * f_cnn)
        return f_final

# ==============================================================================
# Helper Function: Run Validation
# ==============================================================================
def run_validation(model, loader, device, desc="Validation"):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(loader, desc=desc, leave=False)
        for (grid, vec), label in pbar:
            grid, vec, label = grid.to(device), vec.to(device), label.to(device)
            
            f_x = model(grid, vec)
            
            # 删掉了极其愚蠢的 torch.clamp！现在有 gamma=0 护体，不需要阉割梯度了！
            
            loss = one_pop_exponential_loss(f_x, label)
            val_loss += loss.item()
            
            post = compute_posterior(f_x)
            pred = (post > 0.5).float()
            correct += (pred == label).sum().item()
            total += label.size(0)

    avg_loss = val_loss / len(loader)
    acc = correct / total
    return avg_loss, acc

# ==============================================================================
# Main Training Logic
# ==============================================================================
def train_fusion(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load MLP
    print(f"Loading MLP from {args.mlp_dir}...")
    with open(os.path.join(args.mlp_dir, "mlp_config.json"), "r") as f:
        mlp_conf = json.load(f)
        
    mlp = EvidenceMLP(
        input_size=mlp_conf["input_dim"],
        hidden_sizes=mlp_conf["hidden_sizes"],
        dropout=mlp_conf["dropout"]
    )
    mlp.load_state_dict(torch.load(os.path.join(args.mlp_dir, "mlp_best.pt"), map_location=device))
    
    # 2. Build Fusion Model
    cnn = Small3DCNN(input_channels=1, feature_dim=128)

    print("Locking the Teacher: Freezing MLP parameters...")
    for param in mlp.parameters():
        param.requires_grad = False

    # 使用我们注入了秘密知识的 SmartFusionNetwork
    model = SmartFusionNetwork(mlp, cnn)

    if torch.cuda.device_count() > 1:
        print(f"Detected {torch.cuda.device_count()} GPUs! Using DataParallel.")
        model = torch.nn.DataParallel(model)
    model.to(device)
    
    # 3. Resume Logic
    best_loss = float('inf')
    
    if args.resume:
        print(f"\nResuming training from: {args.resume}")
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume, map_location=device)
            ckpt_keys = list(checkpoint.keys())
            state_dict = checkpoint
            
            if ckpt_keys[0].startswith('module.') and not isinstance(model, torch.nn.DataParallel):
                state_dict = {k.replace('module.', ''): v for k, v in checkpoint.items()}
            elif not ckpt_keys[0].startswith('module.') and isinstance(model, torch.nn.DataParallel):
                state_dict = {'module.' + k: v for k, v in checkpoint.items()}
            
            model.load_state_dict(state_dict)
            print("Checkpoint loaded successfully!")
        else:
            print(f"Error: Checkpoint file not found: {args.resume}")
            sys.exit(1)

    # 包含 gamma 和 mlp_temp 在内的所有 require_grad 参数
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                           lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # 4. Data
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
    print(f"Fusion Data Ready. Train: {len(train_ds)}, Val: {len(val_ds)}")

    # 5. Baseline Check
    if args.resume:
        print("\nCalculating baseline performance from loaded checkpoint...")
        baseline_loss, baseline_acc = run_validation(model, val_loader, device, desc="Baseline Check")
        best_loss = baseline_loss
        print(f"Resumed Model Baseline -> Loss: {baseline_loss:.4f} | Acc: {baseline_acc:.4f}")
    else:
        print("\nVerifying Smart Initialization (Gamma=0 ensures fallback to MLP baseline)...")
        baseline_loss, baseline_acc = run_validation(model, val_loader, device, desc="Init Check")
        print(f"Initial Accuracy (Should match MLP alone): {baseline_acc:.4f} | Init Loss: {baseline_loss:.4f}")
        best_loss = baseline_loss

    # 6. Training Loop
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        
        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"Ep {epoch}")
        
        for i, ((grid, vec), label) in enumerate(pbar):
            grid, vec, label = grid.to(device), vec.to(device), label.to(device)
            
            f_x = model(grid, vec)
            
            # 删掉 Clamp，直接算 Loss
            loss = one_pop_exponential_loss(f_x, label)
            
            # Normalize loss for gradient accumulation
            loss = loss / args.accum_steps
            loss.backward()
            
            # Perform optimization step every accum_steps
            if ((i + 1) % args.accum_steps == 0) or ((i + 1) == len(train_loader)):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                
            train_loss += (loss.item() * args.accum_steps)
            
            # 获取秘密监控器：gamma 和 temp 的值
            current_gamma = model.module.gamma.item() if isinstance(model, torch.nn.DataParallel) else model.gamma.item()
            current_temp = model.module.mlp_temp.item() if isinstance(model, torch.nn.DataParallel) else model.mlp_temp.item()
            
            # 把诊断信息打在公屏上，这才是你判断网络有没有学到东西的唯一标准
            pbar.set_postfix({
                'loss': f"{(loss.item() * args.accum_steps):.4f}", 
                'gamma': f"{current_gamma:.4f}",
                'temp': f"{current_temp:.2f}"
            })
            
        avg_train_loss = train_loss / len(train_loader)
        
        # --- Validation ---
        avg_val_loss, val_acc = run_validation(model, val_loader, device, desc=f"Val Ep {epoch}")
        print(f"Fusion Ep {epoch} | Tr Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Acc: {val_acc:.4f}")
        
        scheduler.step(avg_val_loss)
        
        last_path = os.path.join(args.out_dir, "fusion_last.pt")
        torch.save(model.state_dict(), last_path)

        if avg_val_loss < best_loss:
            print(f"  New Best! ({best_loss:.4f} -> {avg_val_loss:.4f}). Saving best model...")
            best_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(args.out_dir, "fusion_best.pt"))
        else:
            print(f"  Result {avg_val_loss:.4f} >= Best {best_loss:.4f}. Updated 'last.pt'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vector_dir", type=str, default="./data/vectors")
    parser.add_argument("--grid_dir", type=str, default="./data/grids")
    parser.add_argument("--mlp_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="./models/fusion")
    parser.add_argument("--lr", type=float, default=5e-5) # 学习率保持这样，让 gamma 慢慢爬
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8) 
    parser.add_argument("--accum_steps", type=int, default=4) 
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint")
    
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    train_fusion(args)