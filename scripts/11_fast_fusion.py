#!/usr/bin/env python3
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import json

from evidence.core import one_pop_exponential_loss, compute_posterior

class EvidenceFitter(nn.Module):
    def __init__(self):
        super().__init__()
        # Temp=1.0, Gamma=0.0 起手式
        self.temp = nn.Parameter(torch.tensor(1.0))
        self.gamma = nn.Parameter(torch.tensor(0.0))

    def forward(self, f_mlp, f_gnn):
        # 【护城河 1】：严防 Temp 接近 0 导致的除以零崩溃
        safe_temp = torch.clamp(self.temp, min=0.01)
        return (f_mlp / safe_temp) + (self.gamma * f_gnn)

def main(args):
    print("Loading Extracted Evidences...")
    cache_path = os.path.join(args.out_dir, "sync_evidence.pt")
    cache = torch.load(cache_path, map_location='cpu', weights_only=False)
    
    tr_mlp, tr_gnn, tr_y = cache["train"]["f_mlp"], cache["train"]["f_gnn"], cache["train"]["y"]
    val_mlp, val_gnn, val_y = cache["val"]["f_mlp"], cache["val"]["f_gnn"], cache["val"]["y"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr_mlp, tr_gnn, tr_y = tr_mlp.to(device), tr_gnn.to(device), tr_y.to(device)
    val_mlp, val_gnn, val_y = val_mlp.to(device), val_gnn.to(device), val_y.to(device)

    model = EvidenceFitter().to(device)
    
    # 【护城河 2】：使用 AdamW 引入 Weight Decay
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    
    # 【护城河 3】：学习率动态退火
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=100)

    os.makedirs(args.out_dir, exist_ok=True)
    best_loss = float('inf')
    best_acc = 0.0
    best_params = {}
    
    # 【护城河 4】：Early Stopping 计数器
    patience_counter = 0

    print(f"Starting Light-Speed Optimization on {device}...")
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        
        f_tr = model(tr_mlp, tr_gnn)
        loss = one_pop_exponential_loss(f_tr, tr_y)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            f_val = model(val_mlp, val_gnn)
            val_loss = one_pop_exponential_loss(f_val, val_y).item()
            pred = (compute_posterior(f_val) > 0.5).float()
            val_acc = (pred.view(-1) == val_y.view(-1)).float().mean().item()

        scheduler.step(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            best_acc = val_acc
            patience_counter = 0 
            best_params = {"temp": model.temp.item(), "gamma": model.gamma.item(), "val_loss": val_loss, "val_acc": val_acc}
        else:
            patience_counter += 1

        if epoch % 500 == 0 or epoch == args.epochs - 1:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Ep {epoch:04d} | LR: {current_lr:.1e} | Tr Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | Temp: {model.temp.item():.4f} | Gamma: {model.gamma.item():.4f}")

        if patience_counter >= args.patience:
            print(f"\n[Early Stopping] Triggered! No improvement for {args.patience} epochs. Stopping at Ep {epoch}.")
            break
            
    print(f"\nOptimization Finished! BEST ACCURACY: {best_acc:.4f} (Loss: {best_loss:.4f})")
    print(f"Optimal Parameters: Temp = {best_params['temp']:.4f}, Gamma = {best_params['gamma']:.4f}")

    with open(os.path.join(args.out_dir, "optimal_fusion_params.json"), "w") as f:
        json.dump(best_params, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="/work/hdd/bdne/jdong8/fusion_models/ultimate_fusion")
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=1000)
    args = parser.parse_args()
    main(args)