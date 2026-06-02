#!/usr/bin/env python3
import os
import argparse
import math
import torch
import torch.nn as nn
import torch.optim as optim
import json

from evidence.core import l_pop_transform

# ==============================================================================
# RIGOROUS BAYESIAN FUSION ENGINE
# ==============================================================================
class BayesianEvidenceFitter(nn.Module):
    def __init__(self, alpha=2.0):
        super().__init__()
        # Temp handles the overconfidence of the Macroscopic MLP
        self.temp = nn.Parameter(torch.tensor(1.0))
        # Gamma strictly controls the signal-to-noise injection from the microscopic GNN
        # Start small because we know the GNN data was heavily blasted by numerical noise
        self.gamma = nn.Parameter(torch.tensor(0.01))
        self.alpha = alpha

    def forward(self, f_mlp, f_gnn):
        safe_temp = torch.clamp(self.temp, min=0.01)
        
        # [CRITICAL MATH CORRECTION]: 
        # Map raw neural network logits to Log Bayes Factor (J) space STRICTLY BEFORE addition.
        J_mlp = l_pop_transform(f_mlp, alpha=self.alpha)
        J_gnn = l_pop_transform(f_gnn, alpha=self.alpha)
        
        # Log-evidences superpose linearly. Mathematics is preserved.
        J_fused = (J_mlp / safe_temp) + (self.gamma * J_gnn)
        
        return J_fused

# We rewrite the 1-POP loss explicitly to accept pre-computed J_fused 
# avoiding the catastrophic double J-transform.
def direct_exponential_loss(J_fused, targets, c_prior):
    targets = targets.view_as(J_fused)
    term = (0.5 - targets) * (J_fused + c_prior)
    term = torch.clamp(term, min=-20.0, max=7.0)
    return torch.mean(torch.exp(term))

def direct_posterior(J_fused, c_prior):
    return torch.sigmoid(J_fused + c_prior)

def main(args):
    print(">>> Engaging Strict Bayesian Fusion Protocol...")
    cache_path = os.path.join(args.out_dir, "sync_evidence.pt")
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"[FATAL] Where is {cache_path}? Did you run the synchronizer?")
        
    cache = torch.load(cache_path, map_location='cpu', weights_only=False)
    
    tr_mlp, tr_gnn, tr_y = cache["train"]["f_mlp"], cache["train"]["f_gnn"], cache["train"]["y"]
    val_mlp, val_gnn, val_y = cache["val"]["f_mlp"], cache["val"]["f_gnn"], cache["val"]["y"]

    # Calculate rigorous Bayesian prior from the training dataset imbalance
    n1 = tr_y.sum().item()
    n0 = len(tr_y) - n1
    c_prior = math.log(n1 / max(1, n0))
    print(f">>> Dataset Baselines: N0 = {n0}, N1 = {n1} | Bayesian Prior (c) = {c_prior:.6f}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr_mlp, tr_gnn, tr_y = tr_mlp.to(device), tr_gnn.to(device), tr_y.to(device)
    val_mlp, val_gnn, val_y = val_mlp.to(device), val_gnn.to(device), val_y.to(device)

    model = BayesianEvidenceFitter(alpha=2.0).to(device)
    
    # High learning rate is fine for 2 parameters, but weight decay keeps gamma from hallucinating
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=100)

    os.makedirs(args.out_dir, exist_ok=True)
    best_loss = float('inf')
    best_acc = 0.0
    best_params = {}
    patience_counter = 0

    print(f"\n>>> Starting Light-Speed Optimization on {device}...")
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        
        J_tr_fused = model(tr_mlp, tr_gnn)
        loss = direct_exponential_loss(J_tr_fused, tr_y, c_prior)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            J_val_fused = model(val_mlp, val_gnn)
            val_loss = direct_exponential_loss(J_val_fused, val_y, c_prior).item()
            
            pred = (direct_posterior(J_val_fused, c_prior) > 0.5).float()
            val_acc = (pred.view(-1) == val_y.view(-1)).float().mean().item()

        scheduler.step(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            best_acc = val_acc
            patience_counter = 0 
            best_params = {
                "temp": model.temp.item(), 
                "gamma": model.gamma.item(), 
                "val_loss": val_loss, 
                "val_acc": val_acc
            }
        else:
            patience_counter += 1

        if epoch % 500 == 0 or epoch == args.epochs - 1:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Ep {epoch:04d} | LR: {current_lr:.1e} | Tr Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | Temp: {model.temp.item():.4f} | Gamma: {model.gamma.item():.4f}")

        if patience_counter >= args.patience:
            print(f"\n[Early Stopping] Triggered! No improvement for {args.patience} epochs.")
            break
            
    print("\n" + "="*60)
    print(">>> OPTIMIZATION FINISHED")
    print(f">>> BEST ACCURACY : {best_acc:.4f} (Loss: {best_loss:.4f})")
    print(f">>> OPTIMAL PARAMS: Temp = {best_params['temp']:.4f}, Gamma = {best_params['gamma']:.4f}")
    print("="*60 + "\n")

    with open(os.path.join(args.out_dir, "optimal_fusion_params.json"), "w") as f:
        json.dump(best_params, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="/work/hdd/bdne/jdong8/fusion_models/ultimate_fusion")
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=1000)
    args = parser.parse_args()
    main(args)