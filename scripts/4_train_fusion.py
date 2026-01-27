#!/usr/bin/env python3
import os
import json
import argparse
import torch
import torch.optim as optim
from evidence.models import EvidenceMLP, Small3DCNN, FusionEvidenceNetwork
from evidence.data import FusionDataset, get_balanced_loader
from evidence.core import one_pop_exponential_loss, compute_posterior

def train_fusion(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # ==========================================================================
    # 1. Load Pre-trained MLP (The Teacher)
    # ==========================================================================
    print(f"Loading MLP from {args.mlp_dir}...")
    
    # Load Config
    with open(os.path.join(args.mlp_dir, "mlp_config.json"), "r") as f:
        mlp_conf = json.load(f)
        
    mlp = EvidenceMLP(
        input_size=mlp_conf["input_size"],
        hidden_sizes=mlp_conf["hidden_sizes"],
        dropout=mlp_conf["dropout"]
    )
    # Load Weights
    mlp.load_state_dict(torch.load(os.path.join(args.mlp_dir, "mlp_best.pt"), map_location=device))
    
    # ==========================================================================
    # 2. Build Fusion Model
    # ==========================================================================
    cnn = Small3DCNN(input_channels=1, feature_dim=64)
    
    # This automatically Freezes MLP and applies Smart Init
    model = FusionEvidenceNetwork(mlp, cnn).to(device)
    
    # Only optimize parameters that require gradients (CNN + Fusion Head)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                           lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # ==========================================================================
    # 3. Data (Lazy Loading NPY)
    # ==========================================================================
    # Note: fusion_ds[i] returns ((grid, vec), label)
    train_ds = FusionDataset(
        vector_path=os.path.join(args.vector_dir, "train_data.pt"),
        grid_path=os.path.join(args.grid_dir, "train_grids.npy")
    )
    val_ds = FusionDataset(
        vector_path=os.path.join(args.vector_dir, "val_data.pt"),
        grid_path=os.path.join(args.grid_dir, "val_grids.npy")
    )
    
    train_loader = get_balanced_loader(train_ds, batch_size=args.batch_size)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    
    print(f"Fusion Data Ready. Train: {len(train_ds)}, Val: {len(val_ds)}")

    # ==========================================================================
    # 4. Training Loop
    # ==========================================================================
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        
        # Batch: ((grid, vec), label)
        for (grid, vec), label in train_loader:
            grid, vec, label = grid.to(device), vec.to(device), label.to(device)
            
            optimizer.zero_grad()
            f_x = model(grid, vec) # Joint prediction
            
            loss = one_pop_exponential_loss(f_x, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        
        # --- Validation ---
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for (grid, vec), label in val_loader:
                grid, vec, label = grid.to(device), vec.to(device), label.to(device)
                f_x = model(grid, vec)
                
                loss = one_pop_exponential_loss(f_x, label)
                val_loss += loss.item()
                
                post = compute_posterior(f_x)
                pred = (post > 0.5).float()
                correct += (pred == label).sum().item()
                total += label.size(0)

        avg_val_loss = val_loss / len(val_loader)
        acc = correct / total
        
        print(f"Fusion Ep {epoch} | Tr Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Acc: {acc:.4f}")
        
        scheduler.step(avg_val_loss)
        
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(args.out_dir, "fusion_best.pt"))
            print("  -> Saved Best Fusion Model")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vector_dir", type=str, default="./data/vectors")
    parser.add_argument("--grid_dir", type=str, default="./data/grids")
    parser.add_argument("--mlp_dir", type=str, required=True, help="Path to folder containing mlp_best.pt")
    parser.add_argument("--out_dir", type=str, default="./models/fusion")
    parser.add_argument("--lr", type=float, default=5e-5) # Lower LR for fusion fine-tuning
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32) # Grids use more VRAM, lower batch size
    args = parser.parse_args()
    
    train_fusion(args)