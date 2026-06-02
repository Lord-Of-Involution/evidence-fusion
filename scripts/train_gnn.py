#!/usr/bin/env python3
import os
import argparse
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Sampler
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from evidence.gnn_dataset import QuijotePointCloudDataset
from evidence.gnn_models import MicroTopologyGNN
from evidence.models import EvidenceMLP, EarlyFusionNetwork
from evidence.core import one_pop_exponential_loss, compute_posterior, l_pop_transform

# ==============================================================================
# 0. Solo GNN Wrapper (Since MicroTopologyGNN now outputs pure features)
# ==============================================================================
class SoloEvidenceGNN(nn.Module):
    def __init__(self, gnn_backbone):
        super().__init__()
        self.gnn = gnn_backbone
        # Pure topological evidence head
        self.head = nn.Sequential(
            nn.Linear(self.gnn.output_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
        self.mil_temp = nn.Parameter(torch.tensor(1.0))

    def forward(self, data):
        feat = self.gnn(
            x=data.x, 
            pos=data.pos, 
            batch=data.batch, 
            sub_batch=data.sub_batch, 
            los=getattr(data, 'los', None)
        )
        global_evidence = self.head(feat)
        safe_temp = torch.clamp(self.mil_temp, min=0.1)
        return global_evidence / safe_temp

# ==============================================================================
# 1. Chunked Sampler for Determinism
# ==============================================================================
class ChunkedRandomSampler(Sampler):
    def __init__(self, data_source, chunk_size=512):
        self.data_source = data_source
        self.chunk_size = chunk_size

    def __iter__(self):
        n = len(self.data_source)
        chunks = [list(range(i, min(i + self.chunk_size, n))) for i in range(0, n, self.chunk_size)]
        random.shuffle(chunks)
        for chunk in chunks:
            random.shuffle(chunk)
            for idx in chunk:
                yield idx
    def __len__(self):
        return len(self.data_source)

# ==============================================================================
# 2. Evidence Extraction Engine
# ==============================================================================
def extract_and_save_evidence(model, loader, device, desc):
    all_f, all_y = [], []
    model.eval()
    with torch.no_grad():
        for data in tqdm(loader, desc=desc, leave=False):
            data = data.to(device)
            f_x = model(data)
            all_f.append(f_x.cpu())
            all_y.append(data.y.cpu())
    return torch.cat(all_f), torch.cat(all_y)

# ==============================================================================
# 3. Main Training Pipeline
# ==============================================================================
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n>>> Firing up GPUs on {device}... Engaging Topology: {args.geometry.upper()}")

    train_ds = QuijotePointCloudDataset(
        vector_pt_path=os.path.join(args.vector_dir, "train_data.pt"),
        catalog_h5_path=os.path.join(args.catalog_dir, "train_catalogs.h5"),
        centers_pt_path=os.path.join(args.catalog_dir, "train_centers.pt"),
        subbox_size=args.subbox_size,
        num_subboxes=args.num_subboxes
    )
    val_ds = QuijotePointCloudDataset(
        vector_pt_path=os.path.join(args.vector_dir, "val_data.pt"),
        catalog_h5_path=os.path.join(args.catalog_dir, "val_catalogs.h5"),
        centers_pt_path=os.path.join(args.catalog_dir, "val_centers.pt"),
        subbox_size=args.subbox_size,
        num_subboxes=args.num_subboxes
    )

    train_sampler = ChunkedRandomSampler(train_ds, chunk_size=512)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Rigorous Bayesian Prior Calculation
    n1 = train_ds.labels.sum().item()
    n0 = len(train_ds.labels) - n1
    c_prior = math.log(n1 / max(1, n0))
    print("\n" + "="*60)
    print(f">>> Dataset Stats: N0 = {n0}, N1 = {n1}")
    print(f">>> Bayesian Prior Bias (c) = {c_prior:.6f}")
    print("="*60 + "\n")

    # ==========================================================================
    # ARCHITECTURE ROUTING (STRICTLY DECOUPLED)
    # ==========================================================================
    gnn_core = MicroTopologyGNN(r_link=args.r_link, hidden_dim=args.hidden_dim, geometry=args.geometry)
    
    if args.early_fusion:
        print(f"\n>>> [ARCHITECTURE] EARLY FUSION ENGAGED. Wiring EvidenceMLP (dim={args.vec_dim}) with TopologyGNN.")
        # Instantiate the identical MLP used in your solo training script
        mlp_core = EvidenceMLP(
            input_size=args.vec_dim, 
            hidden_sizes=[512, 512, 128], 
            batchnorm=True
        )
        model = EarlyFusionNetwork(mlp_backbone=mlp_core, gnn_backbone=gnn_core, freeze_mlp=False).to(device)
    else:
        print("\n>>> [ARCHITECTURE] PURE SOLO GNN ENGAGED. Blind to global spectra.")
        model = SoloEvidenceGNN(gnn_backbone=gnn_core).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=6)

    os.makedirs(args.out_dir, exist_ok=True)
    best_loss = float('inf')
    start_epoch = 0

    if args.resume and os.path.exists(args.resume):
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['best_loss']
        print(f">>> Resuming from Epoch {start_epoch} (Best Loss: {best_loss:.4f})")

    for epoch in range(start_epoch, args.epochs):

        if epoch == args.bce_epochs:
            print("\n" + "="*60)
            print(f">>> [Phase Shift] Switching from BCE to 1-POP Loss at Epoch {epoch}!")
            optimizer.param_groups[0]['lr'] = args.lr * 0.5 
            print(f">>> Resetting LR to {optimizer.param_groups[0]['lr']} for exponential landscape!")            
            best_loss = float('inf')
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
            print("="*60 + "\n")

        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Ep {epoch:03d} Train", dynamic_ncols=True)
        
        for data in pbar:
            data = data.to(device)
            optimizer.zero_grad()
            f_x = model(data)
            
            if epoch < args.bce_epochs:
                logits = l_pop_transform(f_x, alpha=2.0) + c_prior
                loss = torch.nn.BCEWithLogitsLoss()(logits, data.y.float().view_as(logits))
                desc_loss = "BCE"
            else:
                loss = one_pop_exponential_loss(f_x, data.y.view_as(f_x), alpha=2.0, c=c_prior)
                desc_loss = "1POP"
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'type': desc_loss})

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                f_x = model(data)

                if epoch < args.bce_epochs:
                    logits = l_pop_transform(f_x, alpha=2.0) + c_prior
                    loss = torch.nn.BCEWithLogitsLoss()(logits, data.y.float().view_as(logits))
                else:
                    loss = one_pop_exponential_loss(f_x, data.y.view_as(f_x), alpha=2.0, c=c_prior)
                    
                val_loss += loss.item()
                
                probs = torch.sigmoid(l_pop_transform(f_x, alpha=2.0) + c_prior)
                pred = (probs > 0.5).float()
                correct += (pred.view(-1) == data.y.view(-1)).sum().item()
                total += data.y.size(0)

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        acc = correct / total
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Ep {epoch:03d} | LR: {current_lr:.2e} | Tr Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | Acc: {acc:.6f}", flush=True)
        scheduler.step(avg_val_loss)
        
        is_best = avg_val_loss < best_loss
        if is_best:
            best_loss = avg_val_loss
            
        state = {
            'epoch': epoch, 
            'model_state_dict': model.state_dict(), 
            'optimizer_state_dict': optimizer.state_dict(), 
            'best_loss': best_loss
        }
        torch.save(state, os.path.join(args.out_dir, "gnn_last.pt"))
        if is_best:
            torch.save(state, os.path.join(args.out_dir, "gnn_best.pt"))

    # 5. Extract Pure Evidence
    print("\n" + "="*60)
    print(">>> Training finished! Loading BEST model to extract pure Evidence...")
    
    if args.epochs > 0:
        model.load_state_dict(torch.load(os.path.join(args.out_dir, "gnn_best.pt"), map_location=device)['model_state_dict'])
    
    ext_train_loader = DataLoader(train_ds, batch_size=args.batch_size*2, shuffle=False, num_workers=4, pin_memory=True)
    ext_val_loader   = DataLoader(val_ds, batch_size=args.batch_size*2, shuffle=False, num_workers=4, pin_memory=True)

    tr_f, tr_y = extract_and_save_evidence(model, ext_train_loader, device, "Extract Train GNN")
    val_f, val_y = extract_and_save_evidence(model, ext_val_loader, device, "Extract Val GNN")

    cache_path = os.path.join(args.out_dir, "gnn_evidence.pt")
    torch.save({"train": {"f_gnn": tr_f, "y": tr_y}, "val": {"f_gnn": val_f, "y": val_y}}, cache_path)
    print(f">>> Checkmate. Solo GNN Evidence seamlessly saved to {cache_path}!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vector_dir", type=str, default="/work/hdd/bdne/jdong8/fusion_data_purified/vectors")
    parser.add_argument("--catalog_dir", type=str, default="/work/hdd/bdne/jdong8/fusion_data_purified/catalogs")
    parser.add_argument("--out_dir", type=str, default="/work/hdd/bdne/jdong8/fusion_models/gnn_solo")
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--bce_epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=256) j
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--subbox_size", type=float, default=60.0)
    parser.add_argument("--num_subboxes", type=int, default=8)
    parser.add_argument("--r_link", type=float, default=20.0, help="Graph linking radius")
    parser.add_argument("--hidden_dim", type=int, default=64, help="GNN hidden channels")
    parser.add_argument("--geometry", type=str, default="lightcone", choices=["plane_parallel", "lightcone"])
    parser.add_argument("--vec_dim", type=int, default=400, help="Dimension of macroscopic spectra (P, B)")
    parser.add_argument("--early_fusion", action="store_true", help="If flagged, wires MLP and GNN together for early fusion.")
    
    args = parser.parse_args()
    main(args)