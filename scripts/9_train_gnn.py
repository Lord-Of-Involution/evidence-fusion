#!/usr/bin/env python3
import os
import argparse
import random
import torch
import torch.optim as optim
from torch.utils.data import Sampler
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import math
from evidence.gnn_dataset import QuijotePointCloudDataset
from evidence.gnn_models import EvidenceGNN
from evidence.core import one_pop_exponential_loss, compute_posterior, l_pop_transform


class ChunkedRandomSampler(Sampler):
    def __init__(self, data_source, chunk_size=512):
        self.data_source = data_source
        self.chunk_size = chunk_size

    def __iter__(self):
        n = len(self.data_source)
        chunks =[list(range(i, min(i + self.chunk_size, n))) for i in range(0, n, self.chunk_size)]
        random.shuffle(chunks)
        for chunk in chunks:
            random.shuffle(chunk)
            for idx in chunk:
                yield idx

    def __len__(self):
        return len(self.data_source)

def extract_and_save_evidence(model, loader, device, desc):
    all_f, all_y =[],[]
    model.eval()
    with torch.no_grad():
        for data in tqdm(loader, desc=desc, leave=False):
            data = data.to(device)
            f_x = model(data)
            all_f.append(f_x.cpu())
            all_y.append(data.y.cpu())
    return torch.cat(all_f), torch.cat(all_y)

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Firing up GPUs on {device}...")

    # 把 num_subboxes 传进去！
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
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler, num_workers=16, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=16, pin_memory=True)

    # ======== 【贝叶斯先验偏置计算】 ========
    # GNN 的 dataset 里有 self.labels，直接拿来用
    n1 = train_ds.labels.sum().item()
    n0 = len(train_ds.labels) - n1
    c_prior = math.log(n1 / max(1, n0))
    print("="*60)
    print(f">>> GNN Dataset Stats: N0 = {n0}, N1 = {n1}")
    print(f">>> Bayesian Prior Bias (c) = {c_prior:.6f}")
    print("="*60)

    model = EvidenceGNN(r_link=args.r_link, hidden_dim=args.hidden_dim).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    os.makedirs(args.out_dir, exist_ok=True)
    best_loss = float('inf')
    start_epoch = 0

    if args.resume and os.path.exists(args.resume):
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['best_loss']

    for epoch in range(start_epoch, args.epochs):

        if epoch == args.bce_epochs:
            print("\n" + "="*60)
            print(f">>> [Phase Shift] Switching from BCE to 1-POP Loss at Epoch {epoch}!")

            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * 0.1
                
            print(f">>> Dropped LR to {optimizer.param_groups[0]['lr']} for exponential landscape!")            
            
            print(">>> Resetting best_loss and LR Scheduler...")
            print("="*60 + "\n")
            best_loss = float('inf')
            # 重新初始化 Scheduler，让他忘记 BCE 时期的极低 Loss！
            # 注意 GNN 这里的 patience 是 3
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=3
            )

        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Ep {epoch} Train")
        for data in pbar:
            data = data.to(device)
            optimizer.zero_grad()
            f_x = model(data)
            
            # 【物理升级：Curriculum Learning】
            if epoch < args.bce_epochs:
                # 预热期：将 log-evidence 转回 Logits，跑 BCE
                logits = l_pop_transform(f_x, alpha=2.0) + c_prior
                loss = torch.nn.BCEWithLogitsLoss()(logits, data.y.float().view_as(logits))
                desc_loss = "BCE"
            else:
                # 进阶期：切换为纯粹的 1-POP 贝叶斯精调
                loss = one_pop_exponential_loss(f_x, data.y, c=c_prior)
                desc_loss = "1POP"
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'type': desc_loss})

        model.eval()
        val_loss = 0.0; correct = 0; total = 0
        with torch.no_grad():
            for data in tqdm(val_loader, desc=f"Ep {epoch} Val", leave=False):
                data = data.to(device)
                f_x = model(data)

                if epoch < args.bce_epochs:
                    logits = l_pop_transform(f_x, alpha=2.0) + c_prior
                    loss = torch.nn.BCEWithLogitsLoss()(logits, data.y.float().view_as(logits))
                else:
                    loss = one_pop_exponential_loss(f_x, data.y.view_as(f_x), alpha=2.0, c=c_prior)

                val_loss += loss.item()
                pred = (compute_posterior(f_x, alpha=2.0, c=c_prior) > 0.5).float()
                correct += (pred.view(-1) == data.y.view(-1)).sum().item()
                total += data.y.size(0)

        avg_val_loss = val_loss / len(val_loader)
        acc = correct / total
        
        print(f"Ep {epoch} | Tr Loss: {train_loss/len(train_loader):.4f} | Val Loss: {avg_val_loss:.4f} | Acc: {acc:.4f}", flush=True)
        scheduler.step(avg_val_loss)
        
        is_best = avg_val_loss < best_loss
        if is_best:
            best_loss = avg_val_loss
            
        state = {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'best_loss': best_loss}
        torch.save(state, os.path.join(args.out_dir, "gnn_last.pt"))
        if is_best:
            torch.save(state, os.path.join(args.out_dir, "gnn_best.pt"))

    # ==========================================================
    # 提取 Evidence
    # ==========================================================
    print("\n>>> Training finished! Loading BEST model to extract pure Evidence...")
    model.load_state_dict(torch.load(os.path.join(args.out_dir, "gnn_best.pt"), map_location=device)['model_state_dict'])
    
    ext_train_loader = DataLoader(train_ds, batch_size=args.batch_size*2, shuffle=False, num_workers=16, pin_memory=True)
    ext_val_loader   = DataLoader(val_ds, batch_size=args.batch_size*2, shuffle=False, num_workers=16, pin_memory=True)

    tr_f, tr_y = extract_and_save_evidence(model, ext_train_loader, device, "Extract Train GNN")
    val_f, val_y = extract_and_save_evidence(model, ext_val_loader, device, "Extract Val GNN")

    cache_path = os.path.join(args.out_dir, "gnn_evidence.pt")
    torch.save({"train": {"f_x": tr_f, "y": tr_y}, "val": {"f_x": val_f, "y": val_y}}, cache_path)
    print(f"GNN Evidence seamlessly saved to {cache_path}!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vector_dir", type=str, default="/work/hdd/bdne/jdong8/fusion_data/vectors")
    parser.add_argument("--catalog_dir", type=str, default="/work/hdd/bdne/jdong8/fusion_data/catalogs")
    parser.add_argument("--out_dir", type=str, default="/work/hdd/bdne/jdong8/fusion_models/gnn_solo")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--bce_epochs", type=int, default=10, help="Number of epochs to train with BCE loss before switching to 1-POP")
    parser.add_argument("--batch_size", type=int, default=64) 
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--subbox_size", type=float, default=60.0)
    parser.add_argument("--num_subboxes", type=int, default=8)
    parser.add_argument("--r_link", type=float, default=20.0, help="Graph linking radius")
    parser.add_argument("--hidden_dim", type=int, default=64, help="GNN hidden channels")
    args = parser.parse_args()
    main(args)