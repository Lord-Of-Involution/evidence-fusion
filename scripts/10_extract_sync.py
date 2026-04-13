#!/usr/bin/env python3
import os
import json
import argparse
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from evidence.gnn_dataset import QuijotePointCloudDataset
from evidence.gnn_models import EvidenceGNN
from evidence.models import EvidenceMLP

def extract_synced(loader, mlp, gnn, device, desc):
    all_mlp, all_gnn, all_y = [], [],[]
    with torch.no_grad():
        for data in tqdm(loader, desc=desc):
            data = data.to(device)
            
            # 【修复！破解 PyG 的贪吃蛇陷阱】：把 1D 的拼接向量还原为 [BatchSize, 800] 的矩阵
            vec_matrix = data.vec.view(data.num_graphs, -1)
            f_mlp = mlp(vec_matrix)
            
            f_gnn = gnn(data)
            
            all_mlp.append(f_mlp.cpu())
            all_gnn.append(f_gnn.cpu())
            all_y.append(data.y.cpu())
            
    return torch.cat(all_mlp), torch.cat(all_gnn), torch.cat(all_y)

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Firing up {device} for SYNCED EVIDENCE EXTRACTION...")

    train_ds = QuijotePointCloudDataset(
        vector_pt_path=os.path.join(args.vector_dir, "train_data.pt"),
        catalog_h5_path=os.path.join(args.catalog_dir, "train_catalogs.h5"),
        centers_pt_path=os.path.join(args.catalog_dir, "train_centers.pt"),
        subbox_size=args.subbox_size
    )
    val_ds = QuijotePointCloudDataset(
        vector_pt_path=os.path.join(args.vector_dir, "val_data.pt"),
        catalog_h5_path=os.path.join(args.catalog_dir, "val_catalogs.h5"),
        centers_pt_path=os.path.join(args.catalog_dir, "val_centers.pt"),
        subbox_size=args.subbox_size
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=16, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=16, pin_memory=True)

    print(f"Loading Frozen MLP from {args.mlp_dir}...")
    with open(os.path.join(args.mlp_dir, "mlp_config.json"), "r") as f:
        mlp_conf = json.load(f)
    mlp = EvidenceMLP(
        input_size=mlp_conf["input_dim"], hidden_sizes=mlp_conf["hidden_sizes"], dropout=mlp_conf["dropout"]
    ).to(device)
    mlp.load_state_dict(torch.load(os.path.join(args.mlp_dir, "mlp_best.pt"), map_location=device, weights_only=False))
    mlp.eval()

    print(f"Loading Frozen GNN from {args.gnn_dir}...")
    gnn = EvidenceGNN(r_link=10.0, hidden_dim=64).to(device)
    gnn.load_state_dict(torch.load(os.path.join(args.gnn_dir, "gnn_best.pt"), map_location=device, weights_only=False)['model_state_dict'])
    gnn.eval()

    tr_mlp, tr_gnn, tr_y = extract_synced(train_loader, mlp, gnn, device, "Extracting Train Set")
    val_mlp, val_gnn, val_y = extract_synced(val_loader, mlp, gnn, device, "Extracting Val Set")

    os.makedirs(args.out_dir, exist_ok=True)
    cache_path = os.path.join(args.out_dir, "sync_evidence.pt")
    torch.save({
        "train": {"f_mlp": tr_mlp, "f_gnn": tr_gnn, "y": tr_y},
        "val": {"f_mlp": val_mlp, "f_gnn": val_gnn, "y": val_y}
    }, cache_path)
    print(f"Extraction Complete! Saved 100% aligned evidence to {cache_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vector_dir", type=str, default="/work/hdd/bdne/jdong8/fusion_data/vectors")
    parser.add_argument("--catalog_dir", type=str, default="/work/hdd/bdne/jdong8/fusion_data/catalogs")
    parser.add_argument("--mlp_dir", type=str, default="/work/hdd/bdne/jdong8/fusion_models/mlp")
    parser.add_argument("--gnn_dir", type=str, default="/work/hdd/bdne/jdong8/fusion_models/gnn_solo")
    parser.add_argument("--out_dir", type=str, default="/work/hdd/bdne/jdong8/fusion_models/ultimate_fusion")
    parser.add_argument("--batch_size", type=int, default=128) 
    parser.add_argument("--subbox_size", type=float, default=50.0)
    args = parser.parse_args()
    main(args)