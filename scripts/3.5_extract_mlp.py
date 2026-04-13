#!/usr/bin/env python3
import os
import json
import torch
import argparse
from torch.utils.data import DataLoader, Dataset
from evidence.models import EvidenceMLP

# 极简向量 Dataset
class VectorDataset(Dataset):
    def __init__(self, path):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        self.x = payload['data'] if 'data' in payload else payload['stats']
        self.y = payload['labels']
        self.x = self.x.float()
        self.y = self.y.float().view(-1, 1)
        if self.x.shape[0] < self.x.shape[1] and self.x.shape[0] <= 2000:
             self.x = self.x.T
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.x[idx], self.y[idx]

def extract(model, loader, device, desc):
    all_f, all_y = [],[]
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            f_x = model(x)
            all_f.append(f_x.cpu())
            all_y.append(y.cpu())
    return torch.cat(all_f), torch.cat(all_y)

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Extracting MLP Evidence on {device}...")

    train_ds = VectorDataset(os.path.join(args.vector_dir, "train_data.pt"))
    val_ds = VectorDataset(os.path.join(args.vector_dir, "val_data.pt"))

    # 绝对核心：shuffle=False
    train_dl = DataLoader(train_ds, batch_size=1024, shuffle=False)
    val_dl = DataLoader(val_ds, batch_size=1024, shuffle=False)

    with open(os.path.join(args.mlp_dir, "mlp_config.json"), "r") as f:
        conf = json.load(f)
        
    model = EvidenceMLP(input_size=conf["input_dim"], hidden_sizes=conf["hidden_sizes"], dropout=conf["dropout"]).to(device)
    model.load_state_dict(torch.load(os.path.join(args.mlp_dir, "mlp_best.pt"), map_location=device))
    
    tr_f, tr_y = extract(model, train_dl, device, "MLP Train")
    val_f, val_y = extract(model, val_dl, device, "MLP Val")

    cache_path = os.path.join(args.mlp_dir, "mlp_evidence.pt")
    torch.save({
        "train": {"f_x": tr_f, "y": tr_y},
        "val": {"f_x": val_f, "y": val_y}
    }, cache_path)
    
    print(f"Done! MLP evidence saved to {cache_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vector_dir", type=str, default="/work/hdd/bdne/jdong8/fusion_data/vectors")
    parser.add_argument("--mlp_dir", type=str, default="/work/hdd/bdne/jdong8/fusion_models/mlp")
    args = parser.parse_args()
    main(args)