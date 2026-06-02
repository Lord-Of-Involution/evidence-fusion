import torch
import h5py
import numpy as np
from torch_geometric.data import Data, Dataset

class QuijotePointCloudDataset(Dataset):
    def __init__(self, vector_pt_path, catalog_h5_path, centers_pt_path, subbox_size=60.0, num_subboxes=8, max_nodes_per_box=800):
        super().__init__()
        payload = torch.load(vector_pt_path, map_location='cpu', weights_only=False)
        self.labels = payload['labels']
        self.vectors = payload['data'] if 'data' in payload else payload['stats']
        
        self.h5_file_path = catalog_h5_path
        self.subbox_size = subbox_size
        self.M = num_subboxes 
        self.max_nodes = max_nodes_per_box  
        
        all_centers = torch.load(centers_pt_path, map_location='cpu', weights_only=False)
        
        if all_centers.shape[1] < self.M:
            raise ValueError(f"[FATAL] Precomputed centers ({all_centers.shape[1]}) < requested ({self.M}).")
            
        self.centers = all_centers[:, :self.M, :]
        self.h5_handle = None 

    def len(self):
        return len(self.labels)

    def get(self, idx):
            try:
                with h5py.File(self.h5_file_path, 'r') as h5_handle:
                    grp = h5_handle[str(idx)]
                    
                    label = self.labels[idx]
                    vec = self.vectors[idx]
                    centers = self.centers[idx].numpy()

                    all_pos = []
                    all_x = []
                    all_sub_batch = []
                    all_los = [] 

                    for m_idx, c in enumerate(centers):
                        try:
                            sub_pos = grp[f"sub_{m_idx}"][:]
                        except KeyError:
                            sub_pos = np.array([c, c + 1e-3, c - 1e-3], dtype=np.float32)

                        # Center the subbox immediately for accurate distance metrics
                        sub_pos = sub_pos - c
                        num_gals = len(sub_pos)
                        
                        if num_gals > self.max_nodes:
                            # [CRITICAL FIX: TOPOLOGICAL INTEGRITY]
                            # Stop destroying the metric space with random dropout.
                            # Sort by distance to the origin (center) and keep the dense core.
                            dists = np.linalg.norm(sub_pos, axis=1)
                            keep_idx = np.argsort(dists)[:self.max_nodes]
                            sub_pos = sub_pos[keep_idx]

                        # Absolute Line-of-Sight vector pointing to the subbox center
                        los_vec = c / (np.linalg.norm(c) + 1e-8)
                        
                        all_pos.append(torch.tensor(sub_pos, dtype=torch.float32))
                        all_x.append(torch.ones((len(sub_pos), 1), dtype=torch.float32))
                        all_sub_batch.append(torch.full((len(sub_pos),), m_idx, dtype=torch.long))
                        all_los.append(torch.tensor(los_vec, dtype=torch.float32).repeat(len(sub_pos), 1))

            except (KeyError, OSError):
                return self.get(np.random.randint(0, self.len()))

            pos_tensor = torch.cat(all_pos, dim=0)
            x_tensor = torch.cat(all_x, dim=0)
            sub_batch_tensor = torch.cat(all_sub_batch, dim=0)
            los_tensor = torch.cat(all_los, dim=0)
            
            return Data(x=x_tensor, pos=pos_tensor, y=label, vec=vec, sub_batch=sub_batch_tensor, los=los_tensor)