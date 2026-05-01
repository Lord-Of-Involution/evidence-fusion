import torch
import h5py
import numpy as np
from torch_geometric.data import Data, Dataset

class QuijotePointCloudDataset(Dataset):
    def __init__(self, vector_pt_path, catalog_h5_path, centers_pt_path, subbox_size=60.0, num_subboxes=8):
        super().__init__()
        payload = torch.load(vector_pt_path, map_location='cpu', weights_only=False)
        self.labels = payload['labels']
        self.vectors = payload['data'] if 'data' in payload else payload['stats']
        
        self.h5_file_path = catalog_h5_path
        self.subbox_size = subbox_size
        self.M = num_subboxes 
        
        # 读入所有中心点
        all_centers = torch.load(centers_pt_path, map_location='cpu', weights_only=False)
        
        # 【护城河】：检查预计算的框够不够
        if all_centers.shape[1] < self.M:
            raise ValueError(f"[FATAL] 预计算的中心点只有 {all_centers.shape[1]} 个，但你请求了 {self.M} 个！请重新跑 2.5 脚本。")
            
        # 动态截取前 M 个，极其灵活！
        self.centers = all_centers[:, :self.M, :]
        
        self.h5_handle = None 

    def len(self):
        return len(self.labels)

    def get(self, idx):
        if self.h5_handle is None:
            self.h5_handle = h5py.File(self.h5_file_path, 'r', swmr=True)

        try:
            pos_rsd = self.h5_handle[str(idx)]['pos_rsd'][:]
        except (KeyError, OSError):
            return self.get(np.random.randint(1, self.len()))

        label = self.labels[idx]
        vec = self.vectors[idx]

        # 极速获取中心点
        centers = self.centers[idx].numpy()

        all_pos =[]
        all_x = []
        all_sub_batch =[]

        for m_idx, c in enumerate(centers):
            start = c - (self.subbox_size / 2.0)
            
            shifted_pos = (pos_rsd - start) % 1000.0
            mask = (shifted_pos[:, 0] <= self.subbox_size) & \
                   (shifted_pos[:, 1] <= self.subbox_size) & \
                   (shifted_pos[:, 2] <= self.subbox_size)
            sub_pos = shifted_pos[mask]
            
            if len(sub_pos) < 3:
                sub_pos = np.array([[0.0, 0.0, 0.0],[1.0, 1.0, 1.0]], dtype=np.float32)
            
            sub_pos = sub_pos - (self.subbox_size / 2.0)
            
            all_pos.append(torch.tensor(sub_pos, dtype=torch.float32))
            all_x.append(torch.ones((len(sub_pos), 1), dtype=torch.float32))
            all_sub_batch.append(torch.full((len(sub_pos),), m_idx, dtype=torch.long))

        pos_tensor = torch.cat(all_pos, dim=0)
        x_tensor = torch.cat(all_x, dim=0)
        sub_batch_tensor = torch.cat(all_sub_batch, dim=0)
        
        return Data(x=x_tensor, pos=pos_tensor, y=label, vec=vec, sub_batch=sub_batch_tensor)