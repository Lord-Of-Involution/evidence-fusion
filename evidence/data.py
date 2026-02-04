import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import os
import numpy as np

# ==============================================================================
# 1. Base Vector Dataset (For MLP)
# ==============================================================================
class VectorDataset(Dataset):
    def __init__(self, path):
        super().__init__()

        payload = torch.load(path, map_location="cpu")
        self.x = payload['data'].float()
        self.labels = payload['labels'].float().view(-1, 1)
        

        if self.x.shape[0] == 800 and self.x.shape[1] > 800:
             self.x = self.x.T

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.x[idx], self.labels[idx]

# ==============================================================================
# 2. Fusion Dataset (For Fusion Network)
# ==============================================================================
class FusionDataset(Dataset):
    """
    Handles Multi-modal data: (3D Grid, Vector) -> Label
    
    New Feature:
    - Loads Grids from a large .npy file using Memory Mapping.
    - Zero RAM overhead.
    """
    def __init__(self, vector_path, grid_path, transform_grid=True):
        # 1. Load Vectors (In-Memory)
        self.vector_ds = VectorDataset(vector_path)
        
        # 2. Load Grids (Lazy Mapped)
        if not os.path.exists(grid_path):
             raise FileNotFoundError(f"Grid file not found: {grid_path}")
             
        print(f"Opening Grids (MemMap) from {os.path.basename(grid_path)}...")
        
        # mode='r' means read-only, existing file.
        # It creates a window on the disk without loading everything.
        self.grids = np.load(grid_path, mmap_mode='r')
            
        # 3. Grid Preprocessing
        self.transform_grid = transform_grid
        if self.transform_grid:
            self.shift = 1.1 

        # Sanity Check
        assert len(self.vector_ds) == self.grids.shape[0], \
            f"Size Mismatch! Vectors: {len(self.vector_ds)}, Grids: {self.grids.shape[0]}"

    def __len__(self):
        return len(self.vector_ds)

    def __getitem__(self, idx):
        # Get Vector and Label
        vec, label = self.vector_ds[idx]
        
        # Get Grid from Disk
        # Because self.grids is memmapped, this reads ONLY this specific 64^3 block from SSD.
        # We must copy() it to turn it into a real Tensor in RAM.
        grid_np = self.grids[idx].copy() 
        grid = torch.from_numpy(grid_np)
        
        # On-the-fly Log Transform
        if self.transform_grid:
            grid = torch.log(grid + self.shift)
            
        return (grid, vec), label
# ==============================================================================
# 3. Helper: Balanced Loader
# ==============================================================================
def get_balanced_loader(dataset, batch_size, num_workers=0, **kwargs):
    """
    Creates a DataLoader with WeightedRandomSampler.
    This ensures each batch has roughly 50/50 class distribution,
    correcting for any prior bias in the dataset size.
    """
    # 1. Extract labels (assuming dataset has .labels attribute)
    if hasattr(dataset, 'labels'):
        targets = dataset.labels.view(-1).long().numpy()
    elif hasattr(dataset, 'vector_ds'): # For FusionDataset
        targets = dataset.vector_ds.labels.view(-1).long().numpy()
    else:
        raise ValueError("Dataset must have 'labels' attribute for balanced sampling.")

    # 2. Count classes
    class_counts = np.bincount(targets)
    
    # 3. Calculate weights (Inverse frequency)
    # Weight for class i = 1 / count[i]
    class_weights = 1. / class_counts
    
    # 4. Assign weight to each sample
    sample_weights = torch.tensor([class_weights[t] for t in targets]).float()
    
    # 5. Create Sampler
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    # 6. Create Loader
    # Note: shuffle must be False when using sampler
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        sampler=sampler, 
        num_workers=num_workers,
        pin_memory=True,
        **kwargs
    )
    
    print(f"Created Balanced Loader. Class Counts: {class_counts} -> Weights: {class_weights}")
    return loader