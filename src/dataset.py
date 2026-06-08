import torch
from torch.utils.data import Dataset
import pandas as pd
import rasterio
import numpy as np
import os

class SARDataset(Dataset):
    def __init__(self, csv_file, image_dir):
        # Load the CSV containing ground truth wind speeds and directions
        self.metadata = pd.read_csv(csv_file)
        self.image_dir = image_dir

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        
        # 1. Get True Labels from CSV
        true_speed = row['true_speed_ms']
        true_dir = np.radians(row['true_dir_deg'])
        
        # Convert Speed/Direction to U/V Vectors for the model to learn
        u_vector = true_speed * np.cos(true_dir)
        v_vector = true_speed * np.sin(true_dir)
        labels = torch.tensor([u_vector, v_vector], dtype=torch.float32)

        # 2. Get the corresponding SAR Image
        img_name = os.path.join(self.image_dir, row['filename'])
        
        with rasterio.open(img_name) as src:
            # Extract a 256x256 patch from the center of the TIF
            center_x, center_y = src.width // 2, src.height // 2
            window = rasterio.windows.Window(center_x - 128, center_y - 128, 256, 256)
            img_patch = src.read(1, window=window)
            
            # Normalize the SAR backscatter values between 0 and 1
            img_patch = (img_patch - np.min(img_patch)) / (np.max(img_patch) - np.min(img_patch) + 1e-8)
            img_tensor = torch.tensor(img_patch, dtype=torch.float32).unsqueeze(0) # Add channel dimension

        return img_tensor, labels