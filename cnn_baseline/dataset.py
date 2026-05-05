import os
import sys
import torch
import numpy as np
from torch.utils.data import Dataset

# Add WiMANS-main to path so we can reuse their label encoding
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'WiMANS-main', 'benchmark', 'wifi_csi')))
from load_data import encode_data_y

class STFTDataset(Dataset):
    def __init__(self, data_pd_y, data_dir, max_len=200):
        self.data_pd_y = data_pd_y
        self.data_dir = data_dir
        self.max_len = max_len
        
        # Extract labels using the original repo's encoding logic
        # This returns a shape of (N, 6 users, 9 activities)
        self.labels = encode_data_y(self.data_pd_y, 'activity')
        self.file_ids = self.data_pd_y['label'].tolist()
        
    def __len__(self):
        return len(self.file_ids)
        
    def __getitem__(self, idx):
        file_id = self.file_ids[idx]
        file_path = os.path.join(self.data_dir, f"{file_id}.npy")
        
        # Load the extracted (129, ~179) STFT float matrix
        stft_data = np.load(file_path) 
        
        # Dynamic padding along the time axis (axis=1) to ensure uniform width
        T = stft_data.shape[1]
        if T < self.max_len:
            pad_len = self.max_len - T
            # Zero-pad the right side
            stft_data = np.pad(stft_data, ((0, 0), (0, pad_len)), mode='constant')
        elif T > self.max_len:
            # Truncate if somehow larger than expected
            stft_data = stft_data[:, :self.max_len]
            
        # Add channel dimension: (1, 129, 200)
        stft_tensor = torch.from_numpy(stft_data).float().unsqueeze(0)
        
        # Flatten the 6x9 label matrix into a 54-dimensional multi-label vector
        label_tensor = torch.from_numpy(self.labels[idx]).float().flatten() 
        
        return stft_tensor, label_tensor
