import os
import sys

import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'WiMANS-main', 'benchmark', 'wifi_csi')))
from load_data import encode_data_y


class STFTMultiChannelDataset(Dataset):
    def __init__(self, data_pd_y, data_dir, max_len=200, normalize="log_standard"):
        self.data_pd_y = data_pd_y
        self.data_dir = data_dir
        self.max_len = max_len
        self.normalize = normalize
        self.labels = encode_data_y(self.data_pd_y, "activity")
        self.file_ids = self.data_pd_y["label"].tolist()

    def __len__(self):
        return len(self.file_ids)

    def _pad_or_crop(self, spec):
        time_len = spec.shape[-1]
        if time_len < self.max_len:
            pad_len = self.max_len - time_len
            pad_width = [(0, 0)] * spec.ndim
            pad_width[-1] = (0, pad_len)
            return np.pad(spec, pad_width, mode="constant")
        if time_len > self.max_len:
            return spec[..., :self.max_len]
        return spec

    def _normalize(self, spec):
        if self.normalize == "log_standard":
            spec = np.log1p(spec)
            mean = spec.mean(axis=(-2, -1), keepdims=True)
            std = spec.std(axis=(-2, -1), keepdims=True)
            return (spec - mean) / (std + 1e-6)
        if self.normalize == "log":
            return np.log1p(spec)
        if self.normalize == "standard":
            mean = spec.mean(axis=(-2, -1), keepdims=True)
            std = spec.std(axis=(-2, -1), keepdims=True)
            return (spec - mean) / (std + 1e-6)
        if self.normalize in (None, "none"):
            return spec
        raise ValueError(f"Unknown normalization mode: {self.normalize}")

    def __getitem__(self, idx):
        file_id = self.file_ids[idx]
        file_path = os.path.join(self.data_dir, f"{file_id}.npy")
        spec = np.load(file_path).astype(np.float32)

        if spec.ndim == 2:
            spec = spec[np.newaxis, ...]
        elif spec.ndim != 3:
            raise ValueError(f"Expected 2D or 3D STFT array for {file_path}, got shape {spec.shape}")

        spec = self._pad_or_crop(spec)
        spec = self._normalize(spec).astype(np.float32)

        stft_tensor = torch.from_numpy(spec)
        label_tensor = torch.from_numpy(self.labels[idx]).float().flatten()
        return stft_tensor, label_tensor
