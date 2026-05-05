import os
import sys

import numpy as np
import torch
from torch.utils.data import Dataset

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(ROOT, "WiMANS-main", "benchmark", "wifi_csi"))
from load_data import encode_data_y


class MultiHeadSTFTDataset(Dataset):
    def __init__(
        self,
        data_pd_y,
        data_dir,
        max_len=200,
        normalize="log_standard",
        augment=False,
        time_mask_width=0,
        freq_mask_width=0,
        channel_drop_prob=0.0,
        noise_std=0.0,
        time_shift=0,
    ):
        self.data_pd_y = data_pd_y
        self.data_dir = data_dir
        self.max_len = max_len
        self.normalize = normalize
        self.augment = augment
        self.time_mask_width = time_mask_width
        self.freq_mask_width = freq_mask_width
        self.channel_drop_prob = channel_drop_prob
        self.noise_std = noise_std
        self.time_shift = time_shift
        self.slot_activity = encode_data_y(self.data_pd_y, "activity").astype(np.float32)
        self.activity_set = (self.slot_activity.sum(axis=1) > 0).astype(np.float32)
        self.occupancy = (self.slot_activity.sum(axis=2) > 0).astype(np.float32)
        self.file_ids = self.data_pd_y["label"].tolist()

    def __len__(self):
        return len(self.file_ids)

    def _pad_or_crop(self, spec):
        time_len = spec.shape[-1]
        if time_len < self.max_len:
            pad_width = [(0, 0)] * spec.ndim
            pad_width[-1] = (0, self.max_len - time_len)
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

    def _augment(self, spec):
        if self.time_shift > 0:
            shift = np.random.randint(-self.time_shift, self.time_shift + 1)
            if shift:
                spec = np.roll(spec, shift, axis=-1)

        if self.time_mask_width > 0 and spec.shape[-1] > 1:
            width = np.random.randint(1, min(self.time_mask_width, spec.shape[-1]) + 1)
            start = np.random.randint(0, spec.shape[-1] - width + 1)
            spec[..., start:start + width] = 0.0

        if self.freq_mask_width > 0 and spec.shape[-2] > 1:
            width = np.random.randint(1, min(self.freq_mask_width, spec.shape[-2]) + 1)
            start = np.random.randint(0, spec.shape[-2] - width + 1)
            spec[..., start:start + width, :] = 0.0

        if self.channel_drop_prob > 0.0 and spec.shape[0] > 1:
            drop = np.random.rand(spec.shape[0]) < self.channel_drop_prob
            if drop.all():
                drop[np.random.randint(0, spec.shape[0])] = False
            spec[drop, :, :] = 0.0

        if self.noise_std > 0.0:
            spec = spec + np.random.normal(0.0, self.noise_std, size=spec.shape).astype(np.float32)

        return spec

    def __getitem__(self, idx):
        file_id = self.file_ids[idx]
        path = os.path.join(self.data_dir, f"{file_id}.npy")
        spec = np.load(path).astype(np.float32)
        if spec.ndim == 2:
            spec = spec[np.newaxis, ...]
        elif spec.ndim != 3:
            raise ValueError(f"Expected 2D or 3D STFT array for {path}, got {spec.shape}")

        spec = self._pad_or_crop(spec)
        spec = self._normalize(spec).astype(np.float32)
        if self.augment:
            spec = self._augment(spec).astype(np.float32)
        return {
            "x": torch.from_numpy(spec),
            "activity_set": torch.from_numpy(self.activity_set[idx]).float(),
            "occupancy": torch.from_numpy(self.occupancy[idx]).float(),
            "slot_activity": torch.from_numpy(self.slot_activity[idx].reshape(-1)).float(),
        }
