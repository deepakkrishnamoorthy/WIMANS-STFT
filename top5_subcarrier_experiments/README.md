# Top-5 Subcarrier Experiments

This folder is a side branch for testing whether Top-5 subcarrier features are useful enough to keep the compute small.

It does not modify the existing ResNet baseline code. The scripts here call the reusable training code in `resnet_baseline`.

## Feature Folders

- PCA-STFT Top-5:
  `D:\Deepak\wifi_csi\dataset\stft_top5_npy`

- No-PCA multichannel Top-5:
  `D:\Deepak\wifi_csi\dataset\stft_top5_multichannel_npy`

The multichannel folder keeps all `3 Tx x 3 Rx x 5 subcarriers = 45` streams as STFT channels. That is the cleaner test of whether Top-5 subcarriers preserve enough activity information.

## Recommended First Run

```powershell
python top5_subcarrier_experiments\run_top5_activity_set_resnet18.py --features multichannel --band all --epochs 50 --repeat 1 --batch-size 64 --lr 1e-4 --normalize log_standard
```

## Quick Smoke Run

```powershell
python top5_subcarrier_experiments\run_top5_activity_set_resnet18.py --features multichannel --band 5 --epochs 2 --repeat 1 --batch-size 32 --lr 1e-4 --normalize log_standard
```

## Dataset Audit

```powershell
python top5_subcarrier_experiments\audit_top5_dataset.py
```
