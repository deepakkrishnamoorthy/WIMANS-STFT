# Experiment Report: 20260420_1802_STFT_Doppler_pca_amplitude

## Summary
Successfully applied PCA on the CSI amplitude to extract the dominant movement component (PC1). The STFT spectrogram of PC1 reveals clear micro-Doppler signatures corresponding to dynamic human motion. This confirms that Time-Frequency analysis (spectrograms) provides rich feature representations for CNN/ViT architectures.

## Final Metrics
```json
{
    "explained_variance_ratio_pc1": 0.30916266494386435,
    "explained_variance_ratio_pc2": 0.1797878206431321,
    "stft_freq_bins": 65,
    "stft_time_bins": 364
}
```
