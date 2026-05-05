# Experiment Report: 20260420_1818_Dynamic_PCA_realtime_inference_viability

## Summary
Compared Static PCA against Dynamic Sliding Window PCA (window=300). While Dynamic PCA roughly tracks the motion, it suffers from severe boundary discontinuities and scaling mismatches at window edges (MSE: 111.25). This indicates that while PCA is great for offline EDA, real-time inference will likely require a Neural Autoencoder or continuous filtering rather than naive sliding window PCA.

## Final Metrics
```json
{
    "mse": 111.24590670389006
}
```
