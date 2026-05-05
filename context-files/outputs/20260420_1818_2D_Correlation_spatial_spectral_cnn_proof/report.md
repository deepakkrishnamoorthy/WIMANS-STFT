# Experiment Report: 20260420_1818_2D_Correlation_spatial_spectral_cnn_proof

## Summary
Computed the 270x270 correlation matrix. Results show strong block-diagonal structures, meaning adjacent subcarriers within the same antenna link are highly correlated. This localized patch correlation perfectly validates the use of 2D Convolutional layers in our ML baselines, as CNNs are designed to exploit these exact local structural dependencies.

## Final Metrics
```json
{
    "num_features": 270,
    "average_adjacent_subcarrier_correlation": 0.8168184259279045
}
```
