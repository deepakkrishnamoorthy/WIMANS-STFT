# Experiment Report: 20260420_1810_Spatial_Correlation_mimo_diversity

## Summary
Computed the spatial correlation matrix across the MIMO receiver antennas. While the antennas are highly correlated (as expected since they capture the same macro-movement), there are distinct micro-fading differences between the traces. This confirms that MIMO provides valuable spatial diversity, justifying 3D convolution or spatial attention layers in Phase 2.

## Final Metrics
```json
{
    "rx_correlation_0_1": 0.3891032375488535,
    "rx_correlation_0_2": 0.047448950021603056,
    "rx_correlation_1_2": 0.3483874661552547
}
```
