# Experiment Report: 20260420_1754_Phase_Sanitization_baseline

## Summary
Successfully loaded raw complex CSI. The raw phase showed random wraps between -pi and pi due to hardware timing offsets. Applying the linear transformation (unwrapping + slope/intercept removal) successfully stabilized the phase across subcarriers.

## Final Metrics
```json
{
    "num_packets": 2901,
    "csi_matrix_shape": "(3, 3, 30)"
}
```
