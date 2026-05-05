# Experiment Report: 20260421_0859_Frequency_Comparison_2.4G_vs_5G

## Summary
Compared STFT spectrograms for the 2.4 GHz and 5 GHz bands. The 5 GHz band has a smaller wavelength, resulting in larger, more distinct Doppler shifts for the same physical movement velocity. This higher resolution makes 5 GHz preferable for subtle motion detection, though 2.4 GHz penetrates walls better. Models should ideally utilize multi-band fusion if both are available, or adjust convolution kernel sizes depending on the operating frequency.

## Final Metrics
```json
{
    "sample_24g": "act_37_1",
    "sample_5g": "act_55_1"
}
```
