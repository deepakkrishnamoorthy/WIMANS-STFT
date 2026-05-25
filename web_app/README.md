# WiMANS STFT Multi-Head Demo

Offline Gradio demo for the Top-5 STFT multi-head CLSTM model.

The app supports two modes:

1. Select a holdout/test sample from the annotation split.
2. Upload a `.npy` STFT clip manually.

It visualizes:

- Scene-level activity probabilities.
- User occupancy probabilities.
- User-slot activity probabilities.
- STFT heatmaps from the uploaded/selected clip.
- Ground-truth labels when the selected file exists in `annotation.csv`.

This is not real-time sensing. It is an inference and explainability viewer for saved model checkpoints.

Run:

```powershell
python web_app\app.py
```
