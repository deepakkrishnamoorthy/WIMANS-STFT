# Multi-Head Top-5 STFT Experiments

This folder tests the hard user-slot activity task with explicit auxiliary supervision.

Model objective:

```text
shared 45-channel STFT encoder
-> activity-set head: 9 labels
-> occupancy head: 6 labels
-> user-slot activity head: 54 labels
```

The experiment uses existing Top-5 STFT feature folders. It does not create or modify datasets.

Default feature folder:

```text
D:\Deepak\wifi_csi\dataset\stft_top5_multichannel_npy
```

Recommended first run:

```powershell
python multi_head_experiments\train_multi_head_top5.py --features multichannel --model resnet18 --band 5 --epochs 50 --repeat 1 --batch-size 64 --lr 1e-4 --normalize log_standard
```

Compare with:

```powershell
python multi_head_experiments\train_multi_head_top5.py --features multichannel --model clstm --band 5 --epochs 50 --repeat 1 --batch-size 64 --lr 1e-4 --normalize log_standard
```

THAT-style comparison:

```powershell
python multi_head_experiments\train_multi_head_top5.py --features multichannel --model that_style --band 5 --epochs 50 --repeat 1 --batch-size 64 --lr 1e-4 --normalize log_standard
```

Regularized CLSTM run for the current over-prediction pattern:

```powershell
python multi_head_experiments\train_multi_head_top5.py --features multichannel --model clstm --band 5 --epochs 50 --repeat 1 --batch-size 64 --lr 1e-4 --normalize log_standard --slot-pos-weight-scale 0.5 --consistency-occupancy 0.2 --active-count-regularizer 0.05
```

The regularized run reduces the positive-label pressure on the 54-label slot head and adds a small penalty when the predicted number of active users drifts above the true occupancy count.

To test a literal fixed slot `pos_weight=4.0`:

```powershell
python multi_head_experiments\train_multi_head_top5.py --features multichannel --model clstm --band 5 --epochs 50 --repeat 1 --batch-size 64 --lr 1e-4 --normalize log_standard --slot-pos-weight-fixed 4.0 --consistency-occupancy 0.2 --active-count-regularizer 0.05 --result-excel multi_head_experiments\outputs\multi_head_clstm_multichannel_band5_fixedpos4_analysis.xlsx --result-json multi_head_experiments\outputs\multi_head_clstm_multichannel_band5_fixedpos4_results.json
```

Generalization-focused CLSTM run:

```powershell
python multi_head_experiments\train_multi_head_top5.py --features multichannel --model clstm --band 5 --epochs 75 --repeat 1 --batch-size 64 --lr 7e-5 --weight-decay 5e-4 --normalize log_standard --slot-pos-weight-scale 0.5 --consistency-occupancy 0.2 --active-count-regularizer 0.05 --augment --time-mask-width 16 --freq-mask-width 8 --channel-drop-prob 0.05 --noise-std 0.02 --time-shift 12 --mixup-alpha 0.2 --save-dir multi_head_experiments\outputs\saved_models_clstm_augmix_slotpos0p5 --result-excel multi_head_experiments\outputs\multi_head_clstm_multichannel_band5_augmix_slotpos0p5_analysis.xlsx --result-json multi_head_experiments\outputs\multi_head_clstm_multichannel_band5_augmix_slotpos0p5_results.json
```

Per-class threshold calibration version:

```powershell
python multi_head_experiments\train_multi_head_top5.py --features multichannel --model clstm --band 5 --epochs 75 --repeat 5 --batch-size 64 --lr 7e-5 --weight-decay 5e-4 --normalize log_standard --slot-pos-weight-scale 0.5 --consistency-occupancy 0.2 --active-count-regularizer 0.05 --augment --time-mask-width 16 --freq-mask-width 8 --channel-drop-prob 0.05 --noise-std 0.02 --time-shift 12 --mixup-alpha 0.2 --threshold-strategy per_class --tune-threshold combined_score --threshold-min 0.10 --threshold-max 0.90 --threshold-step 0.05 --save-dir multi_head_experiments\outputs\saved_models_clstm_augmix_slotpos0p5_perclass_threshold --result-excel multi_head_experiments\outputs\multi_head_clstm_multichannel_band5_augmix_slotpos0p5_perclass_threshold_analysis.xlsx --result-json multi_head_experiments\outputs\multi_head_clstm_multichannel_band5_augmix_slotpos0p5_perclass_threshold_results.json
```

Quick smoke test:

```powershell
python multi_head_experiments\train_multi_head_top5.py --features multichannel --model resnet18 --band 5 --epochs 2 --repeat 1 --batch-size 64 --lr 1e-4 --normalize log_standard
```

Outputs are written under:

```text
D:\Deepak\wifi_csi\multi_head_experiments\outputs
```

The most important Excel columns are:

```text
direct_activity_set_micro_f1_avg
direct_occupancy_micro_f1_avg
slot_activity_set_micro_f1_avg
active_slot_micro_f1_avg
micro54_avg
slot_pred_active_slots_per_sample_avg
true_active_slots_per_sample_avg
```

Use `slot_pred_active_slots_per_sample_avg` versus `true_active_slots_per_sample_avg` to check whether the 54-label head is still over-predicting active users.

Excel sheets:

```text
summary
repeats
epoch_history
per_user_slot
per_activity
config
```
