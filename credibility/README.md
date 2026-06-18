# Credibility Experiments

This folder contains comparison scripts that make the WiMANS baseline and our
STFT multi-head model speak the same metric language.

## A. Evaluate Our Model With WiMANS-Style Metrics

`evaluate_ours_wimans_metrics.py` loads saved multi-head STFT checkpoints and
evaluates them on the same hold-out split used in our runs.

It reports:

- `wimans_slot_activity_accuracy`: WiMANS-style exact-match accuracy after
  reshaping each sample into six user-slot rows of nine activity bits.
- `wimans_active_slot_activity_accuracy`: the same exact-match accuracy only
  on truly occupied user slots.
- `wimans_empty_slot_accuracy`: exact-match accuracy on empty user slots.
- `sample_exact_54_accuracy`: strict full-sample exact match across all 54
  user-slot labels.
- our usual decomposed F1 metrics, including activity-set, occupancy, active
  slot, and 54-label micro-F1.

The default uses no post-hoc occupancy gate, so it measures the slot-activity
head directly. Use `--occupancy-gate binary` or `--occupancy-gate prob` only
when you want to evaluate the post-processed variant.

## B. Evaluate WiMANS Models With Our Metrics

`train_wimans_with_ours_metrics.py` trains WiMANS-style CLSTM or THAT on raw
amplitude `.npy` files and reports both the original WiMANS slot-activity
accuracy and our decomposed diagnostics.

The default input is `dataset/amp_top5`, matching the Top-5 raw-amplitude
diagnostic setting. Use `--features amp_og` for the full 30-subcarrier
amplitude representation.

The default checkpoint selection mode is `--selection wimans_test`, which
mimics the original WiMANS code: the test split is evaluated during training
and the best test-accuracy epoch is selected. Use `--selection val` for our
clean train/validation/test protocol.
