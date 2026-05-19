# Curriculum Multi-Head Top-5 STFT Experiments

This folder contains a separate curriculum-learning branch for the Top-5 STFT multi-head setup.

It keeps the same train/validation/test split and the same final metrics as `multi_head_experiments/train_multi_head_top5.py`, but changes the training schedule:

1. Stage 1 learns easier global tasks on lower-density samples.
2. Stage 2 introduces harder slot supervision and more active-user density.
3. Stage 3 trains on the full training split using the normal multi-head objective.

The test set is evaluated once at the end, using the best validation checkpoint.

