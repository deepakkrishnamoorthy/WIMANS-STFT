# Proposed Experiments Tracker

This document tracks all planned, ongoing, and completed tasks for the WiMANS dataset. We will begin with strict Exploratory Data Analysis (Phase 1) to understand the physics of the data before prioritizing and moving into Machine Learning (Phase 2 & 3).

## Phase 1: Exploratory Data Analysis (Data Understanding)

| ID | Experiment Name | Goal / Hypothesis | Status | Outcome / Notes |
| :--- | :--- | :--- | :--- | :--- |
| **1.1** | Phase Sanitization Visualization | Plot raw phase vs. a linearly transformed phase to visually confirm that the transformation effectively removes the random hardware noise (CFO) wraps. | 📝 Proposed | - |
| **1.2** | Micro-Doppler (STFT) Analysis | Generate STFT spectrograms for empty room vs. walking vs. waving to see if the frequency spread visually distinguishes the activities. | 📝 Proposed | - |
| **1.3** | Antenna Correlation & Spatial Diversity | Calculate and plot the correlation matrix between different receiver antennas to see how much unique spatial information the MIMO array captures. | 📝 Proposed | - |
| **1.4** | 2.4 GHz vs. 5 GHz SNR Comparison | Plot the amplitude variance and Signal-to-Noise Ratio (SNR) for both bands side-by-side to determine which band provides cleaner signals for subtle movements. | 📝 Proposed | - |
| **1.5** | Multi-User Interference Visualization | Plot the raw time-series amplitude of 1 user vs. 3 users vs. 5 users to visualize the severity of the interference and assess the cocktail party problem. | 📝 Proposed | - |
| **1.6** | 2D Spatial-Spectral Correlation Heatmaps | Compute the correlation matrix across all subcarriers and antennas to see if localized "patches" of correlated information exist, justifying 2D CNN usage. | 📝 Proposed | - |
| **1.7** | Static PCA vs. Sliding Window (Dynamic) PCA | Compare full-trace PCA with a sliding window PCA to assess if PCA is viable for real-time online inference without massive boundary artifacts. | 📝 Proposed | - |

---

## Phase 2: Machine Learning Baselines

| ID | Experiment Name | Goal / Hypothesis | Status | Outcome / Notes |
| :--- | :--- | :--- | :--- | :--- |
| **2.1** | Raw Phase vs. Sanitized Phase ML Baseline | Train a baseline CNN/LSTM. Applying a linear phase transformation (sanitization) will significantly improve classification accuracy compared to raw phase data. | 📝 Proposed | - |
| **2.2** | Micro-Doppler Spectrograms (STFT) + 2D CNN | Converting raw time-series CSI into STFT spectrograms and using a Vision model (ResNet/CNN) will outperform raw time-series models. | 📝 Proposed | - |
| **2.3** | 2.4 GHz vs. 5 GHz ML Performance | Train identical models on the two bands separately to reveal which frequency yields higher accuracy for multi-user HAR. | 📝 Proposed | - |

---

## Phase 3: Advanced Multi-User & Topological Features

| ID | Experiment Name | Goal / Hypothesis | Status | Outcome / Notes |
| :--- | :--- | :--- | :--- | :--- |
| **3.1** | Baseline Superposition Handling | Train a multi-label classification model (e.g., Transformer) on 1-user data and evaluate its degradation as the number of users increases to 5. | 📝 Proposed | - |
| **3.2** | Persistent Homology (TDA) Features | Extract Betti numbers from the CSI phase-space. These topological invariants will improve cross-room generalization compared to raw amplitude/phase. | 📝 Proposed | - |
| **3.3** | Source Separation Pre-processing | Implement a blind source separation algorithm to untangle the superimposed 5-user CSI signals before feeding them into the classifier. | 📝 Proposed | - |

---
*Status Key: 📝 Proposed | ⏳ In Progress | ✅ Completed | ❌ Abandoned*
