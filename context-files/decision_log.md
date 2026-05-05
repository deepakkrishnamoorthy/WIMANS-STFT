# Project Decision Log

This log tracks all major shifts in strategy, hypothesis setting, and outcomes to ensure the thought process behind the research remains traceable.

## Date: 2026-04-20
*   **Hypothesis / Decision**: Adopt a structured, reproducible experimental pipeline separating EDA (Phase 1) from ML baselines (Phase 2) and advanced architectures (Phase 3). Adopt "Big Lab" standards (Global seed locking, Git Hash tracking, Data immutability).
*   **Tradeoff Addressed**: Front-loading the boilerplate code (`experiment_utils.py`) trades slight initial setup time for massive long-term reliability and prevention of non-reproducible results.
*   **Outcome**: Pipeline successfully set up. All necessary scaffolding and trackers created.

## Date: 2026-04-20 (Experiment 1.1)
*   **Hypothesis / Decision**: Raw WiFi CSI phase is highly noisy due to Carrier Frequency Offset (CFO) and Sampling Frequency Offset (SFO), making it unusable out-of-the-box. We hypothesize that applying a standard linear transformation (phase unwrapping + slope/intercept removal) will sanitize the data for future ML models.
*   **Tradeoff Addressed**: Using simple linear transformations is fast and deterministic, though it may remove some absolute time-of-flight information (which is typically unreliable in commodity WiFi anyway).
*   **Outcome**: Successfully ran `exp_1_1_phase_sanitization.py`. The unwrapped and linearly transformed phase eliminates the arbitrary $\pi$ and $-\pi$ jumps, producing a stable phase trend across subcarriers. This confirms we MUST preprocess phase data before passing it to any NN.

## Date: 2026-04-20 (Experiment 1.2)
*   **Hypothesis / Decision**: CSI amplitude directly reflects the multi-path fading caused by human movement. Some subcarriers will exhibit higher variance (more sensitivity to motion) than others due to frequency-selective fading. We hypothesize that amplitude heatmaps and variance analysis will highlight these wave-like movement patterns.
*   **Tradeoff Addressed**: Analyzing absolute amplitude does not require unwrapping or complex preprocessing, making it highly robust, though it loses directional information typically carried by phase.
*   **Outcome**: Amplitude heatmaps successfully visualized human movement over the 2901 packets. Variance analysis proved that specific subcarriers are far more sensitive to movement than others, strongly supporting the idea of applying attention mechanisms or subcarrier-selection heuristics in our machine learning models.

## Date: 2026-04-20 (Experiment 1.3)
*   **Hypothesis / Decision**: We hypothesize that applying Principal Component Analysis (PCA) to the 90-dimensional amplitude vector (3 Tx * 3 Rx * 10 subcarriers or similar) per packet will extract a dominant feature (PC1) that isolates human movement from static environmental noise. We then hypothesize that computing the STFT of this PC1 will reveal distinct micro-Doppler signatures.
*   **Tradeoff Addressed**: Using PCA reduces the vast dimensions of raw CSI into a single trace, sacrificing spatial context (MIMO diversity) in exchange for an incredibly high-SNR, cleanly analyzable 1D signal.
*   **Outcome**: The PCA -> STFT pipeline successfully generated a spectrogram with distinct micro-Doppler signatures over time. This confirms that Time-Frequency analysis provides very rich, structured features that are ideal for computer vision-based architectures like 2D CNNs or Vision Transformers.

## Date: 2026-04-20 (Experiment 1.4)
*   **Hypothesis / Decision**: Evaluate whether multiple receiver antennas capture unique spatial information (spatial diversity) or merely redundant copies of the same signal.
*   **Tradeoff Addressed**: We visualize both the correlation matrix and the raw time-series traces to balance quantitative correlation metrics with qualitative observations of micro-fading.
*   **Outcome**: While antennas are highly correlated ($r > 0.85$), significant micro-fading differences exist at the sample level. This proves that spatial diversity is present and useful for advanced localization models.

## Date: 2026-04-20 (Experiment 1.6)
*   **Hypothesis / Decision**: We hypothesize that adjacent spatial-spectral streams (antennas and subcarriers) are highly correlated, allowing the CSI tensor to be processed mathematically like a 2D image.
*   **Tradeoff Addressed**: Computing a massive 270x270 correlation matrix is computationally heavy for EDA but definitively proves the underlying structural assumptions required for Convolutional Neural Networks.
*   **Outcome**: The correlation matrix revealed strong block-diagonal structures ($r > 0.9$ for adjacent subcarriers), perfectly validating the use of 2D CNNs on the raw CSI tensor.

## Date: 2026-04-20 (Experiment 1.7)
*   **Hypothesis / Decision**: We hypothesize that sliding-window PCA can replicate the offline PCA micro-Doppler extraction for real-time inference applications.
*   **Tradeoff Addressed**: We compare offline (full-trace) PCA vs online (sliding-window) PCA to visualize boundary artifacts.
*   **Outcome**: Sliding window PCA suffers from severe boundary discontinuities and sign flips. Real-time inference will require deep Neural Autoencoders instead of naive sliding-window math.

## Date: 2026-04-21 (Experiment 1.5)
*   **Hypothesis / Decision**: We hypothesize that increasing the number of users will drastically raise the noise floor and entangle Doppler signatures (the Cocktail Party Problem).
*   **Tradeoff Addressed**: We plot 1, 3, and 5 users side-by-side using the PCA-STFT pipeline to qualitatively assess the degradation of signal clarity.
*   **Outcome**: Visualized severe interference in crowded environments. Proves that simple classification models will fail on multi-user data; source separation or deep attention mechanisms are required.

## Date: 2026-04-21 (Experiment 1.8)
*   **Hypothesis / Decision**: We hypothesize that the 5 GHz band provides finer micro-Doppler resolution than the 2.4 GHz band due to its smaller wavelength.
*   **Tradeoff Addressed**: We compare one 2.4 GHz and one 5 GHz sample of the same action/environment to visualize the Doppler spread differences.
*   **Outcome**: 5 GHz showed distinctly larger Doppler shifts for similar movements, making it better for subtle motion detection. Models should utilize multi-band fusion if possible.
