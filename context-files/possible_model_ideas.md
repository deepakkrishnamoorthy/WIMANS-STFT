# Possible Model Ideas Tracker

As we complete EDA experiments, we will log how the observed data properties can inspire or restrict future machine learning architectures.

| Data Property Observed | Potential Model Architecture | Pros | Cons / Unsuitability Reasons |
| :--- | :--- | :--- | :--- |
| **Random Phase CFO/SFO wrapping** (Exp 1.1) | Complex-valued CNNs acting on *linearly sanitized* phase, or Amplitude-only models. | Complex models can leverage relative phase differences. | Unsanitized raw phase will destroy gradient descent. Absolute Time-of-Flight is lost after sanitization. |
| **Subcarrier Amplitude Variance Variability** (Exp 1.2) | Subcarrier Attention Mechanisms (e.g., SE-blocks or Transformers) on the CSI sequence. | Allows the network to dynamically ignore static/noisy subcarriers and focus on motion-sensitive ones. | Increases model parameter count; attention weights may overfit to specific environments. |
| **Clean Micro-Doppler in PC1 STFT** (Exp 1.3) | 2D CNNs (ResNet, EfficientNet) or Vision Transformers acting on the Spectrogram. | Leverages massive existing pre-trained CV architectures. Extracts rich temporal and frequency relationships simultaneously. | PCA discards the spatial (MIMO) information, reducing performance in extremely complex multi-user interference scenarios. |
| **Spatial Micro-Fading Diversity** (Exp 1.4) | 3D CNNs or Spatial-Attention blocks across the Antenna dimension. | Allows the network to triangulate and spatially isolate movements based on tiny phase/amplitude differences between antennas. | Drastically increases compute requirements compared to averaging antennas. |
| **High Adjacent Stream Correlation** (Exp 1.6) | 2D CNNs operating directly on the raw `(Time, Antennas * Subcarriers)` tensor. | Validates that local patches exist, allowing convolutional kernels to effectively extract spatial-spectral features without manual feature engineering. | None. This is a massive green light for CNNs. |
| **Dynamic Boundary Artifacts** (Exp 1.7) | Deep Neural Autoencoders (e.g. 1D CNN Autoencoder) instead of sliding-window PCA. | Projects high-dimensional data smoothly in real-time without the sign-flips and discontinuities of mathematical PCA. | Requires training a separate autoencoder model just for feature extraction. |
| **Multi-User Entanglement** (Exp 1.5) | Source Separation Architectures (e.g., U-Net, Deep Clustering) or Multi-Head Attention. | Capable of separating overlapping Doppler signatures into distinct channels before classification. | Extremely difficult to train; requires massive amounts of data. |
| **High Frequency Doppler Spread** (Exp 1.8) | Multi-band Fusion Networks (combining 2.4G and 5G inputs) or Multi-Scale Kernels. | Leverages 5GHz for fine-grained motion (larger shifts) and 2.4GHz for macro motion / wall penetration. | Requires synchronized dual-band capture hardware in production. |

---
*Note: This table will be updated automatically during the Post-flight documentation phase of every experiment run.*
