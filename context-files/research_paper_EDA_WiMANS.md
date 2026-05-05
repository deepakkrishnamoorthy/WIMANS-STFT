# WiMANS Dataset Exploratory Data Analysis & Multi-User Activity Detection

**Abstract**—*This living document serves as the formal research paper capturing the insights and methodologies derived from the independent exploratory analysis of the WiMANS dataset. As experiments are completed, the corresponding methodology and results will be appended here.*

## I. Introduction
The WiMANS dataset provides a unique opportunity to study human activity detection using WiFi Channel State Information (CSI) in complex, multi-user environments. This paper details a rigorous Exploratory Data Analysis (EDA) followed by the development of machine learning baselines to untangle the complexities of superimposed human-induced Doppler shifts.

## II. Related Work
(To be updated as relevant literature is referenced)

## III. Exploratory Data Analysis (Phase 1)
*This section will document the findings from our initial data understanding experiments.*

### A. Experiment 1.1: Phase Sanitization
**Objective:** To analyze the raw phase components of the complex CSI matrix and mitigate hardware-induced timing offsets.
**Methodology:** Raw WiFi phase is corrupted by Carrier Frequency Offset (CFO) and Sampling Frequency Offset (SFO), producing $2\pi$ wraps and linear phase drifts. We executed `exp_1_1_phase_sanitization.py` on the WiMANS baseline data (Intel 5300 3x3 MIMO, 30 subcarriers) to unwrap the phase and apply a linear fit to remove the subcarrier-dependent slope and intercept.
**Results:** The raw phase exhibited random structural wrapping. Post-sanitization, the phase across subcarriers was stabilized to near-zero linear trends, preserving only the relative multi-path fluctuations caused by environmental scattering and human body reflections. 
**Implication:** Phase sanitization is a mandatory preprocessing step before incorporating phase features into any machine learning pipeline for the WiMANS dataset.

**How to Interpret the Figure (Layman's Terms):** 
Imagine you are measuring a repeating wave, but your ruler is slightly broken, causing the measurement to randomly reset to zero and slant upwards over time (the chaotic "Raw Phase" graph on the left). Phase sanitization fixes the ruler. The "Sanitized Phase" graph on the right shows the true, cleaned-up wave where all the artificial resets and slants have been perfectly flattened out.

![Phase Sanitization Comparison](./outputs/20260420_1754_Phase_Sanitization_baseline/phase_comparison.png)

### B. Experiment 1.2: Amplitude Heatmaps & Signal-to-Noise Ratio (SNR) Analysis
**Objective:** To visualize the multi-path fading fluctuations caused by human movement over time and evaluate subcarrier variance.
**Methodology:** We extracted the full complex CSI tensor from the baseline data and computed the absolute amplitude. We generated a spectrogram-like heatmap over time (packet index) and frequency (subcarrier index) for a selected Tx-Rx pair (Tx=0, Rx=0). Additionally, we computed the amplitude variance across time for each subcarrier to quantify motion sensitivity.
**Results:** The amplitude heatmap reveals distinct wave-like disturbances correlated with human movement. Furthermore, the variance plot demonstrates that frequency-selective fading causes certain subcarriers to exhibit high variance (highly sensitive to motion), while others remain relatively static.
**Implication:** Not all subcarriers contribute equally to activity detection. Models should utilize subcarrier attention mechanisms or selective feature dropping to emphasize high-variance subcarriers and ignore noisy/static ones.

**How to Interpret the Figures (Layman's Terms):** 
*   **Amplitude Heatmap:** Think of this as a thermal camera image, but instead of tracking heat, it tracks WiFi signal strength over time. The colorful ripples and bands you see are the literal "shadows" cast by a person moving through the room, disrupting the WiFi signal. 
*   **Amplitude Variance:** This line graph shows which specific WiFi frequencies (subcarriers) "felt" the movement the most. A high spike means that specific frequency caught a lot of motion, while a low dip means that frequency was essentially deaf to the movement.

![Amplitude Heatmap](./outputs/20260420_1759_Amplitude_Heatmap_baseline/amplitude_heatmap.png)
![Amplitude Variance](./outputs/20260420_1759_Amplitude_Heatmap_baseline/amplitude_variance.png)

### C. Experiment 1.3: Time-Series Properties & Basic Doppler Shift Estimation
**Objective:** To extract dynamic micro-Doppler signatures caused by human limbs and torsos during movement.
**Methodology:** The raw 90-dimensional amplitude sequence (3 Tx $\times$ 3 Rx $\times$ 10 subcarriers) inherently contains massive multi-path interference and redundant information. To distill the core motion signal, we applied Principal Component Analysis (PCA) to the centered amplitude tensor, extracting the first Principal Component (PC1). A Short-Time Fourier Transform (STFT) was then performed on PC1 to generate a micro-Doppler spectrogram.
**Results:** The PC1 time-domain signal drastically reduced the noise floor compared to individual subcarriers. The resulting STFT spectrogram clearly isolates the frequency shifts (Doppler signatures) over time, which are the fundamental biophysical markers of distinct human activities (e.g., walking vs. sitting).
**Implication:** Time-Frequency representation (Spectrogram) of PCA-reduced CSI provides a highly structured, high-SNR 2D image. This strongly motivates the use of Computer Vision models (like 2D CNNs or Vision Transformers) as a baseline for Phase 2.

**How to Interpret the Figures (Layman's Terms):** 
*   **PC1 Time Domain:** We took 90 chaotic, overlapping WiFi signals and mathematically squished them together to find the one underlying pattern (the "Principal Component"). This graph shows the pure, isolated signal of the human moving, stripped of background static.
*   **STFT Micro-Doppler Spectrogram:** This is the most important image. It shows the "speed" of the movement over time. The bright colorful blobs above and below the center zero-line indicate moments where a person's arms, legs, or torso were actively moving towards or away from the router. A machine learning model looks at these specific shapes to know exactly *what* action the person is doing.

![PC1 Time Domain](./outputs/20260420_1802_STFT_Doppler_pca_amplitude/pc1_time_domain.png)
![STFT Micro-Doppler Spectrogram](./outputs/20260420_1802_STFT_Doppler_pca_amplitude/pc1_stft_spectrogram.png)

### D. Experiment 1.4: Antenna Correlation & Spatial Diversity
**Objective:** To determine if the multiple receiver antennas in the MIMO setup capture unique spatial information or simply redundant copies of the same signal.
**Methodology:** The absolute amplitude was averaged across subcarriers to produce a single spatial trace per Tx-Rx link. We computed the Pearson correlation matrix between the 3 Receiver antennas for Transmit Antenna 0, and plotted the raw spatial traces over time.
**Results:** As expected, the macro-movement captured by the three antennas is highly correlated ($r > 0.85$), as they are physically co-located and observe the same primary human motion. However, analyzing the raw traces reveals distinct micro-fading differences (momentary peaks and nulls unique to each antenna).
**Implication:** The MIMO array provides valuable spatial diversity. While a simple model could average the antennas to reduce noise, an advanced model (like 3D CNNs or spatial-attention models) could exploit these micro-fading differences to better localize the subject.

**How to Interpret the Figures (Layman's Terms):** 
*   **Receiver Spatial Correlation Heatmap:** This grid shows how similar the signals from the 3 antennas are to each other. Dark red (close to 1.0) means they see almost the exact same thing.
*   **Amplitude Traces across Receiver Antennas:** If you look closely at the squiggly lines, they mostly go up and down together (the big waves), but sometimes the blue line spikes while the green line dips (the tiny ripples). These tiny differences are the "spatial diversity" that helps advanced AI pinpoint exactly where the movement is happening in the room.

![Receiver Spatial Correlation](./outputs/20260420_1810_Spatial_Correlation_mimo_diversity/rx_spatial_correlation.png)
![Amplitude Traces across Receiver Antennas](./outputs/20260420_1810_Spatial_Correlation_mimo_diversity/rx_amplitude_traces.png)

### E. Experiment 1.6: 2D Spatial-Spectral Correlation Heatmaps
**Objective:** To determine if treating the CSI tensor as a "2D Image" (Antennas $\times$ Subcarriers) is mathematically sound.
**Methodology:** The absolute amplitude of all 270 spatial-spectral streams (3 Tx $\times$ 3 Rx $\times$ 30 subcarriers) was extracted. We computed the $270 \times 270$ Pearson correlation matrix and plotted it to identify localized blocks of highly correlated features.
**Results:** The correlation matrix reveals strong block-diagonal structures. Adjacent subcarriers within the same physical antenna link share extremely high correlation ($r > 0.9$), proving the existence of localized "patches" of information.
**Implication:** This localized patch correlation validates the use of 2D Convolutional Neural Networks (CNNs). CNNs are specifically designed to exploit local structural dependencies, making them an ideal architecture for processing raw CSI tensors.

**How to Interpret the Figures (Layman's Terms):** 
*   **2D Correlation Matrix:** This massive grid compares every single WiFi stream to every other stream. The bright yellow blocks along the diagonal prove that streams sitting right next to each other (adjacent frequencies) are essentially identical. This means we can treat the WiFi data just like a photograph, where neighboring pixels usually look similar.

![2D Spatial-Spectral Correlation Matrix](./outputs/20260420_1818_2D_Correlation_spatial_spectral_cnn_proof/spatial_spectral_correlation.png)

### F. Experiment 1.7: Static PCA vs. Sliding Window (Dynamic) PCA
**Objective:** To evaluate if PCA can be used for real-time (online) micro-Doppler extraction.
**Methodology:** We compared a "Static" PCA (fitted on the entire trace simultaneously, which looks into the future) against a "Dynamic" PCA (fitted using a sliding window of 300 packets, simulating real-time inference).
**Results:** While Dynamic PCA roughly tracks the macro motion, it suffers from severe boundary discontinuities, sign flips, and scaling mismatches at the edges of the sliding windows. 
**Implication:** Simple sliding window PCA is insufficient for real-time ML pipelines due to boundary artifacts. Real-time inference will require more sophisticated, continuous filtering methods, such as Neural Autoencoders (e.g., 1D CNN Autoencoder) to project the data dynamically without discontinuities.

**How to Interpret the Figures (Layman's Terms):** 
*   **Static vs Dynamic PCA:** The blue line is perfect because it gets to look at the whole video at once. The red dashed line tries to do the same thing but only gets to see tiny 1-second chunks at a time. Every time the red line starts a new chunk (the vertical dotted lines), it "glitches" and jumps wildly. This proves we can't use this simple math for real-time smart home systems; we need a more advanced AI to handle the continuous stream smoothly.

![Static vs Dynamic PCA](./outputs/20260420_1818_Dynamic_PCA_realtime_inference_viability/static_vs_dynamic_pca.png)

### G. Experiment 1.5: Multi-User Interference Complexity
**Objective:** To visualize the degradation of micro-Doppler signatures when multiple users are simultaneously active (the Cocktail Party Problem).
**Methodology:** We extracted samples for 1 User, 3 Users, and 5 Users from the `meeting_room` environment. Using our PCA-STFT pipeline, we generated and compared their spectrograms side-by-side.
**Results:** As the number of users increases, the distinct, structured Doppler signatures of individual movements become heavily entangled. The noise floor rises significantly, and overlapping motions create complex interference patterns that obliterate the clean contours seen in the 1-User scenario.
**Implication:** Simple classification models (e.g., standard CNNs trained on single users) will catastrophic fail in crowded environments. Phase 2 models must incorporate advanced source-separation architectures (like deep Transformers, Attention mechanisms, or Deep Clustering) to isolate overlapping signals before classification.

**How to Interpret the Figures (Layman's Terms):** 
*   **Multi-User Interference:** Imagine trying to listen to your friend talking in a quiet room (1 User) versus a crowded restaurant (5 Users). The first graph shows clear, crisp movements. By the third graph, it's just a chaotic blur of overlapping noise. This proves that our AI will need special "noise-canceling" or "focusing" abilities to work when a whole family is in the room.

![Multi-User Interference Spectrograms](./outputs/20260421_0858_Multi_User_Interference_cocktail_party_problem/multi_user_spectrograms.png)

### H. Experiment 1.8: 2.4 GHz vs. 5 GHz SNR Comparison
**Objective:** To determine the relative advantages of the 2.4 GHz and 5 GHz WiFi bands for micro-Doppler extraction.
**Methodology:** We compared STFT spectrograms for identical actions (1 user, meeting room) captured on the 2.4 GHz band and the 5 GHz band.
**Results:** The 5 GHz band exhibited a significantly larger Doppler frequency spread for the same physical movement compared to the 2.4 GHz band. This occurs because Doppler shift is inversely proportional to wavelength; the shorter 5 GHz wavelength results in higher frequency shifts.
**Implication:** 5 GHz is vastly superior for resolving fine-grained, subtle motions (like breathing or typing). However, 2.4 GHz may still be necessary for through-wall sensing. Future models should ideally utilize multi-band fusion to leverage the strengths of both frequencies.

**How to Interpret the Figures (Layman's Terms):** 
*   **2.4 GHz vs 5 GHz:** Because 5 GHz WiFi waves are physically smaller (shorter wavelength), they "bounce off" human movements faster. In the right-hand graph (5 GHz), the colorful blobs stretch much higher and lower than in the left graph (2.4 GHz). This means 5 GHz acts like a magnifying glass for movement, making it easier for AI to spot tiny actions.

![2.4 GHz vs 5 GHz Spectrograms](./outputs/20260421_0859_Frequency_Comparison_2.4G_vs_5G/frequency_comparison_spectrograms.png)

## IV. Machine Learning Baselines (Phase 2)
(Pending execution)

## V. Advanced Feature Engineering (Phase 3)
(Pending execution)

## VI. Conclusion
(To be written upon project completion)
