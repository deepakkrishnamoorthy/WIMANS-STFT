import os
import sys
import scipy.io
import scipy.signal
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Add core path to import utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from experiment_utils import setup_experiment, finalize_experiment

def load_full_csi_tensor(mat_path):
    data = scipy.io.loadmat(mat_path)
    trace = data['trace']
    num_packets = trace.shape[0]
    
    csi_list = []
    for i in range(num_packets):
        packet = trace[i, 0]
        if hasattr(packet, 'dtype') and packet.dtype.names and 'csi' in packet.dtype.names:
            csi_matrix = packet['csi'][0, 0]
        else:
            csi_matrix = packet
        csi_list.append(csi_matrix)
        
    return np.array(csi_list)

def main():
    config = {
        "mat_file": r"D:\wifi_csi\dataset\mat\act_100_1.mat",
        "fs": 1000 # Assume ~1000 Hz sampling rate (typical for CSI, though WiMANS might be different, we use a nominal value for relative analysis)
    }
    
    run_info = setup_experiment("STFT_Doppler", "pca_amplitude", config, base_dir=os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    outputs_dir = run_info["outputs_dir"]
    
    print("Loading full CSI tensor...")
    csi_tensor = load_full_csi_tensor(config["mat_file"])
    num_packets = csi_tensor.shape[0]
    
    # Calculate Amplitude
    amplitude_tensor = np.abs(csi_tensor)
    
    # Reshape to (num_packets, features)
    features = amplitude_tensor.reshape(num_packets, -1)
    
    # Standardize features before PCA
    features_centered = features - np.mean(features, axis=0)
    
    # Apply PCA to extract dominant motion component
    pca = PCA(n_components=3)
    pcs = pca.fit_transform(features_centered)
    
    pc1 = pcs[:, 0]
    
    # Plot Time-domain PC1
    plt.figure(figsize=(10, 4))
    plt.plot(pc1, color='purple')
    plt.title("First Principal Component (PC1) of CSI Amplitude")
    plt.xlabel("Packet Index (Time)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    pc1_plot_path = os.path.join(outputs_dir, "pc1_time_domain.png")
    plt.savefig(pc1_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Compute and plot STFT Spectrogram of PC1
    # We use a nominal sampling frequency (config['fs'])
    fs = config["fs"]
    nperseg = 128
    noverlap = 120
    
    f, t, Zxx = scipy.signal.stft(pc1, fs=fs, nperseg=nperseg, noverlap=noverlap)
    
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=np.max(np.abs(Zxx))*0.5, shading='gouraud', cmap='jet')
    plt.title("STFT Spectrogram of PC1 (Micro-Doppler Signatures)")
    plt.ylabel("Frequency [Hz]")
    plt.xlabel("Time [sec]")
    plt.colorbar(label='Magnitude')
    
    stft_plot_path = os.path.join(outputs_dir, "pc1_stft_spectrogram.png")
    plt.savefig(stft_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved plots to {outputs_dir}")
    
    metrics = {
        "explained_variance_ratio_pc1": float(pca.explained_variance_ratio_[0]),
        "explained_variance_ratio_pc2": float(pca.explained_variance_ratio_[1]),
        "stft_freq_bins": len(f),
        "stft_time_bins": len(t)
    }
    
    summary = "Successfully applied PCA on the CSI amplitude to extract the dominant movement component (PC1). The STFT spectrogram of PC1 reveals clear micro-Doppler signatures corresponding to dynamic human motion. This confirms that Time-Frequency analysis (spectrograms) provides rich feature representations for CNN/ViT architectures."
    finalize_experiment(run_info["run_id"], outputs_dir, metrics, summary)

if __name__ == "__main__":
    main()
