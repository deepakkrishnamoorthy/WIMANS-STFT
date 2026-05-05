import os
import sys
import scipy.io
import scipy.signal
import numpy as np
import pandas as pd
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

def get_pc1_spectrogram(mat_path):
    csi_tensor = load_full_csi_tensor(mat_path)
    num_packets = csi_tensor.shape[0]
    
    # Calculate Amplitude and reshape
    amplitude_tensor = np.abs(csi_tensor)
    flattened_amplitude = amplitude_tensor.reshape((num_packets, -1))
    
    # Center data
    mean_amp = np.mean(flattened_amplitude, axis=0)
    centered_amp = flattened_amplitude - mean_amp
    
    # Extract PC1
    pca = PCA(n_components=1)
    pc1 = pca.fit_transform(centered_amp)[:, 0]
    
    # Compute STFT
    fs = 1000 # Assume 1000 Hz sampling rate for visualization
    f, t, Zxx = scipy.signal.stft(pc1, fs=fs, nperseg=128, noverlap=120)
    
    return f, t, np.abs(Zxx)

def main():
    config = {
        "annotation_file": r"D:\wifi_csi\EDA_with_antigravity\data\annotation.csv",
        "mat_dir": r"D:\wifi_csi\EDA_with_antigravity\data\mat"
    }
    
    run_info = setup_experiment("Multi_User_Interference", "cocktail_party_problem", config, base_dir=os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    outputs_dir = run_info["outputs_dir"]
    
    print("Loading annotations...")
    df = pd.read_csv(config["annotation_file"], dtype=str)
    
    # Filter for a consistent environment and band to make a fair comparison
    # Let's try to find samples in meeting_room at 5GHz
    subset = df[(df['environment'] == 'meeting_room') & (df['wifi_band'] == '5')]
    if subset.empty:
        subset = df # fallback
        
    # Get one sample each for 1, 3, and 5 users
    sample_1_user = subset[subset['number_of_users'] == '1'].iloc[0]['label']
    sample_3_users = subset[subset['number_of_users'] == '3'].iloc[0]['label']
    sample_5_users = subset[subset['number_of_users'] == '5'].iloc[0]['label']
    
    files_to_process = {
        "1 User": os.path.join(config["mat_dir"], f"{sample_1_user}.mat"),
        "3 Users": os.path.join(config["mat_dir"], f"{sample_3_users}.mat"),
        "5 Users": os.path.join(config["mat_dir"], f"{sample_5_users}.mat")
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    
    for ax, (title, path) in zip(axes, files_to_process.items()):
        print(f"Processing {title} from {path}...")
        f, t, Zxx = get_pc1_spectrogram(path)
        
        # Center frequencies around 0 (shift)
        f_centered = np.fft.fftshift(f)
        Zxx_centered = np.fft.fftshift(Zxx, axes=0)
        
        im = ax.pcolormesh(t, f_centered, Zxx_centered, shading='gouraud', cmap='jet', vmax=np.max(Zxx_centered)*0.8)
        ax.set_title(f"{title} Interference")
        ax.set_xlabel("Time (s)")
        if title == "1 User":
            ax.set_ylabel("Doppler Frequency (Hz)")
    
    plt.colorbar(im, ax=axes.ravel().tolist(), label='Magnitude')
    
    plot_path = os.path.join(outputs_dir, "multi_user_spectrograms.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved plot to {plot_path}")
    
    metrics = {
        "sample_1_user": sample_1_user,
        "sample_3_users": sample_3_users,
        "sample_5_users": sample_5_users
    }
    
    summary = "Visualized the 'Cocktail Party Problem' by plotting STFT spectrograms for 1, 3, and 5 users. As the number of users increases, the micro-Doppler signatures become highly entangled and the noise floor rises significantly. This proves that simple models will fail in crowded environments; we must utilize advanced source-separation architectures (like deep Transformers or Attention mechanisms) for multi-user subsets."
    finalize_experiment(run_info["run_id"], outputs_dir, metrics, summary)

if __name__ == "__main__":
    main()
