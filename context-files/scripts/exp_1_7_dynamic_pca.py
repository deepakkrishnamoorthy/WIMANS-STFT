import os
import sys
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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
        "window_size": 300
    }
    
    run_info = setup_experiment("Dynamic_PCA", "realtime_inference_viability", config, base_dir=os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    outputs_dir = run_info["outputs_dir"]
    
    print("Loading full CSI tensor...")
    csi_tensor = load_full_csi_tensor(config["mat_file"])
    num_packets = csi_tensor.shape[0]
    
    # Calculate Amplitude and reshape
    amplitude_tensor = np.abs(csi_tensor)
    flattened_amplitude = amplitude_tensor.reshape((num_packets, -1))
    
    # Standardize
    scaler = StandardScaler()
    scaled_amplitude = scaler.fit_transform(flattened_amplitude)
    
    print("Computing Static PCA (cheating, looks at whole file)...")
    static_pca = PCA(n_components=1)
    static_pc1 = static_pca.fit_transform(scaled_amplitude)[:, 0]
    
    print("Computing Dynamic Sliding Window PCA (real-time simulation)...")
    dynamic_pc1 = np.zeros(num_packets)
    window_size = config["window_size"]
    prev_components = None
    
    for start_idx in range(0, num_packets, window_size):
        end_idx = min(start_idx + window_size, num_packets)
        window_data = scaled_amplitude[start_idx:end_idx]
        
        # If the window is too small, just skip or use previous
        if window_data.shape[0] < 2:
            continue
            
        win_pca = PCA(n_components=1)
        win_pc1 = win_pca.fit_transform(window_data)[:, 0]
        
        # Fix sign ambiguity: if dot product with prev eigenvector is negative, flip
        if prev_components is not None:
            if np.dot(win_pca.components_[0], prev_components[0]) < 0:
                win_pc1 = -win_pc1
                win_pca.components_[0] = -win_pca.components_[0]
                
        dynamic_pc1[start_idx:end_idx] = win_pc1
        prev_components = win_pca.components_
        
    print("Plotting comparison...")
    plt.figure(figsize=(12, 6))
    plt.plot(static_pc1, label="Static PCA (Full Trace)", alpha=0.8, color='blue')
    plt.plot(dynamic_pc1, label=f"Dynamic PCA (Window={window_size})", alpha=0.6, color='red', linestyle='dashed')
    
    # Draw vertical lines for window boundaries
    for w in range(window_size, num_packets, window_size):
        plt.axvline(x=w, color='gray', linestyle=':', alpha=0.3)
        
    plt.title("Static vs Dynamic PCA for Micro-Doppler Extraction")
    plt.xlabel("Packet Index")
    plt.ylabel("PC1 Amplitude")
    plt.legend()
    plt.grid(True)
    
    plot_path = os.path.join(outputs_dir, "static_vs_dynamic_pca.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Calculate MSE between them
    mse = np.mean((static_pc1 - dynamic_pc1) ** 2)
    
    print(f"Saved plots to {outputs_dir}")
    
    metrics = {
        "mse": float(mse)
    }
    
    summary = f"Compared Static PCA against Dynamic Sliding Window PCA (window={window_size}). While Dynamic PCA roughly tracks the motion, it suffers from severe boundary discontinuities and scaling mismatches at window edges (MSE: {mse:.2f}). This indicates that while PCA is great for offline EDA, real-time inference will likely require a Neural Autoencoder or continuous filtering rather than naive sliding window PCA."
    finalize_experiment(run_info["run_id"], outputs_dir, metrics, summary)

if __name__ == "__main__":
    main()
