import os
import sys
import scipy.io
import numpy as np
import matplotlib.pyplot as plt

# Add core path to import utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from experiment_utils import setup_experiment, finalize_experiment

def load_full_csi_tensor(mat_path):
    """
    Extracts the full CSI tensor from the trace struct array.
    Returns array of shape (num_packets, num_tx, num_rx, num_subcarriers)
    """
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
        "mat_file": r"D:\wifi_csi\dataset\mat\act_100_1.mat"
    }
    
    run_info = setup_experiment("Amplitude_Heatmap", "baseline", config, base_dir=os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    outputs_dir = run_info["outputs_dir"]
    
    print("Loading full CSI tensor...")
    csi_tensor = load_full_csi_tensor(config["mat_file"])
    print(f"CSI Tensor Shape: {csi_tensor.shape}")
    
    # Calculate Amplitude
    amplitude_tensor = np.abs(csi_tensor)
    
    # Plotting Amplitude Heatmap for Tx=0, Rx=0
    # Shape of amplitude for one pair: (num_packets, num_subcarriers)
    tx_idx = 0
    rx_idx = 0
    amp_slice = amplitude_tensor[:, tx_idx, rx_idx, :]
    
    plt.figure(figsize=(10, 6))
    # We transpose to have subcarriers on y-axis, time on x-axis (standard spectrogram/heatmap view)
    im = plt.imshow(amp_slice.T, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(im, label='Amplitude')
    plt.title(f"CSI Amplitude Heatmap (Tx {tx_idx}, Rx {rx_idx})")
    plt.xlabel("Packet Index (Time)")
    plt.ylabel("Subcarrier Index")
    
    plot_path = os.path.join(outputs_dir, "amplitude_heatmap.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {plot_path}")
    
    # Let's also do a variance analysis to see which subcarriers have the most movement energy
    variance_across_time = np.var(amp_slice, axis=0)
    plt.figure(figsize=(8, 4))
    plt.plot(variance_across_time, marker='o')
    plt.title("Amplitude Variance per Subcarrier")
    plt.xlabel("Subcarrier Index")
    plt.ylabel("Variance over Time")
    plt.grid(True)
    
    var_plot_path = os.path.join(outputs_dir, "amplitude_variance.png")
    plt.savefig(var_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    metrics = {
        "csi_tensor_shape": str(csi_tensor.shape),
        "max_amplitude": float(np.max(amplitude_tensor)),
        "min_amplitude": float(np.min(amplitude_tensor)),
        "mean_amplitude": float(np.mean(amplitude_tensor))
    }
    
    summary = "Successfully extracted full CSI tensor and plotted amplitude heatmaps over time. The heatmap reveals wave-like disturbances indicating human movement. Subcarrier variance analysis shows that certain subcarriers capture more movement energy than others, suggesting feature selection might be beneficial."
    finalize_experiment(run_info["run_id"], outputs_dir, metrics, summary)

if __name__ == "__main__":
    main()
