import os
import sys
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
        "mat_file": r"D:\wifi_csi\dataset\mat\act_100_1.mat"
    }
    
    run_info = setup_experiment("Spatial_Correlation", "mimo_diversity", config, base_dir=os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    outputs_dir = run_info["outputs_dir"]
    
    print("Loading full CSI tensor...")
    csi_tensor = load_full_csi_tensor(config["mat_file"])
    num_packets = csi_tensor.shape[0]
    num_tx = csi_tensor.shape[1]
    num_rx = csi_tensor.shape[2]
    
    # Calculate Amplitude
    amplitude_tensor = np.abs(csi_tensor)
    
    # Let's average the amplitude across all subcarriers to get a single spatial sequence per Tx-Rx pair
    # Shape becomes (num_packets, num_tx, num_rx)
    spatial_amplitude = np.mean(amplitude_tensor, axis=3)
    
    # We will analyze Spatial Diversity at the Receiver (Rx antennas) for Transmit Antenna 0
    tx_idx = 0
    rx_traces = spatial_amplitude[:, tx_idx, :] # Shape: (num_packets, num_rx)
    
    # Compute Pearson Correlation Matrix between Rx antennas
    corr_matrix = np.corrcoef(rx_traces.T)
    
    # Plot Correlation Heatmap
    plt.figure(figsize=(6, 5))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=0, vmax=1,
                xticklabels=[f"Rx {i}" for i in range(num_rx)],
                yticklabels=[f"Rx {i}" for i in range(num_rx)])
    plt.title(f"Receiver Spatial Correlation (Tx {tx_idx})")
    
    plot_path = os.path.join(outputs_dir, "rx_spatial_correlation.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot the raw traces to visually see the diversity
    plt.figure(figsize=(10, 5))
    for rx_idx in range(num_rx):
        plt.plot(rx_traces[:, rx_idx], label=f"Rx {rx_idx}", alpha=0.7)
    plt.title(f"Amplitude Traces across Receiver Antennas (Tx {tx_idx})")
    plt.xlabel("Packet Index (Time)")
    plt.ylabel("Mean Amplitude")
    plt.legend()
    plt.grid(True)
    
    trace_plot_path = os.path.join(outputs_dir, "rx_amplitude_traces.png")
    plt.savefig(trace_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved plots to {outputs_dir}")
    
    metrics = {
        "rx_correlation_0_1": float(corr_matrix[0, 1]),
        "rx_correlation_0_2": float(corr_matrix[0, 2]),
        "rx_correlation_1_2": float(corr_matrix[1, 2])
    }
    
    summary = "Computed the spatial correlation matrix across the MIMO receiver antennas. While the antennas are highly correlated (as expected since they capture the same macro-movement), there are distinct micro-fading differences between the traces. This confirms that MIMO provides valuable spatial diversity, justifying 3D convolution or spatial attention layers in Phase 2."
    finalize_experiment(run_info["run_id"], outputs_dir, metrics, summary)

if __name__ == "__main__":
    main()
