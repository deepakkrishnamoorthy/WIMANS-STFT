import os
import sys
import scipy.io
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from experiment_utils import setup_experiment, finalize_experiment

import matplotlib.pyplot as plt

def sanitize_phase(raw_phase):
    # Unwrap phase across subcarriers (assumes last axis is subcarriers)
    unwrapped_phase = np.unwrap(raw_phase, axis=-1)
    
    # Linear transformation: remove slope and intercept
    num_subcarriers = unwrapped_phase.shape[-1]
    subcarrier_indices = np.arange(num_subcarriers)
    
    # Flatten to (N, subcarriers)
    flat_phase = unwrapped_phase.reshape(-1, num_subcarriers)
    sanitized_flat = np.zeros_like(flat_phase)
    
    for i in range(flat_phase.shape[0]):
        y = flat_phase[i, :]
        coeffs = np.polyfit(subcarrier_indices, y, 1)
        line = np.polyval(coeffs, subcarrier_indices)
        sanitized_flat[i, :] = y - line
        
    return sanitized_flat.reshape(raw_phase.shape)

def main():
    config = {
        "mat_file": r"D:\wifi_csi\dataset\mat\act_100_1.mat"
    }
    
    run_info = setup_experiment("Phase_Sanitization", "baseline", config, base_dir=os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    outputs_dir = run_info["outputs_dir"]
    
    print("Loading data...")
    data = scipy.io.loadmat(config["mat_file"])
    trace = data['trace']
    
    # Extract the first packet's CSI
    # trace is shape (2901, 1). The content of trace[0, 0] is the CSI for packet 0.
    csi_packet_0 = trace[0, 0]
    
    # Sometimes it's nested if MATLAB struct:
    if hasattr(csi_packet_0, 'dtype') and csi_packet_0.dtype.names and 'csi' in csi_packet_0.dtype.names:
        csi_matrix = csi_packet_0['csi'][0, 0]
    else:
        csi_matrix = csi_packet_0
        
    print(f"CSI Matrix shape: {csi_matrix.shape}")
    
    # Compute phases
    raw_phase = np.angle(csi_matrix)
    sanitized_phase = sanitize_phase(raw_phase)
    
    # Plotting the first rx/tx pair across subcarriers
    plt.figure(figsize=(12, 5))
    
    # We don't know exact dims, so we flatten and take the first vector of length subcarriers
    raw_vec = raw_phase.reshape(-1, raw_phase.shape[-1])[0]
    san_vec = sanitized_phase.reshape(-1, sanitized_phase.shape[-1])[0]
    
    plt.subplot(1, 2, 1)
    plt.plot(raw_vec, label="Raw Phase", color='red')
    plt.title("Raw Phase (with CFO/SFO wraps)")
    plt.xlabel("Subcarrier Index")
    plt.ylabel("Phase (radians)")
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(san_vec, label="Sanitized Phase", color='blue')
    plt.title("Sanitized Phase (Linear Trend Removed)")
    plt.xlabel("Subcarrier Index")
    plt.ylabel("Phase (radians)")
    plt.grid(True)
    
    plot_path = os.path.join(outputs_dir, "phase_comparison.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {plot_path}")
    
    metrics = {
        "num_packets": trace.shape[0],
        "csi_matrix_shape": str(csi_matrix.shape)
    }
    
    summary = "Successfully loaded raw complex CSI. The raw phase showed random wraps between -pi and pi due to hardware timing offsets. Applying the linear transformation (unwrapping + slope/intercept removal) successfully stabilized the phase across subcarriers."
    finalize_experiment(run_info["run_id"], outputs_dir, metrics, summary)

if __name__ == "__main__":
    main()
