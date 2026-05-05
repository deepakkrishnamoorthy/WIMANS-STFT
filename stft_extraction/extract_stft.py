import os
import numpy as np
import scipy.signal as signal
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import argparse

def process_single_file(args):
    filepath, out_dir_npy, out_dir_img = args
    filename = os.path.basename(filepath)
    
    # Check if already processed
    npy_out_path = os.path.join(out_dir_npy, filename)
    img_out_path = os.path.join(out_dir_img, filename.replace('.npy', '.png'))
    if os.path.exists(npy_out_path) and os.path.exists(img_out_path):
        return True
        
    try:
        # Load raw data
        data_csi = np.load(filepath) # Expected shape (T, Tx, Rx, Sub)
        
        # 1. Flatten spatial-spectral dimensions
        # Shape becomes (T, Tx*Rx*Sub)
        T = data_csi.shape[0]
        data_flat = data_csi.reshape(T, -1)
        
        # 2. PCA Extraction
        # Center the data first
        data_centered = data_flat - np.mean(data_flat, axis=0)
        
        pca = PCA(n_components=1)
        pc1 = pca.fit_transform(data_centered).flatten()
        
        # 3. STFT Generation
        # Assuming 1000Hz sampling rate (3000 packets / 3 seconds)
        # Using a window of ~0.25 seconds (256 samples) and high overlap for smooth time resolution
        fs = 1000
        nperseg = 256
        noverlap = 240
        f, t, Zxx = signal.stft(pc1, fs=fs, window='hann', nperseg=nperseg, noverlap=noverlap)
        
        # We take the absolute magnitude to get the spectrogram
        spectrogram = np.abs(Zxx)
        
        # 4. Save the NumPy array for ML Training
        np.save(npy_out_path, spectrogram)
        
        # 5. Save the PNG image for Visual Inspection
        # We turn off axes and labels to just save the pure heatmap
        plt.figure(figsize=(10, 4))
        plt.pcolormesh(t, f, spectrogram, shading='gouraud', cmap='viridis')
        plt.axis('off')
        plt.savefig(img_out_path, bbox_inches='tight', pad_inches=0, dpi=100)
        plt.close('all') # close to free memory
        
        return True
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Extract STFT Spectrograms from CSI data using PCA.")
    parser.add_argument('--in_dir', type=str, default=r'd:\Deepak\wifi_csi\dataset\amp_top5', help="Directory containing input .npy files")
    parser.add_argument('--out_dir_npy', type=str, default=r'd:\Deepak\wifi_csi\dataset\stft_top5_npy', help="Directory to save output .npy files")
    parser.add_argument('--out_dir_img', type=str, default=r'd:\Deepak\wifi_csi\dataset\stft_top5_img', help="Directory to save output .png files")
    parser.add_argument('--workers', type=int, default=os.cpu_count(), help="Number of concurrent workers")
    args = parser.parse_args()

    os.makedirs(args.out_dir_npy, exist_ok=True)
    os.makedirs(args.out_dir_img, exist_ok=True)

    # Get all .npy files
    files = [f for f in os.listdir(args.in_dir) if f.endswith('.npy')]
    filepaths = [os.path.join(args.in_dir, f) for f in files]
    
    # Prepare arguments for multiprocessing
    tasks = [(fp, args.out_dir_npy, args.out_dir_img) for fp in filepaths]
    
    print(f"[*] Starting STFT extraction on {len(files)} files...")
    print(f"[*] Input: {args.in_dir}")
    print(f"[*] NPY Output: {args.out_dir_npy}")
    print(f"[*] IMG Output: {args.out_dir_img}")
    print(f"[*] Using {args.workers} workers for multiprocessing.")
    
    success_count = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        # Use tqdm to show progress
        futures = [executor.submit(process_single_file, task) for task in tasks]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting STFT"):
            if future.result():
                success_count += 1
                
    print(f"[*] Extraction complete! Successfully processed {success_count}/{len(files)} files.")

if __name__ == '__main__':
    main()
