import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
from sklearn.decomposition import PCA
from tqdm import tqdm


def normalize_trace(trace):
    trace = trace.astype(np.float32)
    return trace - np.mean(trace, axis=0, keepdims=True)


def compute_stft_bank(data_csi, mode, fs, nperseg, noverlap):
    time_len = data_csi.shape[0]
    data_flat = normalize_trace(data_csi.reshape(time_len, -1))

    if mode == "pca":
        pca = PCA(n_components=1)
        traces = pca.fit_transform(data_flat).T
    elif mode == "multichannel":
        traces = data_flat.T
    else:
        raise ValueError(f"Unknown mode: {mode}")

    spectrograms = []
    freqs = None
    times = None
    for trace in traces:
        freqs, times, zxx = signal.stft(
            trace,
            fs=fs,
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap,
        )
        spectrograms.append(np.abs(zxx).astype(np.float32))

    spec = np.stack(spectrograms, axis=0)
    if mode == "pca":
        spec = spec[0]
    return spec, freqs, times


def save_preview(spec, freqs, times, out_path, mode):
    if spec.ndim == 3:
        preview = np.mean(spec, axis=0)
    else:
        preview = spec

    plt.figure(figsize=(10, 4))
    vmax = np.percentile(preview, 99)
    plt.pcolormesh(times, freqs, preview, shading="gouraud", cmap="viridis", vmax=vmax)
    plt.axis("off")
    title = "Multi-channel mean STFT" if mode == "multichannel" else "PCA STFT"
    plt.title(title)
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0, dpi=100)
    plt.close("all")


def process_single_file(args):
    filepath, out_dir_npy, out_dir_img, mode, fs, nperseg, noverlap, overwrite = args
    filename = os.path.basename(filepath)
    npy_out_path = os.path.join(out_dir_npy, filename)
    img_out_path = os.path.join(out_dir_img, filename.replace(".npy", ".png"))

    if not overwrite and os.path.exists(npy_out_path) and os.path.exists(img_out_path):
        return True, None

    try:
        data_csi = np.load(filepath)
        spec, freqs, times = compute_stft_bank(data_csi, mode, fs, nperseg, noverlap)
        np.save(npy_out_path, spec.astype(np.float32))
        save_preview(spec, freqs, times, img_out_path, mode)
        return True, None
    except Exception as exc:
        return False, f"{filename}: {exc}"


def main():
    parser = argparse.ArgumentParser(description="Extract PCA or no-PCA multi-channel STFT features from Top-5 CSI.")
    parser.add_argument("--mode", choices=["multichannel", "pca"], default="multichannel")
    parser.add_argument("--in-dir", default=r"D:\Deepak\wifi_csi\dataset\amp_top5")
    parser.add_argument("--out-dir-npy", default=r"D:\Deepak\wifi_csi\dataset\stft_top5_multichannel_npy")
    parser.add_argument("--out-dir-img", default=r"D:\Deepak\wifi_csi\dataset\stft_top5_multichannel_img")
    parser.add_argument("--fs", type=int, default=1000)
    parser.add_argument("--nperseg", type=int, default=256)
    parser.add_argument("--noverlap", type=int, default=240)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.mode == "pca" and args.out_dir_npy.endswith("multichannel_npy"):
        args.out_dir_npy = r"D:\Deepak\wifi_csi\dataset\stft_top5_pca_rebuilt_npy"
    if args.mode == "pca" and args.out_dir_img.endswith("multichannel_img"):
        args.out_dir_img = r"D:\Deepak\wifi_csi\dataset\stft_top5_pca_rebuilt_img"

    os.makedirs(args.out_dir_npy, exist_ok=True)
    os.makedirs(args.out_dir_img, exist_ok=True)

    files = [f for f in os.listdir(args.in_dir) if f.endswith(".npy")]
    tasks = [
        (os.path.join(args.in_dir, f), args.out_dir_npy, args.out_dir_img,
         args.mode, args.fs, args.nperseg, args.noverlap, args.overwrite)
        for f in files
    ]

    print(f"[*] Extracting {args.mode} STFT for {len(tasks)} files")
    print(f"[*] Input: {args.in_dir}")
    print(f"[*] NPY output: {args.out_dir_npy}")
    print(f"[*] Image output: {args.out_dir_img}")

    success = 0
    errors = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process_single_file, task) for task in tasks]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting STFT"):
            ok, error = future.result()
            success += int(ok)
            if error:
                errors.append(error)

    print(f"[*] Extraction complete: {success}/{len(tasks)} files")
    if errors:
        print("[!] First errors:")
        for error in errors[:10]:
            print("   ", error)


if __name__ == "__main__":
    main()
