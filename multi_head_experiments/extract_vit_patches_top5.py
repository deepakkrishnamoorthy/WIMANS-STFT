import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np


DEFAULT_INPUT_DIR = r"D:\Deepak\wifi_csi\dataset\stft_top5_multichannel_npy"
DEFAULT_OUTPUT_DIR = r"D:\Deepak\wifi_csi\dataset\stft_top5_multichannel_vit_patches_npy"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Extract ViT-style patch tokens from Top-5 STFT .npy tensors. "
            "This is a preprocessing utility only; learned patch embeddings "
            "should be added inside a model later."
        )
    )
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--patch-height", type=int, default=16)
    parser.add_argument("--patch-width", type=int, default=16)
    parser.add_argument("--stride-height", type=int, default=None)
    parser.add_argument("--stride-width", type=int, default=None)
    parser.add_argument("--pad-mode", choices=["constant", "edge", "reflect"], default="constant")
    parser.add_argument("--pad-value", type=float, default=0.0)
    parser.add_argument("--layout", choices=["tokens", "grid"], default="tokens")
    parser.add_argument("--dtype", choices=["float32", "float16"], default="float32")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def as_chw(array):
    if array.ndim == 2:
        return array[np.newaxis, :, :]
    if array.ndim != 3:
        raise ValueError(f"Expected 2D or 3D array, got shape {array.shape}")
    return array


def compute_padding(length, patch, stride):
    if length <= patch:
        target = patch
    else:
        steps = int(np.ceil((length - patch) / stride)) + 1
        target = (steps - 1) * stride + patch
    return target - length, target


def pad_array(array, patch_h, patch_w, stride_h, stride_w, pad_mode, pad_value):
    _, height, width = array.shape
    pad_h, padded_h = compute_padding(height, patch_h, stride_h)
    pad_w, padded_w = compute_padding(width, patch_w, stride_w)
    pad_spec = ((0, 0), (0, pad_h), (0, pad_w))
    if pad_h == 0 and pad_w == 0:
        return array, (height, width)
    if pad_mode == "constant":
        padded = np.pad(array, pad_spec, mode=pad_mode, constant_values=pad_value)
    else:
        padded = np.pad(array, pad_spec, mode=pad_mode)
    return padded, (padded_h, padded_w)


def patchify(array, patch_h, patch_w, stride_h, stride_w, layout):
    channels, height, width = array.shape
    grid_h = ((height - patch_h) // stride_h) + 1
    grid_w = ((width - patch_w) // stride_w) + 1

    windows = np.lib.stride_tricks.sliding_window_view(
        array,
        window_shape=(patch_h, patch_w),
        axis=(1, 2),
    )
    windows = windows[:, ::stride_h, ::stride_w, :, :]
    windows = windows[:, :grid_h, :grid_w, :, :]
    patches = windows.transpose(1, 2, 0, 3, 4)

    if layout == "grid":
        return np.ascontiguousarray(patches), grid_h, grid_w

    tokens = patches.reshape(grid_h * grid_w, channels * patch_h * patch_w)
    return np.ascontiguousarray(tokens), grid_h, grid_w


def process_file(path, output_dir, args):
    stride_h = args.stride_height or args.patch_height
    stride_w = args.stride_width or args.patch_width
    array = as_chw(np.load(path)).astype(np.float32, copy=False)
    original_shape = tuple(int(x) for x in array.shape)

    padded, padded_hw = pad_array(
        array,
        args.patch_height,
        args.patch_width,
        stride_h,
        stride_w,
        args.pad_mode,
        args.pad_value,
    )
    patches, grid_h, grid_w = patchify(
        padded,
        args.patch_height,
        args.patch_width,
        stride_h,
        stride_w,
        args.layout,
    )
    patches = patches.astype(args.dtype, copy=False)

    out_path = output_dir / path.name
    if out_path.exists() and not args.overwrite:
        status = "skipped"
    else:
        status = "dry_run" if args.dry_run else "written"
        if not args.dry_run:
            np.save(out_path, patches)

    return {
        "input_file": path.name,
        "output_file": out_path.name,
        "status": status,
        "layout": args.layout,
        "original_shape": json.dumps(original_shape),
        "padded_shape": json.dumps((original_shape[0], int(padded_hw[0]), int(padded_hw[1]))),
        "patch_height": args.patch_height,
        "patch_width": args.patch_width,
        "stride_height": stride_h,
        "stride_width": stride_w,
        "grid_height": grid_h,
        "grid_width": grid_w,
        "num_patches": grid_h * grid_w,
        "patch_dim": original_shape[0] * args.patch_height * args.patch_width,
        "saved_shape": json.dumps(tuple(int(x) for x in patches.shape)),
        "dtype": str(patches.dtype),
    }


def write_manifest(rows, output_dir, dry_run):
    if not rows:
        return
    manifest_path = output_dir / ("manifest_dry_run.csv" if dry_run else "manifest.csv")
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved manifest: {manifest_path}")


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if args.patch_height <= 0 or args.patch_width <= 0:
        raise ValueError("Patch size must be positive.")
    if args.stride_height is not None and args.stride_height <= 0:
        raise ValueError("Stride height must be positive.")
    if args.stride_width is not None and args.stride_width <= 0:
        raise ValueError("Stride width must be positive.")

    files = sorted(input_dir.glob("*.npy"))
    if args.max_files is not None:
        files = files[: args.max_files]
    if not files:
        raise FileNotFoundError(f"No .npy files found in {input_dir}")

    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(
        f"Patch={args.patch_height}x{args.patch_width} "
        f"Stride={(args.stride_height or args.patch_height)}x{(args.stride_width or args.patch_width)} "
        f"Layout={args.layout}"
    )
    print(f"Files: {len(files)}")

    rows = []
    for index, path in enumerate(files, start=1):
        row = process_file(path, output_dir, args)
        rows.append(row)
        if index <= 3 or index == len(files):
            print(
                f"[{index}/{len(files)}] {path.name}: "
                f"{row['original_shape']} -> {row['saved_shape']} ({row['status']})"
            )

    if args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(rows, output_dir, args.dry_run)


if __name__ == "__main__":
    main()
