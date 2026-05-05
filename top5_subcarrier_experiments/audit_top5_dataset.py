import argparse
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_ANNOTATION = r"D:\Deepak\wifi_csi\WiMANS-main\dataset\annotation.csv"
TOP5_PCA_DIR = r"D:\Deepak\wifi_csi\dataset\stft_top5_npy"
TOP5_MULTI_DIR = r"D:\Deepak\wifi_csi\dataset\stft_top5_multichannel_npy"
AMP_TOP5_DIR = r"D:\Deepak\wifi_csi\dataset\amp_top5"
AMP_FULL_DIR = r"D:\Deepak\wifi_csi\dataset\amp_OG"


def collect_files(folder):
    return {Path(path).stem: path for path in glob.glob(os.path.join(folder, "*.npy"))}


def shape_counts(files):
    counts = {}
    for path in files.values():
        shape = tuple(np.load(path, mmap_mode="r").shape)
        counts[shape] = counts.get(shape, 0) + 1
    return sorted(counts.items(), key=lambda item: item[1], reverse=True)


def sample_stats(files, max_samples=300):
    paths = list(files.values())
    if not paths:
        return {}
    step = max(1, len(paths) // max_samples)
    means, stds, maxs = [], [], []
    for path in paths[::step]:
        arr = np.load(path)
        means.append(float(arr.mean()))
        stds.append(float(arr.std()))
        maxs.append(float(arr.max()))
    return {
        "sampled": len(means),
        "mean_median": float(np.median(means)),
        "std_median": float(np.median(stds)),
        "max_median": float(np.median(maxs)),
        "max_p95": float(np.percentile(maxs, 95)),
    }


def folder_audit(name, folder, annotation_labels):
    files = collect_files(folder)
    stems = set(files)
    stats = sample_stats(files)
    shapes = shape_counts(files)
    summary = {
        "feature_set": name,
        "folder": folder,
        "files": len(files),
        "missing_annotation_labels": len(annotation_labels - stems),
        "extra_files_not_in_annotation": len(stems - annotation_labels),
        **stats,
    }
    shape_rows = [
        {"feature_set": name, "shape": str(shape), "count": count}
        for shape, count in shapes
    ]
    return summary, shape_rows


def print_folder_audit(summary, shape_rows):
    name = summary["feature_set"]
    print(f"\n{name}")
    print("-" * len(name))
    print(f"folder: {summary['folder']}")
    print(f"files: {summary['files']}")
    print(f"missing annotation labels: {summary['missing_annotation_labels']}")
    print(f"extra files not in annotation: {summary['extra_files_not_in_annotation']}")
    print(f"top shape counts: {[(row['shape'], row['count']) for row in shape_rows[:8]]}")
    print(f"sample stats: {summary}")


def label_audit(annotation):
    df = pd.read_csv(annotation, dtype=str)
    print("Annotation")
    print("----------")
    print(f"rows: {len(df)}")
    print(f"environments: {df['environment'].value_counts().to_dict()}")
    print(f"wifi bands: {df['wifi_band'].value_counts().to_dict()}")
    print(f"num users: {df['number_of_users'].value_counts().sort_index().to_dict()}")
    return df, set(df["label"])


def compute_reduction():
    top5 = collect_files(AMP_TOP5_DIR)
    full = collect_files(AMP_FULL_DIR)
    common = sorted(set(top5) & set(full))
    if not common:
        return None
    sample_key = common[0]
    top5_shape = np.load(top5[sample_key], mmap_mode="r").shape
    full_shape = np.load(full[sample_key], mmap_mode="r").shape
    top5_streams = int(np.prod(top5_shape[1:]))
    full_streams = int(np.prod(full_shape[1:]))
    print("\nCompute/feature reduction")
    print("-------------------------")
    print(f"sample: {sample_key}")
    print(f"Top-5 amplitude shape: {top5_shape}, streams={top5_streams}")
    print(f"Full amplitude shape: {full_shape}, streams={full_streams}")
    print(f"stream reduction: {full_streams} -> {top5_streams} ({top5_streams / full_streams * 100:.2f}% retained)")
    return {
        "sample": sample_key,
        "top5_shape": str(top5_shape),
        "top5_streams": top5_streams,
        "full_shape": str(full_shape),
        "full_streams": full_streams,
        "retained_percent": top5_streams / full_streams * 100.0,
    }


def write_excel(path, annotation_summary, folder_summaries, shape_rows, reduction):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame([annotation_summary]).to_excel(writer, sheet_name="annotation", index=False)
        pd.DataFrame(folder_summaries).to_excel(writer, sheet_name="feature_summary", index=False)
        pd.DataFrame(shape_rows).to_excel(writer, sheet_name="shape_counts", index=False)
        if reduction is not None:
            pd.DataFrame([reduction]).to_excel(writer, sheet_name="compute_reduction", index=False)
    print(f"\nSaved audit Excel to {path}")


def main():
    parser = argparse.ArgumentParser(description="Audit Top-5 STFT feature folders against WiMANS annotations.")
    parser.add_argument("--annotation", default=DEFAULT_ANNOTATION)
    parser.add_argument("--top5-pca-dir", default=TOP5_PCA_DIR)
    parser.add_argument("--top5-multi-dir", default=TOP5_MULTI_DIR)
    parser.add_argument("--excel", default=r"D:\Deepak\wifi_csi\top5_subcarrier_experiments\outputs\top5_dataset_audit.xlsx")
    args = parser.parse_args()

    df, labels = label_audit(args.annotation)
    annotation_summary = {
        "rows": len(df),
        "classroom_rows": int((df["environment"] == "classroom").sum()),
        "meeting_room_rows": int((df["environment"] == "meeting_room").sum()),
        "empty_room_rows": int((df["environment"] == "empty_room").sum()),
        "band_2_4_rows": int((df["wifi_band"] == "2.4").sum()),
        "band_5_rows": int((df["wifi_band"] == "5").sum()),
    }
    summaries = []
    all_shape_rows = []
    for name, folder in [
        ("Top-5 PCA-STFT", args.top5_pca_dir),
        ("Top-5 Multichannel STFT", args.top5_multi_dir),
    ]:
        summary, rows = folder_audit(name, folder, labels)
        summaries.append(summary)
        all_shape_rows.extend(rows)
        print_folder_audit(summary, rows)
    reduction = compute_reduction()
    write_excel(args.excel, annotation_summary, summaries, all_shape_rows, reduction)


if __name__ == "__main__":
    main()
