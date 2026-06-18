import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, hamming_loss, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
MULTI_HEAD_DIR = ROOT / "multi_head_experiments"
WIMANS_WIFI_DIR = ROOT / "WiMANS-main" / "benchmark" / "wifi_csi"
sys.path.insert(0, str(MULTI_HEAD_DIR))
sys.path.insert(0, str(WIMANS_WIFI_DIR))

from dataset_multi_head import MultiHeadSTFTDataset  # noqa: E402
from load_data import load_data_y  # noqa: E402
from model_multi_head import CLSTMMultiHead, ResNet18MultiHead, THATStyleMultiHead  # noqa: E402


ENVIRONMENTS = ["classroom", "meeting_room", "empty_room"]
ACTIVITY_NAMES = ["nothing", "walk", "rotation", "jump", "wave", "lie_down", "pick_up", "sit_down", "stand_up"]
FEATURE_DIRS = {
    "multichannel": ROOT / "dataset" / "stft_top5_multichannel_npy",
    "pca": ROOT / "dataset" / "stft_top5_npy",
}
MODEL_FACTORIES = {
    "resnet18": lambda input_channels: ResNet18MultiHead(input_channels=input_channels),
    "clstm": lambda input_channels: CLSTMMultiHead(input_channels=input_channels),
    "that_style": lambda input_channels: THATStyleMultiHead(input_channels=input_channels),
}
WEIGHT_RE = re.compile(
    r"^multi_head_(?P<model>resnet18|clstm|that_style)_(?P<features>multichannel|pca)_"
    r"(?P<environment>classroom|meeting_room|empty_room)_(?P<band>2\.4|5|both)_seed(?P<seed>\d+)\.pth$"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate our saved multi-head STFT checkpoints with WiMANS-style and decomposed metrics."
    )
    parser.add_argument("--weights-dir", required=True)
    parser.add_argument("--analysis-excel", default=None, help="Optional run Excel to recover tuned thresholds.")
    parser.add_argument("--annotation", default=str(ROOT / "WiMANS-main" / "dataset" / "annotation.csv"))
    parser.add_argument("--features", choices=list(FEATURE_DIRS), default="multichannel")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--normalize", choices=["log_standard", "log", "standard", "none"], default="log_standard")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--split-seed", type=int, default=39)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--environment", choices=ENVIRONMENTS, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--band", choices=["2.4", "5", "both"], default=None)
    parser.add_argument("--occupancy-gate", choices=["none", "binary", "prob"], default="none")
    parser.add_argument("--occupancy-gate-power", type=float, default=1.0)
    parser.add_argument("--out-dir", default=str(ROOT / "credibility" / "outputs"))
    parser.add_argument("--tag", default="ours_wimans_style")
    return parser.parse_args()


def discover_weights(weights_dir, environment=None, seed=None, band=None):
    rows = []
    for name in sorted(os.listdir(weights_dir)):
        match = WEIGHT_RE.match(name)
        if not match:
            continue
        row = match.groupdict()
        row["seed"] = int(row["seed"])
        row["path"] = str(Path(weights_dir) / name)
        if environment is not None and row["environment"] != environment:
            continue
        if seed is not None and row["seed"] != seed:
            continue
        if band is not None and row["band"] != band:
            continue
        rows.append(row)
    if not rows:
        raise FileNotFoundError(f"No matching multi-head weights found in {weights_dir}")
    return rows


def read_thresholds(path):
    thresholds = {}
    if not path or not os.path.exists(path):
        return thresholds
    repeats = pd.read_excel(path, sheet_name="repeats")
    if not {"environment", "seed", "threshold"}.issubset(repeats.columns):
        return thresholds
    for _, row in repeats.iterrows():
        thresholds[(str(row["environment"]), int(row["seed"]))] = float(row["threshold"])
    return thresholds


def infer_input_channels(data_dir):
    for filename in os.listdir(data_dir):
        if filename.endswith(".npy"):
            sample = np.load(Path(data_dir) / filename, mmap_mode="r")
            return 1 if sample.ndim == 2 else sample.shape[0]
    raise FileNotFoundError(f"No .npy files found in {data_dir}")


def split_data(data_pd_y, split_seed, test_size, val_size):
    train_val_y, test_y = train_test_split(
        data_pd_y,
        test_size=test_size,
        shuffle=True,
        random_state=split_seed,
    )
    train_test_split(
        train_val_y,
        test_size=val_size,
        shuffle=True,
        random_state=split_seed,
    )
    return test_y


def collect_predictions(model, dataset, batch_size, num_workers, device):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    labels = {"activity_set": [], "occupancy": [], "slot_activity": []}
    probs = {"activity_set": [], "occupancy": [], "slot_activity": []}
    model.eval()
    with torch.no_grad():
        for batch in loader:
            outputs = model(batch["x"].to(device))
            for key in labels:
                labels[key].append(batch[key].cpu().numpy())
                probs[key].append(torch.sigmoid(outputs[key]).cpu().numpy())
    return {k: np.vstack(v) for k, v in labels.items()}, {k: np.vstack(v) for k, v in probs.items()}


def apply_prob_gate(slot_probs, occupancy_probs, gate_power):
    gate = np.power(np.clip(occupancy_probs.reshape(-1, 6, 1), 0.0, 1.0), gate_power)
    return (slot_probs.reshape(-1, 6, 9) * gate).reshape(-1, 54)


def apply_binary_gate(slot_pred, occupancy_pred):
    gated = slot_pred.reshape(-1, 6, 9).copy()
    gated[occupancy_pred.reshape(-1, 6) == 0, :] = 0
    return gated.reshape(-1, 54)


def compute_metrics(labels, probs, threshold, occupancy_gate, occupancy_gate_power):
    activity_pred = (probs["activity_set"] > threshold).astype(int)
    occupancy_pred = (probs["occupancy"] > threshold).astype(int)

    slot_probs = probs["slot_activity"]
    if occupancy_gate == "prob":
        slot_probs = apply_prob_gate(slot_probs, probs["occupancy"], occupancy_gate_power)
    slot_pred = (slot_probs > threshold).astype(int)
    if occupancy_gate == "binary":
        slot_pred = apply_binary_gate(slot_pred, occupancy_pred)

    slot_true = labels["slot_activity"].astype(int)
    true_activity_set = labels["activity_set"].astype(int)
    true_occupancy = labels["occupancy"].astype(int)

    true_user_rows = slot_true.reshape(-1, 6, 9).reshape(-1, 9)
    pred_user_rows = slot_pred.reshape(-1, 6, 9).reshape(-1, 9)
    active_mask = true_occupancy.reshape(-1) == 1
    empty_mask = ~active_mask

    slot_activity_set_pred = (slot_pred.reshape(-1, 6, 9).sum(axis=1) > 0).astype(int)
    slot_occupancy_pred = (slot_pred.reshape(-1, 6, 9).sum(axis=2) > 0).astype(int)

    metrics = {
        "threshold": threshold,
        "occupancy_gate": occupancy_gate,
        "wimans_slot_activity_accuracy": accuracy_score(true_user_rows, pred_user_rows) * 100.0,
        "wimans_active_slot_activity_accuracy": accuracy_score(true_user_rows[active_mask], pred_user_rows[active_mask]) * 100.0,
        "wimans_empty_slot_accuracy": accuracy_score(true_user_rows[empty_mask], pred_user_rows[empty_mask]) * 100.0,
        "sample_exact_54_accuracy": accuracy_score(slot_true, slot_pred) * 100.0,
        "sample_activity_set_exact_accuracy": accuracy_score(true_activity_set, activity_pred) * 100.0,
        "slot_collapsed_activity_set_exact_accuracy": accuracy_score(true_activity_set, slot_activity_set_pred) * 100.0,
        "occupancy_exact_accuracy": accuracy_score(true_occupancy, occupancy_pred) * 100.0,
        "slot_collapsed_occupancy_exact_accuracy": accuracy_score(true_occupancy, slot_occupancy_pred) * 100.0,
        "activity_set_micro_f1": f1_score(true_activity_set, activity_pred, average="micro", zero_division=0) * 100.0,
        "occupancy_micro_f1": f1_score(true_occupancy, occupancy_pred, average="micro", zero_division=0) * 100.0,
        "slot_activity_set_micro_f1": f1_score(true_activity_set, slot_activity_set_pred, average="micro", zero_division=0) * 100.0,
        "slot_occupancy_micro_f1": f1_score(true_occupancy, slot_occupancy_pred, average="micro", zero_division=0) * 100.0,
        "slot54_micro_f1": f1_score(slot_true, slot_pred, average="micro", zero_division=0) * 100.0,
        "slot54_macro_f1": f1_score(slot_true, slot_pred, average="macro", zero_division=0) * 100.0,
        "active_slot_micro_f1": f1_score(true_user_rows[active_mask], pred_user_rows[active_mask], average="micro", zero_division=0) * 100.0,
        "hamming54": hamming_loss(slot_true, slot_pred) * 100.0,
        "true_active_slots_per_sample": true_occupancy.sum(axis=1).mean(),
        "pred_active_slots_from_slot_head": slot_occupancy_pred.sum(axis=1).mean(),
        "pred_active_slots_from_occupancy_head": occupancy_pred.sum(axis=1).mean(),
    }
    return metrics, slot_pred, activity_pred, occupancy_pred


def per_activity_rows(labels, activity_pred, slot_pred, metadata):
    true_activity_set = labels["activity_set"].astype(int)
    slot_true = labels["slot_activity"].astype(int)
    slot_activity_set_pred = (slot_pred.reshape(-1, 6, 9).sum(axis=1) > 0).astype(int)
    p, r, f, support = precision_recall_fscore_support(
        true_activity_set, activity_pred, average=None, zero_division=0
    )
    sp, sr, sf, _ = precision_recall_fscore_support(
        true_activity_set, slot_activity_set_pred, average=None, zero_division=0
    )
    slot_p, slot_r, slot_f, slot_support = precision_recall_fscore_support(
        slot_true, slot_pred, average=None, zero_division=0
    )
    rows = []
    for idx, name in enumerate(ACTIVITY_NAMES):
        related = list(range(idx, 54, 9))
        row = {
            **metadata,
            "activity": name,
            "activity_head_precision": p[idx] * 100.0,
            "activity_head_recall": r[idx] * 100.0,
            "activity_head_f1": f[idx] * 100.0,
            "activity_head_support": int(support[idx]),
            "slot_collapsed_precision": sp[idx] * 100.0,
            "slot_collapsed_recall": sr[idx] * 100.0,
            "slot_collapsed_f1": sf[idx] * 100.0,
            "slot54_mean_f1_for_activity": float(np.mean(slot_f[related]) * 100.0),
            "slot54_total_support_for_activity": int(np.sum(slot_support[related])),
        }
        rows.append(row)
    return rows


def summarize(values):
    return {"avg": float(np.mean(values)), "std": float(np.std(values))}


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.data_dir) if args.data_dir else FEATURE_DIRS[args.features]
    thresholds = read_thresholds(args.analysis_excel)
    weights = discover_weights(args.weights_dir, args.environment, args.seed, args.band)
    input_channels = infer_input_channels(data_dir)
    device = torch.device(args.device)

    summary_rows = []
    per_activity = []
    for weight in weights:
        env = weight["environment"]
        band = weight["band"]
        seed = weight["seed"]
        model_name = weight["model"]
        threshold = thresholds.get((env, seed), args.threshold)

        data_pd_y = load_data_y(args.annotation, [env], [band])
        test_y = split_data(data_pd_y, args.split_seed, args.test_size, args.val_size)
        dataset = MultiHeadSTFTDataset(test_y, str(data_dir), max_len=200, normalize=args.normalize)

        model = MODEL_FACTORIES[model_name](input_channels).to(device)
        state = torch.load(weight["path"], map_location=device)
        model.load_state_dict(state)
        labels, probs = collect_predictions(model, dataset, args.batch_size, args.num_workers, device)
        metrics, slot_pred, activity_pred, occupancy_pred = compute_metrics(
            labels,
            probs,
            threshold,
            args.occupancy_gate,
            args.occupancy_gate_power,
        )

        metadata = {
            "model": model_name,
            "features": weight["features"],
            "band": band,
            "environment": env,
            "seed": seed,
            "threshold": threshold,
            "test_samples": len(dataset),
            "checkpoint": weight["path"],
        }
        row = {**metadata, **metrics}
        summary_rows.append(row)
        per_activity.extend(per_activity_rows(labels, activity_pred, slot_pred, metadata))

        print(
            f"[*] {model_name} {weight['features']} env={env} band={band} seed={seed} th={threshold:.3f} "
            f"| WiMANS-slot-acc={metrics['wimans_slot_activity_accuracy']:.2f}% "
            f"| active-slot-acc={metrics['wimans_active_slot_activity_accuracy']:.2f}% "
            f"| empty-slot-acc={metrics['wimans_empty_slot_accuracy']:.2f}% "
            f"| 54-F1={metrics['slot54_micro_f1']:.2f}% "
            f"| ActSet-F1={metrics['activity_set_micro_f1']:.2f}%"
        )

    summary_df = pd.DataFrame(summary_rows)
    per_activity_df = pd.DataFrame(per_activity)

    metric_cols = [
        "wimans_slot_activity_accuracy",
        "wimans_active_slot_activity_accuracy",
        "wimans_empty_slot_accuracy",
        "sample_exact_54_accuracy",
        "activity_set_micro_f1",
        "occupancy_micro_f1",
        "slot_activity_set_micro_f1",
        "active_slot_micro_f1",
        "slot54_micro_f1",
    ]
    grouped_rows = []
    for (model, features, band, env), group in summary_df.groupby(["model", "features", "band", "environment"]):
        row = {"model": model, "features": features, "band": band, "environment": env, "repeats": len(group)}
        for col in metric_cols:
            stats = summarize(group[col].to_numpy())
            row[f"{col}_avg"] = stats["avg"]
            row[f"{col}_std"] = stats["std"]
            row[f"{col}_text"] = f"{stats['avg']:.2f}+/-{stats['std']:.2f}"
        grouped_rows.append(row)
    grouped_df = pd.DataFrame(grouped_rows)

    base = out_dir / args.tag
    summary_path = base.with_suffix(".summary.csv")
    grouped_path = base.with_suffix(".grouped.csv")
    per_activity_path = base.with_suffix(".per_activity.csv")
    json_path = base.with_suffix(".json")
    excel_path = base.with_suffix(".xlsx")

    summary_df.to_csv(summary_path, index=False)
    grouped_df.to_csv(grouped_path, index=False)
    per_activity_df.to_csv(per_activity_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary_rows, "grouped": grouped_rows}, f, indent=2)
    with pd.ExcelWriter(excel_path) as writer:
        grouped_df.to_excel(writer, sheet_name="grouped", index=False)
        summary_df.to_excel(writer, sheet_name="repeats", index=False)
        per_activity_df.to_excel(writer, sheet_name="per_activity", index=False)

    print("\n================================================================================================================")
    print("Credibility A: Our STFT Multi-Head Model Evaluated With WiMANS-Style Metrics")
    print("================================================================================================================")
    display_cols = [
        "band",
        "environment",
        "wimans_slot_activity_accuracy_text",
        "wimans_active_slot_activity_accuracy_text",
        "wimans_empty_slot_accuracy_text",
        "activity_set_micro_f1_text",
        "slot54_micro_f1_text",
    ]
    print(grouped_df[display_cols].to_string(index=False))
    print("================================================================================================================")
    print(f"Saved summary CSV: {summary_path}")
    print(f"Saved grouped CSV: {grouped_path}")
    print(f"Saved per-activity CSV: {per_activity_path}")
    print(f"Saved Excel: {excel_path}")
    print(f"Saved JSON: {json_path}")


if __name__ == "__main__":
    main()
