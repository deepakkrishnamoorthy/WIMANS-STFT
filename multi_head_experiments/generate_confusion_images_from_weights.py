import argparse
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(ROOT, "WiMANS-main", "benchmark", "wifi_csi"))

from dataset_multi_head import MultiHeadSTFTDataset
from load_data import load_data_y
from model_multi_head import CLSTMMultiHead, ResNet18MultiHead, THATStyleMultiHead


ACTIVITY_NAMES = ["nothing", "walk", "rotation", "jump", "wave", "lie_down", "pick_up", "sit_down", "stand_up"]
ENVIRONMENTS = ["classroom", "meeting_room", "empty_room"]
BAND_GROUPS = {"2.4": ["2.4"], "5": ["5"], "both": ["2.4", "5"]}
FEATURE_DIRS = {
    "multichannel": r"D:\Deepak\wifi_csi\dataset\stft_top5_multichannel_npy",
    "pca": r"D:\Deepak\wifi_csi\dataset\stft_top5_npy",
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
    parser = argparse.ArgumentParser(description="Generate confusion matrix images from saved multi-head checkpoints.")
    parser.add_argument("--weights-dir", required=True)
    parser.add_argument("--analysis-excel", default=None, help="Optional run Excel; used for exact thresholds/config when available.")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--annotation", default=r"D:\Deepak\wifi_csi\WiMANS-main\dataset\annotation.csv")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--normalize", default="log_standard")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--split-seed", type=int, default=39)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-tables", action="store_true")
    parser.add_argument("--environment", choices=ENVIRONMENTS, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def read_key_value_config(path):
    if not path or not os.path.exists(path):
        return {}
    cfg = pd.read_excel(path, sheet_name="config")
    if {"key", "value"}.issubset(cfg.columns):
        return {str(row["key"]): row["value"] for _, row in cfg.iterrows()}
    return cfg.iloc[0].to_dict()


def read_thresholds(path):
    thresholds = {}
    if not path or not os.path.exists(path):
        return thresholds
    repeats = pd.read_excel(path, sheet_name="repeats")
    for _, row in repeats.iterrows():
        thresholds[(str(row["environment"]), int(row["seed"]))] = float(row.get("threshold", 0.5))
    return thresholds


def discover_weights(weights_dir):
    rows = []
    for name in sorted(os.listdir(weights_dir)):
        match = WEIGHT_RE.match(name)
        if not match:
            continue
        row = match.groupdict()
        row["seed"] = int(row["seed"])
        row["path"] = os.path.join(weights_dir, name)
        rows.append(row)
    if not rows:
        raise FileNotFoundError(f"No matching multi-head .pth files found in {weights_dir}")
    return rows


def infer_input_channels(data_dir):
    for filename in os.listdir(data_dir):
        if filename.endswith(".npy"):
            sample = np.load(os.path.join(data_dir, filename), mmap_mode="r")
            return 1 if sample.ndim == 2 else sample.shape[0]
    raise FileNotFoundError(f"No .npy files found in {data_dir}")


def split_test(data_pd_y, split_seed, test_size, val_size):
    train_val_y, test_y = train_test_split(data_pd_y, test_size=test_size, shuffle=True, random_state=split_seed)
    train_test_split(train_val_y, test_size=val_size, shuffle=True, random_state=split_seed)
    return test_y


def collect_predictions(model, dataset, args, device):
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    labels = {"activity_set": [], "occupancy": [], "slot_activity": []}
    probs = {"activity_set": [], "occupancy": [], "slot_activity": []}
    model.eval()
    with torch.no_grad():
        for batch in loader:
            outputs = model(batch["x"].to(device))
            for key in labels:
                labels[key].append(batch[key].numpy())
                probs[key].append(torch.sigmoid(outputs[key]).cpu().numpy())
    return {k: np.vstack(v) for k, v in labels.items()}, {k: np.vstack(v) for k, v in probs.items()}


def binary_counts(y_true, y_pred):
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    rows = []
    for idx in range(y_true.shape[1]):
        true_col = y_true[:, idx]
        pred_col = y_pred[:, idx]
        rows.append([
            int(((true_col == 0) & (pred_col == 0)).sum()),
            int(((true_col == 0) & (pred_col == 1)).sum()),
            int(((true_col == 1) & (pred_col == 0)).sum()),
            int(((true_col == 1) & (pred_col == 1)).sum()),
        ])
    return np.array(rows)


def binary_f1_by_label(y_true, y_pred):
    return np.array([
        f1_score(y_true[:, idx].astype(int), y_pred[:, idx].astype(int), zero_division=0) * 100.0
        for idx in range(y_true.shape[1])
    ])


def active_activity_confusion(slot_true, slot_probs, threshold):
    true_user = slot_true.reshape(-1, 6, 9).astype(int)
    prob_user = slot_probs.reshape(-1, 6, 9)
    pred_user = (prob_user > threshold).astype(int)
    true_rows = true_user.reshape(-1, 9)
    pred_rows = pred_user.reshape(-1, 9)
    prob_rows = prob_user.reshape(-1, 9)
    active_mask = true_rows.sum(axis=1) > 0
    matrix = np.zeros((9, 10), dtype=int)
    for true_vec, pred_vec, prob_vec in zip(true_rows[active_mask], pred_rows[active_mask], prob_rows[active_mask]):
        true_idx = int(np.argmax(true_vec))
        pred_idx = 9 if pred_vec.sum() == 0 else int(np.argmax(prob_vec))
        matrix[true_idx, pred_idx] += 1
    return matrix


def slot_f1_heatmap(slot_true, slot_pred):
    values = binary_f1_by_label(slot_true, slot_pred)
    return values.reshape(6, 9)


def plot_heatmap(matrix, row_labels, col_labels, title, path, fmt=".0f", cmap="Blues"):
    fig_w = max(8, len(col_labels) * 0.85)
    fig_h = max(5, len(row_labels) * 0.45)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(matrix, cmap=cmap, aspect="auto")
    ax.set_title(title)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, format(matrix[i, j], fmt), ha="center", va="center", color="black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def print_matrix(title, matrix, row_labels, col_labels):
    df = pd.DataFrame(matrix, index=row_labels, columns=col_labels)
    print("\n" + title)
    print(df.to_string())


def summarize_metrics(labels, probs, threshold):
    activity_pred = (probs["activity_set"] > threshold).astype(int)
    occupancy_pred = (probs["occupancy"] > threshold).astype(int)
    slot_pred = (probs["slot_activity"] > threshold).astype(int)
    slot_true = labels["slot_activity"].astype(int)
    true_occ = labels["occupancy"].astype(int)
    true_activity = labels["activity_set"].astype(int)
    true_user = slot_true.reshape(-1, 6, 9)
    pred_user = slot_pred.reshape(-1, 6, 9)
    slot_activity_pred = (pred_user.sum(axis=1) > 0).astype(int)
    slot_occ_pred = (pred_user.sum(axis=2) > 0).astype(int)
    active_mask = true_occ.reshape(-1) == 1
    return {
        "activity_head_f1": f1_score(true_activity, activity_pred, average="micro", zero_division=0) * 100.0,
        "occupancy_head_f1": f1_score(true_occ, occupancy_pred, average="micro", zero_division=0) * 100.0,
        "slot_activity_set_f1": f1_score(true_activity, slot_activity_pred, average="micro", zero_division=0) * 100.0,
        "slot_occupancy_f1": f1_score(true_occ, slot_occ_pred, average="micro", zero_division=0) * 100.0,
        "slot54_f1": f1_score(slot_true, slot_pred, average="micro", zero_division=0) * 100.0,
        "active_slot_f1": f1_score(true_user.reshape(-1, 9)[active_mask], pred_user.reshape(-1, 9)[active_mask], average="micro", zero_division=0) * 100.0,
        "true_active_slots": true_occ.sum(axis=1).mean(),
        "pred_active_slots_slot_head": slot_occ_pred.sum(axis=1).mean(),
        "pred_active_slots_occupancy_head": occupancy_pred.sum(axis=1).mean(),
    }


def main():
    args = parse_args()
    weights_dir = os.path.abspath(args.weights_dir)
    out_dir = os.path.abspath(args.out_dir or os.path.join(weights_dir, "confusion_images"))
    os.makedirs(out_dir, exist_ok=True)

    cfg = read_key_value_config(args.analysis_excel)
    thresholds = read_thresholds(args.analysis_excel)
    rows = discover_weights(weights_dir)
    if args.environment is not None:
        rows = [row for row in rows if row["environment"] == args.environment]
    if args.seed is not None:
        rows = [row for row in rows if row["seed"] == args.seed]
    if not rows:
        raise FileNotFoundError("No checkpoints matched the requested filters.")
    first = rows[0]

    features = str(cfg.get("features", first["features"]))
    data_dir = args.data_dir or str(cfg.get("data_dir", FEATURE_DIRS[features]))
    annotation = str(cfg.get("annotation", args.annotation))
    normalize = str(cfg.get("normalize", args.normalize))
    split_seed = int(cfg.get("split_seed", args.split_seed))
    test_size = float(cfg.get("test_size", args.test_size))
    val_size = float(cfg.get("val_size", args.val_size))
    input_channels = infer_input_channels(data_dir)
    device = torch.device(args.device)

    table_rows = []
    summary_rows = []

    for row in rows:
        model_name = row["model"]
        env = row["environment"]
        band = row["band"]
        seed = row["seed"]
        threshold = thresholds.get((env, seed), args.threshold)
        threshold_source = "analysis_excel" if (env, seed) in thresholds else "command_default"
        tag = f"{model_name}_{row['features']}_{env}_band{band}_seed{seed}"

        data_pd_y = load_data_y(
            annotation,
            var_environment=[env],
            var_wifi_band=BAND_GROUPS[band],
            var_num_users=["0", "1", "2", "3", "4", "5"],
        )
        test_y = split_test(data_pd_y, split_seed, test_size, val_size)
        dataset = MultiHeadSTFTDataset(test_y, data_dir, max_len=200, normalize=normalize)

        model = MODEL_FACTORIES[model_name](input_channels).to(device)
        state = torch.load(row["path"], map_location=device)
        model.load_state_dict(state)
        labels, probs = collect_predictions(model, dataset, args, device)

        activity_pred = (probs["activity_set"] > threshold).astype(int)
        occupancy_pred = (probs["occupancy"] > threshold).astype(int)
        slot_pred = (probs["slot_activity"] > threshold).astype(int)
        slot_true = labels["slot_activity"].astype(int)
        true_occ = labels["occupancy"].astype(int)
        slot_occ_pred = (slot_pred.reshape(-1, 6, 9).sum(axis=2) > 0).astype(int)

        metrics = summarize_metrics(labels, probs, threshold)
        metrics.update({
            "model": model_name,
            "features": row["features"],
            "environment": env,
            "band": band,
            "seed": seed,
            "threshold": threshold,
            "threshold_source": threshold_source,
        })
        summary_rows.append(metrics)

        print("\n" + "=" * 100)
        print(f"{tag} | threshold={threshold:.2f} ({threshold_source}) | checkpoint={row['path']}")
        print(
            f"ActivityHead F1={metrics['activity_head_f1']:.2f}% | "
            f"OccupancyHead F1={metrics['occupancy_head_f1']:.2f}% | "
            f"SlotActSet F1={metrics['slot_activity_set_f1']:.2f}% | "
            f"ActiveSlot F1={metrics['active_slot_f1']:.2f}% | "
            f"54-F1={metrics['slot54_f1']:.2f}%"
        )
        print(
            f"Active slots true={metrics['true_active_slots']:.2f}, "
            f"slot-head pred={metrics['pred_active_slots_slot_head']:.2f}, "
            f"occupancy-head pred={metrics['pred_active_slots_occupancy_head']:.2f}"
        )

        activity_counts = binary_counts(labels["activity_set"], activity_pred)
        occupancy_counts = binary_counts(true_occ, occupancy_pred)
        slot_occupancy_counts = binary_counts(true_occ, slot_occ_pred)
        slot_f1 = slot_f1_heatmap(slot_true, slot_pred)
        active_cm = active_activity_confusion(slot_true, probs["slot_activity"], threshold)

        print_matrix("Activity head 2x2 counts per activity [TN FP FN TP]", activity_counts, ACTIVITY_NAMES, ["TN", "FP", "FN", "TP"])
        print_matrix("Occupancy head 2x2 counts per user [TN FP FN TP]", occupancy_counts, [f"user_{i}" for i in range(1, 7)], ["TN", "FP", "FN", "TP"])
        print_matrix("Slot collapsed occupancy 2x2 counts per user [TN FP FN TP]", slot_occupancy_counts, [f"user_{i}" for i in range(1, 7)], ["TN", "FP", "FN", "TP"])
        print_matrix("Active user-slot activity confusion", active_cm, ACTIVITY_NAMES, ACTIVITY_NAMES + ["missed"])

        plot_heatmap(activity_counts, ACTIVITY_NAMES, ["TN", "FP", "FN", "TP"], f"{tag} activity head counts", os.path.join(out_dir, f"{tag}_activity_head_counts.png"), cmap="YlGnBu")
        plot_heatmap(occupancy_counts, [f"user_{i}" for i in range(1, 7)], ["TN", "FP", "FN", "TP"], f"{tag} occupancy head counts", os.path.join(out_dir, f"{tag}_occupancy_head_counts.png"), cmap="YlOrBr")
        plot_heatmap(slot_occupancy_counts, [f"user_{i}" for i in range(1, 7)], ["TN", "FP", "FN", "TP"], f"{tag} slot collapsed occupancy counts", os.path.join(out_dir, f"{tag}_slot_occupancy_counts.png"), cmap="YlOrRd")
        plot_heatmap(slot_f1, [f"user_{i}" for i in range(1, 7)], ACTIVITY_NAMES, f"{tag} 54-label F1 heatmap", os.path.join(out_dir, f"{tag}_slot54_f1_heatmap.png"), fmt=".1f", cmap="viridis")
        plot_heatmap(active_cm, ACTIVITY_NAMES, ACTIVITY_NAMES + ["missed"], f"{tag} active-slot activity confusion", os.path.join(out_dir, f"{tag}_active_slot_activity_confusion.png"), cmap="Blues")

        for label, values in [
            ("activity_head", binary_f1_by_label(labels["activity_set"], activity_pred)),
            ("occupancy_head", binary_f1_by_label(true_occ, occupancy_pred)),
            ("slot_occupancy", binary_f1_by_label(true_occ, slot_occ_pred)),
        ]:
            names = ACTIVITY_NAMES if label == "activity_head" else [f"user_{i}" for i in range(1, 7)]
            for name, value in zip(names, values):
                table_rows.append({"tag": tag, "environment": env, "seed": seed, "table": label, "label": name, "f1": value})
        for user_idx in range(6):
            for activity_idx, activity in enumerate(ACTIVITY_NAMES):
                table_rows.append({
                    "tag": tag,
                    "environment": env,
                    "seed": seed,
                    "table": "slot54",
                    "label": f"user_{user_idx + 1}_{activity}",
                    "f1": slot_f1[user_idx, activity_idx],
                })

    print("\n" + "=" * 100)
    summary_df = pd.DataFrame(summary_rows)
    print("Summary across processed checkpoints")
    print(summary_df[[
        "environment", "seed", "threshold", "threshold_source", "activity_head_f1", "occupancy_head_f1",
        "slot_activity_set_f1", "active_slot_f1", "slot54_f1",
        "true_active_slots", "pred_active_slots_slot_head", "pred_active_slots_occupancy_head",
    ]].round(3).to_string(index=False))

    if args.save_tables:
        summary_df.to_csv(os.path.join(out_dir, "summary_metrics.csv"), index=False)
        pd.DataFrame(table_rows).to_csv(os.path.join(out_dir, "per_label_f1.csv"), index=False)

    print(f"\nSaved confusion images to: {out_dir}")


if __name__ == "__main__":
    main()
