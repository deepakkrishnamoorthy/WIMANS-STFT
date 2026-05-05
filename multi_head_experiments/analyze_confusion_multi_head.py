import argparse
import os
import sys

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


def parse_args():
    parser = argparse.ArgumentParser(description="Post-hoc confusion/F1 analysis for multi-head Top-5 STFT runs.")
    parser.add_argument("--analysis-excel", required=True, help="Excel file produced by train_multi_head_top5.py.")
    parser.add_argument("--out-excel", default=None, help="Output Excel file for confusion tables.")
    parser.add_argument("--annotation", default=None)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def read_config(path):
    cfg = pd.read_excel(path, sheet_name="config")
    if {"key", "value"}.issubset(cfg.columns):
        return {str(row["key"]): row["value"] for _, row in cfg.iterrows()}
    return cfg.iloc[0].to_dict()


def infer_input_channels(data_dir):
    for filename in os.listdir(data_dir):
        if filename.endswith(".npy"):
            sample = np.load(os.path.join(data_dir, filename), mmap_mode="r")
            return 1 if sample.ndim == 2 else sample.shape[0]
    raise FileNotFoundError(f"No .npy files found in {data_dir}")


def split_data(data_pd_y, split_seed, test_size, val_size):
    train_val_y, test_y = train_test_split(data_pd_y, test_size=test_size, shuffle=True, random_state=split_seed)
    train_test_split(train_val_y, test_size=val_size, shuffle=True, random_state=split_seed)
    return test_y


def load_checkpoint(path, model, device):
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.eval()


def collect_predictions(model, dataset, batch_size, num_workers, device):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    labels = {"activity_set": [], "occupancy": [], "slot_activity": []}
    probs = {"activity_set": [], "occupancy": [], "slot_activity": []}
    with torch.no_grad():
        for batch in loader:
            outputs = model(batch["x"].to(device))
            for key in labels:
                labels[key].append(batch[key].numpy())
                probs[key].append(torch.sigmoid(outputs[key]).cpu().numpy())
    return {k: np.vstack(v) for k, v in labels.items()}, {k: np.vstack(v) for k, v in probs.items()}


def binary_rows(y_true, y_pred, names, env, head):
    rows = []
    for idx, name in enumerate(names):
        true_col = y_true[:, idx].astype(int)
        pred_col = y_pred[:, idx].astype(int)
        tp = int(((true_col == 1) & (pred_col == 1)).sum())
        tn = int(((true_col == 0) & (pred_col == 0)).sum())
        fp = int(((true_col == 0) & (pred_col == 1)).sum())
        fn = int(((true_col == 1) & (pred_col == 0)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({
            "environment": env,
            "head": head,
            "label": name,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "support": int(true_col.sum()),
            "predicted_positive": int(pred_col.sum()),
            "precision": precision * 100.0,
            "recall": recall * 100.0,
            "f1": f1 * 100.0,
        })
    true_flat = y_true.astype(int).reshape(-1)
    pred_flat = y_pred.astype(int).reshape(-1)
    tp = int(((true_flat == 1) & (pred_flat == 1)).sum())
    tn = int(((true_flat == 0) & (pred_flat == 0)).sum())
    fp = int(((true_flat == 0) & (pred_flat == 1)).sum())
    fn = int(((true_flat == 1) & (pred_flat == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    rows.append({
        "environment": env,
        "head": head,
        "label": "MICRO_ALL",
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "support": int(true_flat.sum()),
        "predicted_positive": int(pred_flat.sum()),
        "precision": precision * 100.0,
        "recall": recall * 100.0,
        "f1": f1 * 100.0,
    })
    return rows


def active_slot_activity_matrix(slot_true, slot_probs, threshold, env):
    true_user = slot_true.reshape(-1, 6, 9).astype(int)
    prob_user = slot_probs.reshape(-1, 6, 9)
    pred_user = (prob_user > threshold).astype(int)
    true_rows = true_user.reshape(-1, 9)
    prob_rows = prob_user.reshape(-1, 9)
    pred_rows = pred_user.reshape(-1, 9)
    active_mask = true_rows.sum(axis=1) > 0
    matrix = pd.DataFrame(0, index=ACTIVITY_NAMES, columns=ACTIVITY_NAMES + ["missed"])
    for true_vec, pred_vec, prob_vec in zip(true_rows[active_mask], pred_rows[active_mask], prob_rows[active_mask]):
        true_name = ACTIVITY_NAMES[int(np.argmax(true_vec))]
        pred_name = "missed" if pred_vec.sum() == 0 else ACTIVITY_NAMES[int(np.argmax(prob_vec))]
        matrix.loc[true_name, pred_name] += 1
    matrix.insert(0, "environment", env)
    matrix.insert(1, "true_activity", matrix.index)
    return matrix.reset_index(drop=True)


def false_active_rows(slot_true, slot_probs, threshold, env):
    true_user = slot_true.reshape(-1, 6, 9).astype(int)
    prob_user = slot_probs.reshape(-1, 6, 9)
    pred_user = (prob_user > threshold).astype(int)
    true_rows = true_user.reshape(-1, 9)
    prob_rows = prob_user.reshape(-1, 9)
    pred_rows = pred_user.reshape(-1, 9)
    empty_mask = true_rows.sum(axis=1) == 0
    rows = []
    for idx, name in enumerate(ACTIVITY_NAMES):
        rows.append({
            "environment": env,
            "predicted_activity_on_empty_slot": name,
            "count": int(((pred_rows[empty_mask, idx] == 1)).sum()),
            "mean_probability": float(prob_rows[empty_mask, idx].mean()) if empty_mask.any() else 0.0,
        })
    rows.append({
        "environment": env,
        "predicted_activity_on_empty_slot": "ANY_FALSE_ACTIVE_SLOT",
        "count": int((pred_rows[empty_mask].sum(axis=1) > 0).sum()),
        "mean_probability": float(prob_rows[empty_mask].max(axis=1).mean()) if empty_mask.any() else 0.0,
    })
    return rows


def summarize(labels, probs, threshold, env):
    activity_pred = (probs["activity_set"] > threshold).astype(int)
    occupancy_pred = (probs["occupancy"] > threshold).astype(int)
    slot_pred = (probs["slot_activity"] > threshold).astype(int)
    slot_true = labels["slot_activity"].astype(int)
    true_activity_set = labels["activity_set"].astype(int)
    true_occupancy = labels["occupancy"].astype(int)
    true_user = slot_true.reshape(-1, 6, 9)
    pred_user = slot_pred.reshape(-1, 6, 9)
    slot_activity_set_pred = (pred_user.sum(axis=1) > 0).astype(int)
    slot_occupancy_pred = (pred_user.sum(axis=2) > 0).astype(int)
    active_mask = true_occupancy.reshape(-1) == 1
    true_user_rows = true_user.reshape(-1, 9)
    pred_user_rows = pred_user.reshape(-1, 9)
    return {
        "environment": env,
        "threshold": threshold,
        "activity_head_micro_f1": f1_score(true_activity_set, activity_pred, average="micro", zero_division=0) * 100.0,
        "occupancy_head_micro_f1": f1_score(true_occupancy, occupancy_pred, average="micro", zero_division=0) * 100.0,
        "slot_activity_set_micro_f1": f1_score(true_activity_set, slot_activity_set_pred, average="micro", zero_division=0) * 100.0,
        "slot_occupancy_micro_f1": f1_score(true_occupancy, slot_occupancy_pred, average="micro", zero_division=0) * 100.0,
        "slot54_micro_f1": f1_score(slot_true, slot_pred, average="micro", zero_division=0) * 100.0,
        "active_slot_activity_micro_f1": f1_score(true_user_rows[active_mask], pred_user_rows[active_mask], average="micro", zero_division=0) * 100.0,
        "true_active_slots_per_sample": true_occupancy.sum(axis=1).mean(),
        "pred_active_slots_from_slot_head": slot_occupancy_pred.sum(axis=1).mean(),
        "pred_active_slots_from_occupancy_head": occupancy_pred.sum(axis=1).mean(),
    }


def main():
    args = parse_args()
    cfg = read_config(args.analysis_excel)
    repeats = pd.read_excel(args.analysis_excel, sheet_name="repeats")

    model_name = str(cfg.get("model", repeats.iloc[0]["model"]))
    features = str(cfg.get("features", repeats.iloc[0].get("features", "multichannel")))
    data_dir = args.data_dir or str(cfg.get("data_dir", FEATURE_DIRS[features]))
    annotation = args.annotation or str(cfg.get("annotation", r"D:\Deepak\wifi_csi\WiMANS-main\dataset\annotation.csv"))
    normalize = str(cfg.get("normalize", "log_standard"))
    split_seed = int(cfg.get("split_seed", 39))
    test_size = float(cfg.get("test_size", 0.2))
    val_size = float(cfg.get("val_size", 0.2))

    device = torch.device(args.device)
    input_channels = infer_input_channels(data_dir)

    summary_rows = []
    activity_rows = []
    occupancy_rows = []
    slot_occupancy_rows = []
    slot54_rows = []
    active_activity_mats = []
    false_active = []

    for _, repeat in repeats.iterrows():
        env = repeat["environment"]
        band_name = str(repeat["band"])
        threshold = float(repeat.get("threshold", 0.5))
        model_path = str(repeat["model_path"])
        if not os.path.isabs(model_path):
            model_path = os.path.join(ROOT, model_path)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Checkpoint not found for {env}: {model_path}")

        wifi_bands = BAND_GROUPS[band_name]
        data_pd_y = load_data_y(
            annotation,
            var_environment=[env],
            var_wifi_band=wifi_bands,
            var_num_users=["0", "1", "2", "3", "4", "5"],
        )
        test_y = split_data(data_pd_y, split_seed, test_size, val_size)
        dataset = MultiHeadSTFTDataset(test_y, data_dir, max_len=200, normalize=normalize)

        model = MODEL_FACTORIES[model_name](input_channels).to(device)
        load_checkpoint(model_path, model, device)
        labels, probs = collect_predictions(model, dataset, args.batch_size, args.num_workers, device)

        activity_pred = (probs["activity_set"] > threshold).astype(int)
        occupancy_pred = (probs["occupancy"] > threshold).astype(int)
        slot_pred = (probs["slot_activity"] > threshold).astype(int)
        slot_true = labels["slot_activity"].astype(int)
        slot_true_user = slot_true.reshape(-1, 6, 9)
        slot_pred_user = slot_pred.reshape(-1, 6, 9)
        slot_occupancy_pred = (slot_pred_user.sum(axis=2) > 0).astype(int)
        true_occupancy = labels["occupancy"].astype(int)

        summary_rows.append(summarize(labels, probs, threshold, env))
        activity_rows.extend(binary_rows(labels["activity_set"], activity_pred, ACTIVITY_NAMES, env, "activity_head_9"))
        occupancy_rows.extend(binary_rows(true_occupancy, occupancy_pred, [f"user_{i}" for i in range(1, 7)], env, "occupancy_head_6"))
        slot_occupancy_rows.extend(binary_rows(true_occupancy, slot_occupancy_pred, [f"user_{i}" for i in range(1, 7)], env, "slot_collapsed_occupancy_6"))
        slot_names = [f"user_{u}_{name}" for u in range(1, 7) for name in ACTIVITY_NAMES]
        slot54_rows.extend(binary_rows(slot_true, slot_pred, slot_names, env, "slot_activity_54"))
        active_activity_mats.append(active_slot_activity_matrix(slot_true, probs["slot_activity"], threshold, env))
        false_active.extend(false_active_rows(slot_true, probs["slot_activity"], threshold, env))

    out_excel = args.out_excel
    if out_excel is None:
        stem = os.path.splitext(args.analysis_excel)[0]
        out_excel = f"{stem}_confusion.xlsx"
    os.makedirs(os.path.dirname(os.path.abspath(out_excel)), exist_ok=True)

    with pd.ExcelWriter(out_excel, engine="openpyxl") as writer:
        pd.DataFrame(summary_rows).to_excel(writer, sheet_name="summary", index=False)
        pd.DataFrame(activity_rows).to_excel(writer, sheet_name="activity_head_2x2", index=False)
        pd.DataFrame(occupancy_rows).to_excel(writer, sheet_name="occupancy_head_2x2", index=False)
        pd.DataFrame(slot_occupancy_rows).to_excel(writer, sheet_name="slot_occupancy_2x2", index=False)
        pd.DataFrame(slot54_rows).to_excel(writer, sheet_name="slot54_2x2_f1", index=False)
        pd.concat(active_activity_mats, ignore_index=True).to_excel(writer, sheet_name="active_slot_activity_cm", index=False)
        pd.DataFrame(false_active).to_excel(writer, sheet_name="false_active_empty_slots", index=False)
        pd.DataFrame([{"key": k, "value": str(v)} for k, v in cfg.items()]).to_excel(writer, sheet_name="source_config", index=False)
    print(f"Saved confusion analysis to {out_excel}")


if __name__ == "__main__":
    main()
