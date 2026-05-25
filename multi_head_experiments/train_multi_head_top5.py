import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, hamming_loss, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(ROOT, "WiMANS-main", "benchmark", "wifi_csi"))

from dataset_multi_head import MultiHeadSTFTDataset
from load_data import load_data_y
from model_multi_head import CLSTMMultiHead, ResNet18MultiHead, THATStyleMultiHead


ENVIRONMENTS = ["classroom", "meeting_room", "empty_room"]
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
SELECT_METRICS = [
    "combined_score",
    "micro54",
    "active_slot_micro_f1",
    "slot_activity_set_micro_f1",
    "direct_activity_set_micro_f1",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-head Top-5 STFT experiment for activity, occupancy, and 54 slot labels.")
    parser.add_argument("--annotation", default=r"D:\Deepak\wifi_csi\WiMANS-main\dataset\annotation.csv")
    parser.add_argument("--features", choices=["multichannel", "pca"], default="multichannel")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--model", choices=list(MODEL_FACTORIES), default="resnet18")
    parser.add_argument("--band", choices=["2.4", "5", "both", "all"], default="5")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--threshold-strategy", choices=["global", "per_head", "per_class"], default="global")
    parser.add_argument("--threshold-min", type=float, default=0.1)
    parser.add_argument("--threshold-max", type=float, default=0.9)
    parser.add_argument("--threshold-step", type=float, default=0.05)
    parser.add_argument(
        "--tune-threshold",
        choices=["none", "combined_score", "micro54", "active_slot_micro_f1", "direct_activity_set_micro_f1"],
        default="combined_score",
    )
    parser.add_argument("--select-metric", choices=SELECT_METRICS, default="combined_score")
    parser.add_argument("--normalize", choices=["log_standard", "log", "standard", "none"], default="log_standard")
    parser.add_argument("--loss-activity-set", type=float, default=0.5)
    parser.add_argument("--loss-occupancy", type=float, default=0.5)
    parser.add_argument("--loss-slot-activity", type=float, default=1.0)
    parser.add_argument("--activity-pos-weight-scale", type=float, default=1.0)
    parser.add_argument("--occupancy-pos-weight-scale", type=float, default=1.0)
    parser.add_argument("--slot-pos-weight-scale", type=float, default=1.0)
    parser.add_argument("--activity-pos-weight-fixed", type=float, default=None)
    parser.add_argument("--occupancy-pos-weight-fixed", type=float, default=None)
    parser.add_argument("--slot-pos-weight-fixed", type=float, default=None)
    parser.add_argument("--consistency-activity-set", type=float, default=0.0)
    parser.add_argument("--consistency-occupancy", type=float, default=0.0)
    parser.add_argument("--active-count-regularizer", type=float, default=0.0)
    parser.add_argument("--occupancy-gate", choices=["none", "binary", "prob"], default="none")
    parser.add_argument("--occupancy-gate-power", type=float, default=1.0)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--time-mask-width", type=int, default=0)
    parser.add_argument("--freq-mask-width", type=int, default=0)
    parser.add_argument("--channel-drop-prob", type=float, default=0.0)
    parser.add_argument("--noise-std", type=float, default=0.0)
    parser.add_argument("--time-shift", type=int, default=0)
    parser.add_argument("--mixup-alpha", type=float, default=0.0)
    parser.add_argument("--split-seed", type=int, default=39)
    parser.add_argument("--seed-start", type=int, default=39)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--result-json", default=None)
    parser.add_argument("--result-excel", default=None)
    args = parser.parse_args()

    out_dir = os.path.join(ROOT, "multi_head_experiments", "outputs")
    tag = f"multi_head_{args.model}_{args.features}_band{args.band}".replace(".", "p")
    args.save_dir = args.save_dir or os.path.join(out_dir, f"saved_models_{tag}")
    args.result_json = args.result_json or os.path.join(out_dir, f"{tag}_results.json")
    args.result_excel = args.result_excel or os.path.join(out_dir, f"{tag}_analysis.xlsx")
    return args


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def infer_input_channels(data_dir):
    for filename in os.listdir(data_dir):
        if filename.endswith(".npy"):
            sample = np.load(os.path.join(data_dir, filename), mmap_mode="r")
            return 1 if sample.ndim == 2 else sample.shape[0]
    raise FileNotFoundError(f"No .npy files found in {data_dir}")


def calculate_pos_weight(labels, scale=1.0, fixed=None, clamp_max=50.0):
    labels = labels.reshape(labels.shape[0], -1).astype(np.float32)
    if fixed is not None:
        return torch.full((labels.shape[1],), float(fixed), dtype=torch.float32)
    pos = labels.sum(axis=0)
    neg = labels.shape[0] - pos
    base = np.clip(neg / np.maximum(pos, 1.0), 1.0, clamp_max)
    weighted = 1.0 + (base - 1.0) * scale
    return torch.tensor(weighted, dtype=torch.float32)


def split_data(data_pd_y, args):
    train_val_y, test_y = train_test_split(
        data_pd_y,
        test_size=args.test_size,
        shuffle=True,
        random_state=args.split_seed,
    )
    train_y, val_y = train_test_split(
        train_val_y,
        test_size=args.val_size,
        shuffle=True,
        random_state=args.split_seed,
    )
    return train_y, val_y, test_y


def make_loader(dataset, args, shuffle):
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def make_criteria(train_dataset, args, device):
    return {
        "activity_set": nn.BCEWithLogitsLoss(
            pos_weight=calculate_pos_weight(
                train_dataset.activity_set,
                scale=args.activity_pos_weight_scale,
                fixed=args.activity_pos_weight_fixed,
            ).to(device)
        ),
        "occupancy": nn.BCEWithLogitsLoss(
            pos_weight=calculate_pos_weight(
                train_dataset.occupancy,
                scale=args.occupancy_pos_weight_scale,
                fixed=args.occupancy_pos_weight_fixed,
            ).to(device)
        ),
        "slot_activity": nn.BCEWithLogitsLoss(
            pos_weight=calculate_pos_weight(
                train_dataset.slot_activity,
                scale=args.slot_pos_weight_scale,
                fixed=args.slot_pos_weight_fixed,
            ).to(device)
        ),
    }


def probabilistic_or(probs, dim):
    return 1.0 - torch.prod(1.0 - probs.clamp(1e-6, 1.0 - 1e-6), dim=dim)


def combined_loss(outputs, targets, criteria, args):
    loss = (
        args.loss_activity_set * criteria["activity_set"](outputs["activity_set"], targets["activity_set"])
        + args.loss_occupancy * criteria["occupancy"](outputs["occupancy"], targets["occupancy"])
        + args.loss_slot_activity * criteria["slot_activity"](outputs["slot_activity"], targets["slot_activity"])
    )
    slot_probs = torch.sigmoid(outputs["slot_activity"]).view(-1, 6, 9)
    if args.consistency_activity_set > 0.0:
        slot_activity_set = probabilistic_or(slot_probs, dim=1)
        direct_activity_set = torch.sigmoid(outputs["activity_set"])
        loss = loss + args.consistency_activity_set * nn.functional.mse_loss(direct_activity_set, slot_activity_set)
    if args.consistency_occupancy > 0.0:
        slot_occupancy = probabilistic_or(slot_probs, dim=2)
        direct_occupancy = torch.sigmoid(outputs["occupancy"])
        loss = loss + args.consistency_occupancy * nn.functional.mse_loss(direct_occupancy, slot_occupancy)
    if args.active_count_regularizer > 0.0:
        slot_occupancy = probabilistic_or(slot_probs, dim=2)
        pred_active_count = slot_occupancy.sum(dim=1)
        true_active_count = targets["occupancy"].sum(dim=1)
        loss = loss + args.active_count_regularizer * nn.functional.smooth_l1_loss(pred_active_count, true_active_count)
    return loss


def batch_to_device(batch, device):
    return {
        "x": batch["x"].to(device),
        "activity_set": batch["activity_set"].to(device),
        "occupancy": batch["occupancy"].to(device),
        "slot_activity": batch["slot_activity"].to(device),
    }


def apply_mixup(batch, alpha):
    if alpha <= 0.0 or batch["x"].size(0) < 2:
        return batch
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(batch["x"].size(0), device=batch["x"].device)
    mixed = {}
    for key, value in batch.items():
        mixed[key] = lam * value + (1.0 - lam) * value[index]
    return mixed


def collect_outputs(model, loader, criteria, args, device):
    model.eval()
    total_loss = 0.0
    labels = {"activity_set": [], "occupancy": [], "slot_activity": []}
    probs = {"activity_set": [], "occupancy": [], "slot_activity": []}
    with torch.no_grad():
        for batch in loader:
            batch = batch_to_device(batch, device)
            outputs = model(batch["x"])
            loss = combined_loss(outputs, batch, criteria, args)
            total_loss += loss.item() * batch["x"].size(0)
            for key in labels:
                labels[key].append(batch[key].cpu().numpy())
                probs[key].append(torch.sigmoid(outputs[key]).cpu().numpy())
    return (
        {key: np.vstack(value) for key, value in labels.items()},
        {key: np.vstack(value) for key, value in probs.items()},
        total_loss / len(loader.dataset),
    )


def threshold_grid(args):
    return np.arange(args.threshold_min, args.threshold_max + 1e-9, args.threshold_step)


def make_default_thresholds(args):
    if args.threshold_strategy == "global":
        return float(args.threshold)
    if args.threshold_strategy == "per_head":
        return {
            "activity_set": float(args.threshold),
            "occupancy": float(args.threshold),
            "slot_activity": float(args.threshold),
        }
    return {
        "activity_set": np.full(9, float(args.threshold), dtype=np.float32),
        "occupancy": np.full(6, float(args.threshold), dtype=np.float32),
        "slot_activity": np.full(9, float(args.threshold), dtype=np.float32),
    }


def format_threshold(threshold):
    if isinstance(threshold, dict):
        serializable = {}
        for key, value in threshold.items():
            if isinstance(value, np.ndarray):
                serializable[key] = [round(float(x), 4) for x in value.tolist()]
            else:
                serializable[key] = round(float(value), 4)
        return json.dumps(serializable)
    return f"{float(threshold):.4f}"


def apply_binary_occupancy_gate(slot_pred, occupancy_pred, occupancy_gate):
    if occupancy_gate != "binary":
        return slot_pred
    gated_slot = slot_pred.reshape(-1, 6, 9).copy()
    gated_slot[occupancy_pred.reshape(-1, 6) == 0, :] = 0
    return gated_slot.reshape(-1, 54)


def predict_with_thresholds(probs, threshold, occupancy_gate="none", occupancy_gate_power=1.0):
    gated_probs = probs
    if occupancy_gate == "prob":
        gated_probs = dict(probs)
        occupancy_probs = np.clip(probs["occupancy"].reshape(-1, 6, 1), 0.0, 1.0)
        gate = np.power(occupancy_probs, occupancy_gate_power)
        gated_probs["slot_activity"] = (probs["slot_activity"].reshape(-1, 6, 9) * gate).reshape(-1, 54)

    if not isinstance(threshold, dict):
        occupancy_pred = (gated_probs["occupancy"] > float(threshold)).astype(int)
        slot_pred = (gated_probs["slot_activity"] > float(threshold)).astype(int)
        return {
            "activity_set": (gated_probs["activity_set"] > float(threshold)).astype(int),
            "occupancy": occupancy_pred,
            "slot_activity": apply_binary_occupancy_gate(slot_pred, occupancy_pred, occupancy_gate),
        }

    activity_threshold = threshold["activity_set"]
    occupancy_threshold = threshold["occupancy"]
    slot_threshold = threshold["slot_activity"]

    activity_pred = (gated_probs["activity_set"] > activity_threshold).astype(int)
    occupancy_pred = (gated_probs["occupancy"] > occupancy_threshold).astype(int)
    if isinstance(slot_threshold, np.ndarray) and slot_threshold.size == 9:
        slot_pred = (gated_probs["slot_activity"].reshape(-1, 6, 9) > slot_threshold.reshape(1, 1, 9)).astype(int).reshape(-1, 54)
    else:
        slot_pred = (gated_probs["slot_activity"] > slot_threshold).astype(int)
    slot_pred = apply_binary_occupancy_gate(slot_pred, occupancy_pred, occupancy_gate)
    return {
        "activity_set": activity_pred,
        "occupancy": occupancy_pred,
        "slot_activity": slot_pred,
    }


def compute_metrics(labels, probs, threshold, occupancy_gate="none", occupancy_gate_power=1.0):
    predictions = predict_with_thresholds(probs, threshold, occupancy_gate, occupancy_gate_power)
    direct_activity_pred = predictions["activity_set"]
    direct_occupancy_pred = predictions["occupancy"]
    slot_pred = predictions["slot_activity"]

    true_activity_set = labels["activity_set"].astype(int)
    true_occupancy = labels["occupancy"].astype(int)
    true_slot = labels["slot_activity"].astype(int)

    true_user = true_slot.reshape(-1, 6, 9)
    pred_user = slot_pred.reshape(-1, 6, 9)
    slot_activity_set_pred = (pred_user.sum(axis=1) > 0).astype(int)
    slot_occupancy_pred = (pred_user.sum(axis=2) > 0).astype(int)

    active_mask = true_occupancy.reshape(-1) == 1
    empty_mask = ~active_mask
    true_user_rows = true_user.reshape(-1, 9)
    pred_user_rows = pred_user.reshape(-1, 9)
    if active_mask.any():
        active_slot_micro_f1 = f1_score(true_user_rows[active_mask], pred_user_rows[active_mask], average="micro", zero_division=0) * 100.0
        active_slot_exact = accuracy_score(true_user_rows[active_mask], pred_user_rows[active_mask]) * 100.0
    else:
        active_slot_micro_f1 = 0.0
        active_slot_exact = 0.0
    empty_slot_accuracy = (pred_user_rows[empty_mask].sum(axis=1) == 0).mean() * 100.0 if empty_mask.any() else 0.0

    per_user_rows = []
    for user_idx in range(6):
        per_user_rows.append({
            "user_slot": user_idx + 1,
            "direct_occupancy_f1": f1_score(true_occupancy[:, user_idx], direct_occupancy_pred[:, user_idx], zero_division=0) * 100.0,
            "slot_occupancy_f1": f1_score(true_occupancy[:, user_idx], slot_occupancy_pred[:, user_idx], zero_division=0) * 100.0,
            "slot_activity_micro_f1": f1_score(true_user[:, user_idx, :], pred_user[:, user_idx, :], average="micro", zero_division=0) * 100.0,
            "true_active_rate": true_occupancy[:, user_idx].mean() * 100.0,
            "direct_pred_active_rate": direct_occupancy_pred[:, user_idx].mean() * 100.0,
            "slot_pred_active_rate": slot_occupancy_pred[:, user_idx].mean() * 100.0,
        })

    direct_precision, direct_recall, direct_f1, support = precision_recall_fscore_support(
        true_activity_set,
        direct_activity_pred,
        average=None,
        zero_division=0,
    )
    slot_precision, slot_recall, slot_f1, _ = precision_recall_fscore_support(
        true_activity_set,
        slot_activity_set_pred,
        average=None,
        zero_division=0,
    )
    per_activity_rows = []
    for idx, name in enumerate(ACTIVITY_NAMES):
        per_activity_rows.append({
            "activity": name,
            "direct_activity_precision": direct_precision[idx] * 100.0,
            "direct_activity_recall": direct_recall[idx] * 100.0,
            "direct_activity_f1": direct_f1[idx] * 100.0,
            "slot_collapsed_precision": slot_precision[idx] * 100.0,
            "slot_collapsed_recall": slot_recall[idx] * 100.0,
            "slot_collapsed_f1": slot_f1[idx] * 100.0,
            "activity_set_support": int(support[idx]),
            "direct_mean_probability": float(probs["activity_set"][:, idx].mean()),
            "slot_mean_probability": float(probs["slot_activity"].reshape(-1, 6, 9)[:, :, idx].max(axis=1).mean()),
        })

    direct_activity_f1_micro = f1_score(true_activity_set, direct_activity_pred, average="micro", zero_division=0) * 100.0
    direct_occupancy_f1_micro = f1_score(true_occupancy, direct_occupancy_pred, average="micro", zero_division=0) * 100.0
    slot_activity_set_f1_micro = f1_score(true_activity_set, slot_activity_set_pred, average="micro", zero_division=0) * 100.0
    slot_occupancy_f1_micro = f1_score(true_occupancy, slot_occupancy_pred, average="micro", zero_division=0) * 100.0
    micro54 = f1_score(true_slot, slot_pred, average="micro", zero_division=0) * 100.0

    metrics = {
        "threshold": format_threshold(threshold),
        "direct_activity_set_micro_f1": direct_activity_f1_micro,
        "direct_activity_set_exact": accuracy_score(true_activity_set, direct_activity_pred) * 100.0,
        "direct_occupancy_micro_f1": direct_occupancy_f1_micro,
        "direct_occupancy_exact": accuracy_score(true_occupancy, direct_occupancy_pred) * 100.0,
        "slot_activity_set_micro_f1": slot_activity_set_f1_micro,
        "slot_activity_set_exact": accuracy_score(true_activity_set, slot_activity_set_pred) * 100.0,
        "slot_occupancy_micro_f1": slot_occupancy_f1_micro,
        "slot_occupancy_exact": accuracy_score(true_occupancy, slot_occupancy_pred) * 100.0,
        "micro54": micro54,
        "macro54": f1_score(true_slot, slot_pred, average="macro", zero_division=0) * 100.0,
        "hamming54": hamming_loss(true_slot, slot_pred) * 100.0,
        "exact54": accuracy_score(true_slot, slot_pred) * 100.0,
        "active_slot_micro_f1": active_slot_micro_f1,
        "active_slot_exact": active_slot_exact,
        "empty_slot_accuracy": empty_slot_accuracy,
        "true_active_slots_per_sample": true_occupancy.sum(axis=1).mean(),
        "direct_pred_active_slots_per_sample": direct_occupancy_pred.sum(axis=1).mean(),
        "slot_pred_active_slots_per_sample": slot_occupancy_pred.sum(axis=1).mean(),
        "true_activity_count_per_sample": true_activity_set.sum(axis=1).mean(),
        "direct_pred_activity_count_per_sample": direct_activity_pred.sum(axis=1).mean(),
        "slot_pred_activity_count_per_sample": slot_activity_set_pred.sum(axis=1).mean(),
        "per_user": per_user_rows,
        "per_activity": per_activity_rows,
    }
    metrics["combined_score"] = (
        0.35 * metrics["micro54"]
        + 0.35 * metrics["active_slot_micro_f1"]
        + 0.15 * metrics["direct_activity_set_micro_f1"]
        + 0.15 * metrics["direct_occupancy_micro_f1"]
    )
    return metrics


def tune_global_threshold(labels, probs, mode, args):
    if mode == "none":
        return None
    best_threshold, best_score = 0.5, -1.0
    for threshold in threshold_grid(args):
        metrics = compute_metrics(
            labels,
            probs,
            float(threshold),
            getattr(args, "occupancy_gate", "none"),
            getattr(args, "occupancy_gate_power", 1.0),
        )
        score = metrics[mode]
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


def tune_binary_thresholds(y_true, y_prob, args):
    thresholds = np.full(y_true.shape[1], args.threshold, dtype=np.float32)
    for idx in range(y_true.shape[1]):
        best_threshold, best_score = args.threshold, -1.0
        for threshold in threshold_grid(args):
            pred = (y_prob[:, idx] > threshold).astype(int)
            score = f1_score(y_true[:, idx].astype(int), pred, zero_division=0)
            if score > best_score:
                best_score = score
                best_threshold = float(threshold)
        thresholds[idx] = best_threshold
    return thresholds


def tune_slot_activity_thresholds(labels, probs, args):
    y_true = labels["slot_activity"].reshape(-1, 6, 9).astype(int)
    y_prob = probs["slot_activity"].reshape(-1, 6, 9)
    thresholds = np.full(9, args.threshold, dtype=np.float32)
    for activity_idx in range(9):
        true_col = y_true[:, :, activity_idx].reshape(-1)
        prob_col = y_prob[:, :, activity_idx].reshape(-1)
        best_threshold, best_score = args.threshold, -1.0
        for threshold in threshold_grid(args):
            pred = (prob_col > threshold).astype(int)
            score = f1_score(true_col, pred, zero_division=0)
            if score > best_score:
                best_score = score
                best_threshold = float(threshold)
        thresholds[activity_idx] = best_threshold
    return thresholds


def tune_threshold(labels, probs, args):
    if args.tune_threshold == "none":
        return make_default_thresholds(args)
    if args.threshold_strategy == "global":
        return tune_global_threshold(labels, probs, args.tune_threshold, args) or float(args.threshold)
    if args.threshold_strategy == "per_head":
        return {
            "activity_set": tune_global_threshold(labels, probs, "direct_activity_set_micro_f1", args) or float(args.threshold),
            "occupancy": tune_global_threshold(labels, probs, "direct_occupancy_micro_f1", args) or float(args.threshold),
            "slot_activity": tune_global_threshold(labels, probs, "active_slot_micro_f1", args) or float(args.threshold),
        }
    return {
        "activity_set": tune_binary_thresholds(labels["activity_set"], probs["activity_set"], args),
        "occupancy": tune_binary_thresholds(labels["occupancy"], probs["occupancy"], args),
        "slot_activity": tune_slot_activity_thresholds(labels, probs, args),
    }


def train_once(args, env_name, band_name, wifi_bands, repeat_idx, input_channels):
    seed = args.seed_start + repeat_idx
    set_seed(seed)

    data_pd_y = load_data_y(
        args.annotation,
        var_environment=[env_name],
        var_wifi_band=wifi_bands,
        var_num_users=["0", "1", "2", "3", "4", "5"],
    )
    train_y, val_y, test_y = split_data(data_pd_y, args)

    train_dataset = MultiHeadSTFTDataset(
        train_y,
        args.data_dir,
        max_len=200,
        normalize=args.normalize,
        augment=args.augment,
        time_mask_width=args.time_mask_width,
        freq_mask_width=args.freq_mask_width,
        channel_drop_prob=args.channel_drop_prob,
        noise_std=args.noise_std,
        time_shift=args.time_shift,
    )
    val_dataset = MultiHeadSTFTDataset(val_y, args.data_dir, max_len=200, normalize=args.normalize)
    test_dataset = MultiHeadSTFTDataset(test_y, args.data_dir, max_len=200, normalize=args.normalize)

    train_loader = make_loader(train_dataset, args, shuffle=True)
    val_loader = make_loader(val_dataset, args, shuffle=False)
    test_loader = make_loader(test_dataset, args, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MODEL_FACTORIES[args.model](input_channels).to(device)
    criteria = make_criteria(train_dataset, args, device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_score = -1.0
    best_epoch = 0
    best_threshold = make_default_thresholds(args)
    best_model_wts = copy.deepcopy(model.state_dict())
    epoch_history = []

    print(f"\n[*] MultiHead Model={args.model} Features={args.features} Env={env_name} Band={band_name} Seed={seed}")
    print(
        f"    Channels={input_channels} Split train/val/test={len(train_dataset)}/{len(val_dataset)}/{len(test_dataset)}"
        f" Select={args.select_metric} Tune={args.tune_threshold} ThresholdStrategy={args.threshold_strategy}"
        f" LossWeights(activity/occupancy/slot)={args.loss_activity_set}/{args.loss_occupancy}/{args.loss_slot_activity}"
        f" PosScale(activity/occupancy/slot)={args.activity_pos_weight_scale}/{args.occupancy_pos_weight_scale}/{args.slot_pos_weight_scale}"
        f" PosFixed(activity/occupancy/slot)={args.activity_pos_weight_fixed}/{args.occupancy_pos_weight_fixed}/{args.slot_pos_weight_fixed}"
        f" Reg(activity/occupancy/count)={args.consistency_activity_set}/{args.consistency_occupancy}/{args.active_count_regularizer}"
        f" OccupancyGate={args.occupancy_gate} GatePower={args.occupancy_gate_power}"
    )
    if args.augment or args.mixup_alpha > 0.0:
        print(
            f"    Augment={args.augment} TimeMask={args.time_mask_width} FreqMask={args.freq_mask_width}"
            f" ChannelDrop={args.channel_drop_prob} NoiseStd={args.noise_std} TimeShift={args.time_shift}"
            f" MixupAlpha={args.mixup_alpha}"
        )

    for epoch in range(args.epochs):
        start = time.time()
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = batch_to_device(batch, device)
            batch = apply_mixup(batch, args.mixup_alpha)
            optimizer.zero_grad()
            outputs = model(batch["x"])
            loss = combined_loss(outputs, batch, criteria, args)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch["x"].size(0)
        train_loss /= len(train_loader.dataset)

        val_labels, val_probs, val_loss = collect_outputs(model, val_loader, criteria, args, device)
        threshold = tune_threshold(val_labels, val_probs, args)
        val_metrics = compute_metrics(val_labels, val_probs, threshold, args.occupancy_gate, args.occupancy_gate_power)
        score = val_metrics[args.select_metric]
        if score > best_score:
            best_score = score
            best_epoch = epoch + 1
            best_threshold = threshold
            best_model_wts = copy.deepcopy(model.state_dict())

        epoch_history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "threshold": format_threshold(threshold),
            "selected_score": score,
            "combined_score": val_metrics["combined_score"],
            "direct_activity_set_micro_f1": val_metrics["direct_activity_set_micro_f1"],
            "direct_occupancy_micro_f1": val_metrics["direct_occupancy_micro_f1"],
            "slot_activity_set_micro_f1": val_metrics["slot_activity_set_micro_f1"],
            "slot_occupancy_micro_f1": val_metrics["slot_occupancy_micro_f1"],
            "active_slot_micro_f1": val_metrics["active_slot_micro_f1"],
            "micro54": val_metrics["micro54"],
            "macro54": val_metrics["macro54"],
            "empty_slot_accuracy": val_metrics["empty_slot_accuracy"],
            "slot_pred_active_slots_per_sample": val_metrics["slot_pred_active_slots_per_sample"],
            "true_active_slots_per_sample": val_metrics["true_active_slots_per_sample"],
            "direct_pred_active_slots_per_sample": val_metrics["direct_pred_active_slots_per_sample"],
            "slot_pred_activity_count_per_sample": val_metrics["slot_pred_activity_count_per_sample"],
            "direct_pred_activity_count_per_sample": val_metrics["direct_pred_activity_count_per_sample"],
            "true_activity_count_per_sample": val_metrics["true_activity_count_per_sample"],
        })

        print(
            f"    Epoch {epoch + 1:03d}/{args.epochs} - {time.time() - start:.1f}s"
            f" - Train Loss {train_loss:.4f} - Val Loss {val_loss:.4f}"
            f" - Head ActSet F1 {val_metrics['direct_activity_set_micro_f1']:.2f}%"
            f" - Head Occ F1 {val_metrics['direct_occupancy_micro_f1']:.2f}%"
            f" - Slot Active F1 {val_metrics['active_slot_micro_f1']:.2f}%"
            f" - 54-F1 {val_metrics['micro54']:.2f}%"
            f" - Th {format_threshold(threshold)}"
        )

    model.load_state_dict(best_model_wts)
    test_labels, test_probs, test_loss = collect_outputs(model, test_loader, criteria, args, device)
    test_metrics = compute_metrics(test_labels, test_probs, best_threshold, args.occupancy_gate, args.occupancy_gate_power)
    test_metrics["loss"] = test_loss

    os.makedirs(args.save_dir, exist_ok=True)
    model_path = os.path.join(args.save_dir, f"multi_head_{args.model}_{args.features}_{env_name}_{band_name}_seed{seed}.pth")
    torch.save(best_model_wts, model_path)

    print(
        f"    Final Test Only - Best Epoch {best_epoch} - Th {format_threshold(best_threshold)}"
        f" - Head ActSet F1 {test_metrics['direct_activity_set_micro_f1']:.2f}%"
        f" - Head Occ F1 {test_metrics['direct_occupancy_micro_f1']:.2f}%"
        f" - Slot Active F1 {test_metrics['active_slot_micro_f1']:.2f}%"
        f" - 54-F1 {test_metrics['micro54']:.2f}%"
    )
    test_metrics.update({
        "seed": seed,
        "best_epoch": best_epoch,
        "model_path": model_path,
        "epoch_history": epoch_history,
        "threshold": format_threshold(best_threshold),
        "threshold_strategy": args.threshold_strategy,
    })
    return test_metrics


def summarize(values):
    values = np.array(values, dtype=np.float64)
    return {"avg": float(values.mean()), "std": float(values.std()), "text": f"{values.mean():.2f}+/-{values.std():.2f}"}


def print_table(results):
    print("\n" + "=" * 124)
    print("Multi-Head Top-5 STFT Results - Final Test Only")
    print("=" * 124)
    print(
        f"{'Band':<8} | {'Environment':<13} | {'Head ActSet':<14} | {'Head Occ':<14} | "
        f"{'Slot ActSet':<14} | {'ActiveSlot':<14} | {'54-F1':<14}"
    )
    print("-" * 124)
    for band_name, band_results in results.items():
        for env_name, env_results in band_results.items():
            print(
                f"{band_name:<8} | {env_name:<13} | "
                f"{env_results['direct_activity_set_micro_f1']['text']:<14} | "
                f"{env_results['direct_occupancy_micro_f1']['text']:<14} | "
                f"{env_results['slot_activity_set_micro_f1']['text']:<14} | "
                f"{env_results['active_slot_micro_f1']['text']:<14} | "
                f"{env_results['micro54']['text']:<14}"
            )
    print("=" * 124)


def write_excel(args, results):
    summary_rows, repeat_rows, epoch_rows, per_user_rows, per_activity_rows = [], [], [], [], []
    for band_name, band_results in results.items():
        for env_name, env_results in band_results.items():
            summary_rows.append({
                "model": args.model,
                "features": args.features,
                "band": band_name,
                "environment": env_name,
                "occupancy_gate": args.occupancy_gate,
                "occupancy_gate_power": args.occupancy_gate_power,
                "direct_activity_set_micro_f1_avg": env_results["direct_activity_set_micro_f1"]["avg"],
                "direct_occupancy_micro_f1_avg": env_results["direct_occupancy_micro_f1"]["avg"],
                "slot_activity_set_micro_f1_avg": env_results["slot_activity_set_micro_f1"]["avg"],
                "slot_occupancy_micro_f1_avg": env_results["slot_occupancy_micro_f1"]["avg"],
                "active_slot_micro_f1_avg": env_results["active_slot_micro_f1"]["avg"],
                "micro54_avg": env_results["micro54"]["avg"],
                "empty_slot_accuracy_avg": env_results["empty_slot_accuracy"]["avg"],
                "slot_pred_active_slots_per_sample_avg": env_results["slot_pred_active_slots_per_sample"]["avg"],
                "true_active_slots_per_sample_avg": env_results["true_active_slots_per_sample"]["avg"],
            })
            for repeat in env_results["repeats"]:
                repeat_rows.append({
                    "model": args.model,
                    "features": args.features,
                    "band": band_name,
                    "environment": env_name,
                    "seed": repeat["seed"],
                    "best_epoch": repeat["best_epoch"],
                    "threshold": repeat["threshold"],
                    "occupancy_gate": args.occupancy_gate,
                    "occupancy_gate_power": args.occupancy_gate_power,
                    "combined_score": repeat["combined_score"],
                    "direct_activity_set_micro_f1": repeat["direct_activity_set_micro_f1"],
                    "direct_activity_set_exact": repeat["direct_activity_set_exact"],
                    "direct_occupancy_micro_f1": repeat["direct_occupancy_micro_f1"],
                    "slot_activity_set_micro_f1": repeat["slot_activity_set_micro_f1"],
                    "slot_occupancy_micro_f1": repeat["slot_occupancy_micro_f1"],
                    "active_slot_micro_f1": repeat["active_slot_micro_f1"],
                    "micro54": repeat["micro54"],
                    "macro54": repeat["macro54"],
                    "empty_slot_accuracy": repeat["empty_slot_accuracy"],
                    "slot_pred_active_slots_per_sample": repeat["slot_pred_active_slots_per_sample"],
                    "direct_pred_active_slots_per_sample": repeat["direct_pred_active_slots_per_sample"],
                    "true_active_slots_per_sample": repeat["true_active_slots_per_sample"],
                    "slot_pred_activity_count_per_sample": repeat["slot_pred_activity_count_per_sample"],
                    "direct_pred_activity_count_per_sample": repeat["direct_pred_activity_count_per_sample"],
                    "true_activity_count_per_sample": repeat["true_activity_count_per_sample"],
                    "model_path": repeat["model_path"],
                })
                for row in repeat["epoch_history"]:
                    epoch_rows.append({
                        "model": args.model,
                        "features": args.features,
                        "band": band_name,
                        "environment": env_name,
                        "seed": repeat["seed"],
                        **row,
                    })
                for row in repeat["per_user"]:
                    per_user_rows.append({"band": band_name, "environment": env_name, "seed": repeat["seed"], **row})
                for row in repeat["per_activity"]:
                    per_activity_rows.append({"band": band_name, "environment": env_name, "seed": repeat["seed"], **row})

    os.makedirs(os.path.dirname(args.result_excel), exist_ok=True)
    with pd.ExcelWriter(args.result_excel, engine="openpyxl") as writer:
        pd.DataFrame(summary_rows).to_excel(writer, sheet_name="summary", index=False)
        pd.DataFrame(repeat_rows).to_excel(writer, sheet_name="repeats", index=False)
        pd.DataFrame(epoch_rows).to_excel(writer, sheet_name="epoch_history", index=False)
        pd.DataFrame(per_user_rows).to_excel(writer, sheet_name="per_user_slot", index=False)
        pd.DataFrame(per_activity_rows).to_excel(writer, sheet_name="per_activity", index=False)
        pd.DataFrame([{"key": k, "value": str(v)} for k, v in vars(args).items()]).to_excel(writer, sheet_name="config", index=False)
    print(f"Saved Excel analysis to {args.result_excel}")


def main():
    args = parse_args()
    args.data_dir = args.data_dir or FEATURE_DIRS[args.features]
    input_channels = infer_input_channels(args.data_dir)
    selected_bands = BAND_GROUPS if args.band == "all" else {args.band: BAND_GROUPS[args.band]}

    results = {}
    for band_name, wifi_bands in selected_bands.items():
        results[band_name] = {}
        for env_name in ENVIRONMENTS:
            repeats = [train_once(args, env_name, band_name, wifi_bands, idx, input_channels) for idx in range(args.repeat)]
            results[band_name][env_name] = {
                "direct_activity_set_micro_f1": summarize([r["direct_activity_set_micro_f1"] for r in repeats]),
                "direct_occupancy_micro_f1": summarize([r["direct_occupancy_micro_f1"] for r in repeats]),
                "slot_activity_set_micro_f1": summarize([r["slot_activity_set_micro_f1"] for r in repeats]),
                "slot_occupancy_micro_f1": summarize([r["slot_occupancy_micro_f1"] for r in repeats]),
                "active_slot_micro_f1": summarize([r["active_slot_micro_f1"] for r in repeats]),
                "micro54": summarize([r["micro54"] for r in repeats]),
                "empty_slot_accuracy": summarize([r["empty_slot_accuracy"] for r in repeats]),
                "slot_pred_active_slots_per_sample": summarize([r["slot_pred_active_slots_per_sample"] for r in repeats]),
                "true_active_slots_per_sample": summarize([r["true_active_slots_per_sample"] for r in repeats]),
                "repeats": repeats,
            }

    print_table(results)
    os.makedirs(os.path.dirname(args.result_json), exist_ok=True)
    with open(args.result_json, "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "results": results}, f, indent=4)
    print(f"Saved JSON results to {args.result_json}")
    write_excel(args, results)


if __name__ == "__main__":
    main()
