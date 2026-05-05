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
sys.path.insert(0, os.path.join(ROOT, "resnet_baseline"))
sys.path.append(os.path.join(ROOT, "WiMANS-main", "benchmark", "wifi_csi"))

from dataset_multichannel import STFTMultiChannelDataset
from load_data import load_data_y
from model_research_multichannel import (
    ResNet18MultiChannel,
    ResNetCLSTMMultiChannel,
    ResNetSlotAttentionMultiChannel,
    ResNetTHATStyleMultiChannel,
)


ENVIRONMENTS = ["classroom", "meeting_room", "empty_room"]
ACTIVITY_NAMES = ["nothing", "walk", "rotation", "jump", "wave", "lie_down", "pick_up", "sit_down", "stand_up"]
BAND_GROUPS = {"2.4": ["2.4"], "5": ["5"], "both": ["2.4", "5"]}
FEATURE_DIRS = {
    "multichannel": r"D:\Deepak\wifi_csi\dataset\stft_top5_multichannel_npy",
    "pca": r"D:\Deepak\wifi_csi\dataset\stft_top5_npy",
}
MODEL_FACTORIES = {
    "resnet18": lambda input_channels: ResNet18MultiChannel(input_channels=input_channels, num_classes=54),
    "clstm": lambda input_channels: ResNetCLSTMMultiChannel(input_channels=input_channels, num_classes=54),
    "slot_attention": lambda input_channels: ResNetSlotAttentionMultiChannel(input_channels=input_channels, num_users=6, num_activities=9),
    "that_style": lambda input_channels: ResNetTHATStyleMultiChannel(input_channels=input_channels, num_classes=54),
}


def parse_args():
    out_dir = os.path.join(ROOT, "top5_subcarrier_experiments", "outputs")
    parser = argparse.ArgumentParser(description="Diagnostic hard 54-label Top-5 STFT experiment.")
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
    parser.add_argument("--tune-threshold", choices=["none", "activity_set", "micro54"], default="activity_set")
    parser.add_argument("--select-metric", choices=["activity_set_micro_f1", "active_slot_micro_f1", "micro54"], default="activity_set_micro_f1")
    parser.add_argument("--normalize", choices=["log_standard", "log", "standard", "none"], default="log_standard")
    parser.add_argument("--split-seed", type=int, default=39)
    parser.add_argument("--seed-start", type=int, default=39)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--save-dir", default=os.path.join(out_dir, "saved_models_54_diagnostic"))
    parser.add_argument("--result-json", default=os.path.join(out_dir, "top5_54_diagnostic_results.json"))
    parser.add_argument("--result-excel", default=os.path.join(out_dir, "top5_54_diagnostic_analysis.xlsx"))
    return parser.parse_args()


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


def calculate_pos_weight(labels, clamp_max=50.0):
    labels_2d = labels.reshape(labels.shape[0], -1).astype(np.float32)
    pos = labels_2d.sum(axis=0)
    neg = labels_2d.shape[0] - pos
    return torch.tensor(np.clip(neg / np.maximum(pos, 1.0), 1.0, clamp_max), dtype=torch.float32)


def make_loader(dataset, args, shuffle):
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


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


def collect_outputs(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    labels_all, probs_all = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            total_loss += criterion(outputs, labels).item() * inputs.size(0)
            labels_all.append(labels.cpu().numpy())
            probs_all.append(torch.sigmoid(outputs).cpu().numpy())
    return np.vstack(labels_all), np.vstack(probs_all), total_loss / len(loader.dataset)


def compute_diagnostics(labels, probs, threshold):
    preds = (probs > threshold).astype(int)
    labels54 = labels.astype(int)
    labels_user = labels54.reshape(-1, 6, 9)
    preds_user = preds.reshape(-1, 6, 9)
    probs_user = probs.reshape(-1, 6, 9)

    true_activity_set = (labels_user.sum(axis=1) > 0).astype(int)
    pred_activity_set = (preds_user.sum(axis=1) > 0).astype(int)
    true_occupancy = (labels_user.sum(axis=2) > 0).astype(int)
    pred_occupancy = (preds_user.sum(axis=2) > 0).astype(int)

    active_mask = true_occupancy.reshape(-1) == 1
    empty_mask = ~active_mask
    true_user_rows = labels_user.reshape(-1, 9)
    pred_user_rows = preds_user.reshape(-1, 9)

    active_slot_micro_f1 = f1_score(true_user_rows[active_mask], pred_user_rows[active_mask], average="micro", zero_division=0) * 100.0
    active_slot_exact = accuracy_score(true_user_rows[active_mask], pred_user_rows[active_mask]) * 100.0
    empty_slot_accuracy = (pred_user_rows[empty_mask].sum(axis=1) == 0).mean() * 100.0

    per_user_rows = []
    for user_idx in range(6):
        per_user_rows.append({
            "user_slot": user_idx + 1,
            "occupancy_f1": f1_score(true_occupancy[:, user_idx], pred_occupancy[:, user_idx], zero_division=0) * 100.0,
            "slot_activity_micro_f1": f1_score(labels_user[:, user_idx, :], preds_user[:, user_idx, :], average="micro", zero_division=0) * 100.0,
            "true_active_rate": true_occupancy[:, user_idx].mean() * 100.0,
            "pred_active_rate": pred_occupancy[:, user_idx].mean() * 100.0,
        })

    precision, recall, f1, support = precision_recall_fscore_support(
        true_activity_set,
        pred_activity_set,
        average=None,
        zero_division=0,
    )
    per_activity_rows = []
    for idx, name in enumerate(ACTIVITY_NAMES):
        per_activity_rows.append({
            "activity": name,
            "activity_set_precision": precision[idx] * 100.0,
            "activity_set_recall": recall[idx] * 100.0,
            "activity_set_f1": f1[idx] * 100.0,
            "activity_set_support": int(support[idx]),
            "mean_probability": float(probs_user[:, :, idx].max(axis=1).mean()),
        })

    return {
        "threshold": float(threshold),
        "micro54": f1_score(labels54, preds, average="micro", zero_division=0) * 100.0,
        "macro54": f1_score(labels54, preds, average="macro", zero_division=0) * 100.0,
        "hamming54": hamming_loss(labels54, preds) * 100.0,
        "exact54": accuracy_score(labels54, preds) * 100.0,
        "activity_set_micro_f1": f1_score(true_activity_set, pred_activity_set, average="micro", zero_division=0) * 100.0,
        "activity_set_macro_f1": f1_score(true_activity_set, pred_activity_set, average="macro", zero_division=0) * 100.0,
        "activity_set_exact": accuracy_score(true_activity_set, pred_activity_set) * 100.0,
        "occupancy_micro_f1": f1_score(true_occupancy, pred_occupancy, average="micro", zero_division=0) * 100.0,
        "occupancy_exact": accuracy_score(true_occupancy, pred_occupancy) * 100.0,
        "active_slot_micro_f1": active_slot_micro_f1,
        "active_slot_exact": active_slot_exact,
        "empty_slot_accuracy": empty_slot_accuracy,
        "true_active_slots_per_sample": true_occupancy.sum(axis=1).mean(),
        "pred_active_slots_per_sample": pred_occupancy.sum(axis=1).mean(),
        "true_activity_count_per_sample": true_activity_set.sum(axis=1).mean(),
        "pred_activity_count_per_sample": pred_activity_set.sum(axis=1).mean(),
        "per_user": per_user_rows,
        "per_activity": per_activity_rows,
    }


def metric_for_selection(metrics, key):
    if key == "micro54":
        return metrics["micro54"]
    return metrics[key]


def tune_threshold(labels, probs, mode):
    if mode == "none":
        return None
    best_threshold, best_score = 0.5, -1.0
    for threshold in np.arange(0.1, 0.91, 0.05):
        metrics = compute_diagnostics(labels, probs, float(threshold))
        score = metrics["activity_set_micro_f1"] if mode == "activity_set" else metrics["micro54"]
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


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

    train_dataset = STFTMultiChannelDataset(train_y, args.data_dir, max_len=200, normalize=args.normalize)
    val_dataset = STFTMultiChannelDataset(val_y, args.data_dir, max_len=200, normalize=args.normalize)
    test_dataset = STFTMultiChannelDataset(test_y, args.data_dir, max_len=200, normalize=args.normalize)

    train_loader = make_loader(train_dataset, args, shuffle=True)
    val_loader = make_loader(val_dataset, args, shuffle=False)
    test_loader = make_loader(test_dataset, args, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MODEL_FACTORIES[args.model](input_channels).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=calculate_pos_weight(train_dataset.labels).to(device))
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_score = -1.0
    best_epoch = 0
    best_threshold = args.threshold
    best_model_wts = copy.deepcopy(model.state_dict())

    print(f"\n[*] 54-diagnostic Model={args.model} Features={args.features} Env={env_name} Band={band_name} Seed={seed}")
    print(f"    Channels={input_channels} Split train/val/test={len(train_dataset)}/{len(val_dataset)}/{len(test_dataset)} Select={args.select_metric} Tune={args.tune_threshold}")

    for epoch in range(args.epochs):
        start = time.time()
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
        train_loss /= len(train_loader.dataset)

        val_labels, val_probs, val_loss = collect_outputs(model, val_loader, criterion, device)
        threshold = tune_threshold(val_labels, val_probs, args.tune_threshold) or args.threshold
        val_metrics = compute_diagnostics(val_labels, val_probs, threshold)
        score = metric_for_selection(val_metrics, args.select_metric)
        if score > best_score:
            best_score = score
            best_epoch = epoch + 1
            best_threshold = threshold
            best_model_wts = copy.deepcopy(model.state_dict())

        print(
            f"    Epoch {epoch + 1:03d}/{args.epochs} - {time.time() - start:.1f}s"
            f" - Train Loss {train_loss:.4f} - Val Loss {val_loss:.4f}"
            f" - Val ActSet F1 {val_metrics['activity_set_micro_f1']:.2f}%"
            f" - Val ActiveSlot F1 {val_metrics['active_slot_micro_f1']:.2f}%"
            f" - Val 54-F1 {val_metrics['micro54']:.2f}%"
            f" - Th {threshold:.2f}"
        )

    model.load_state_dict(best_model_wts)
    test_labels, test_probs, test_loss = collect_outputs(model, test_loader, criterion, device)
    test_metrics = compute_diagnostics(test_labels, test_probs, best_threshold)
    test_metrics["loss"] = test_loss

    os.makedirs(args.save_dir, exist_ok=True)
    model_path = os.path.join(args.save_dir, f"54diag_{args.model}_{args.features}_{env_name}_{band_name}_seed{seed}.pth")
    torch.save(best_model_wts, model_path)

    print(
        f"    Final Test Only - Best Epoch {best_epoch} - Th {best_threshold:.2f}"
        f" - ActSet F1 {test_metrics['activity_set_micro_f1']:.2f}%"
        f" - ActiveSlot F1 {test_metrics['active_slot_micro_f1']:.2f}%"
        f" - 54-F1 {test_metrics['micro54']:.2f}%"
        f" - EmptySlot Acc {test_metrics['empty_slot_accuracy']:.2f}%"
    )
    test_metrics.update({"seed": seed, "best_epoch": best_epoch, "model_path": model_path})
    return test_metrics


def summarize(values):
    values = np.array(values, dtype=np.float64)
    return {"avg": float(values.mean()), "std": float(values.std()), "text": f"{values.mean():.2f}+/-{values.std():.2f}"}


def print_table(results):
    print("\n" + "=" * 104)
    print("Top-5 Hard 54-Label Diagnostic Results - Final Test Only")
    print("=" * 104)
    print(f"{'Band':<10} | {'Environment':<13} | {'ActSet F1':<14} | {'ActiveSlot F1':<14} | {'54-F1':<14} | {'EmptySlot Acc':<14}")
    print("-" * 104)
    for band_name, band_results in results.items():
        for env_name, env_results in band_results.items():
            print(
                f"{band_name:<10} | {env_name:<13} | "
                f"{env_results['activity_set_micro_f1']['text']:<14} | "
                f"{env_results['active_slot_micro_f1']['text']:<14} | "
                f"{env_results['micro54']['text']:<14} | "
                f"{env_results['empty_slot_accuracy']['text']:<14}"
            )
    print("=" * 104)


def write_excel(args, results):
    summary_rows, repeat_rows, per_user_rows, per_activity_rows = [], [], [], []
    for band_name, band_results in results.items():
        for env_name, env_results in band_results.items():
            summary_rows.append({
                "model": args.model,
                "features": args.features,
                "band": band_name,
                "environment": env_name,
                "activity_set_micro_f1_avg": env_results["activity_set_micro_f1"]["avg"],
                "active_slot_micro_f1_avg": env_results["active_slot_micro_f1"]["avg"],
                "micro54_avg": env_results["micro54"]["avg"],
                "empty_slot_accuracy_avg": env_results["empty_slot_accuracy"]["avg"],
                "occupancy_micro_f1_avg": env_results["occupancy_micro_f1"]["avg"],
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
                    "activity_set_micro_f1": repeat["activity_set_micro_f1"],
                    "active_slot_micro_f1": repeat["active_slot_micro_f1"],
                    "micro54": repeat["micro54"],
                    "macro54": repeat["macro54"],
                    "empty_slot_accuracy": repeat["empty_slot_accuracy"],
                    "occupancy_micro_f1": repeat["occupancy_micro_f1"],
                    "pred_active_slots_per_sample": repeat["pred_active_slots_per_sample"],
                    "true_active_slots_per_sample": repeat["true_active_slots_per_sample"],
                    "pred_activity_count_per_sample": repeat["pred_activity_count_per_sample"],
                    "true_activity_count_per_sample": repeat["true_activity_count_per_sample"],
                    "model_path": repeat["model_path"],
                })
                for row in repeat["per_user"]:
                    per_user_rows.append({"band": band_name, "environment": env_name, "seed": repeat["seed"], **row})
                for row in repeat["per_activity"]:
                    per_activity_rows.append({"band": band_name, "environment": env_name, "seed": repeat["seed"], **row})

    os.makedirs(os.path.dirname(args.result_excel), exist_ok=True)
    with pd.ExcelWriter(args.result_excel, engine="openpyxl") as writer:
        pd.DataFrame(summary_rows).to_excel(writer, sheet_name="summary", index=False)
        pd.DataFrame(repeat_rows).to_excel(writer, sheet_name="repeats", index=False)
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
                "activity_set_micro_f1": summarize([r["activity_set_micro_f1"] for r in repeats]),
                "active_slot_micro_f1": summarize([r["active_slot_micro_f1"] for r in repeats]),
                "micro54": summarize([r["micro54"] for r in repeats]),
                "empty_slot_accuracy": summarize([r["empty_slot_accuracy"] for r in repeats]),
                "occupancy_micro_f1": summarize([r["occupancy_micro_f1"] for r in repeats]),
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
