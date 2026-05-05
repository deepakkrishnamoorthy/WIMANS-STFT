import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, hamming_loss, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'WiMANS-main', 'benchmark', 'wifi_csi')))
from load_data import load_data_y

from dataset_multichannel import STFTMultiChannelDataset
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
    "pca": r"D:\Deepak\wifi_csi\dataset\stft_top5_npy",
    "multichannel": r"D:\Deepak\wifi_csi\dataset\stft_top5_multichannel_npy",
}
MODEL_FACTORIES = {
    "resnet18_multichannel": ResNet18MultiChannel,
    "clstm_multichannel": ResNetCLSTMMultiChannel,
    "slot_attention_multichannel": ResNetSlotAttentionMultiChannel,
    "that_style_multichannel": ResNetTHATStyleMultiChannel,
}


def parse_args(model_name, description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--annotation", default=r"D:\Deepak\wifi_csi\WiMANS-main\dataset\annotation.csv")
    parser.add_argument("--features", choices=["multichannel", "pca"], default="multichannel")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--band", choices=["2.4", "5", "both", "all"], default="all")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--normalize", choices=["log_standard", "log", "standard", "none"], default="log_standard")
    parser.add_argument("--split-seed", type=int, default=39)
    parser.add_argument("--seed-start", type=int, default=39)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--save-dir", default=os.path.join(os.path.dirname(__file__), f"saved_models_{model_name}"))
    parser.add_argument("--result-json", default=os.path.join(os.path.dirname(__file__), f"{model_name}_activity_holdout_results.json"))
    parser.add_argument("--result-excel", default=os.path.join(os.path.dirname(__file__), f"{model_name}_activity_holdout_analysis.xlsx"))
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


def activity_accuracy(labels, preds):
    return accuracy_score(labels.reshape(-1, 9).astype(int), preds.reshape(-1, 9).astype(int)) * 100.0


def calculate_pos_weight(labels, clamp_max=50.0):
    labels_2d = labels.reshape(labels.shape[0], -1).astype(np.float32)
    pos = labels_2d.sum(axis=0)
    neg = labels_2d.shape[0] - pos
    return torch.tensor(np.clip(neg / np.maximum(pos, 1.0), 1.0, clamp_max), dtype=torch.float32)


def analyze_activity_outputs(labels, probs, preds):
    labels_user = labels.reshape(-1, 9).astype(int)
    preds_user = preds.reshape(-1, 9).astype(int)
    probs_user = probs.reshape(-1, 9)
    active_mask = labels_user.sum(axis=1) == 1
    true_idx = labels_user[active_mask].argmax(axis=1)
    pred_idx = probs_user[active_mask].argmax(axis=1)
    cm = confusion_matrix(true_idx, pred_idx, labels=list(range(9)))
    cm_norm = cm.astype(np.float64) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    report = classification_report(true_idx, pred_idx, labels=list(range(9)), target_names=ACTIVITY_NAMES, output_dict=True, zero_division=0)
    precision, recall, f1, support = precision_recall_fscore_support(labels_user, preds_user, average=None, zero_division=0)
    pred_positive_count = preds_user.sum(axis=1)
    return {
        "threshold_none_rate": float((pred_positive_count == 0).mean() * 100.0),
        "threshold_multi_rate": float((pred_positive_count > 1).mean() * 100.0),
        "per_activity": [
            {
                "activity": name,
                "argmax_precision": float(report[name]["precision"]),
                "argmax_recall": float(report[name]["recall"]),
                "argmax_f1": float(report[name]["f1-score"]),
                "argmax_support": int(report[name]["support"]),
                "threshold_precision": float(precision[idx]),
                "threshold_recall": float(recall[idx]),
                "threshold_f1": float(f1[idx]),
                "threshold_support": int(support[idx]),
            }
            for idx, name in enumerate(ACTIVITY_NAMES)
        ],
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_normalized": cm_norm.tolist(),
    }


def evaluate(model, loader, criterion, device, threshold, include_analysis=False):
    model.eval()
    total_loss = 0.0
    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            total_loss += criterion(outputs, labels).item() * inputs.size(0)
            probs = torch.sigmoid(outputs)
            all_probs.append(probs.cpu().numpy())
            all_preds.append((probs > threshold).float().cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    labels = np.vstack(all_labels)
    preds = np.vstack(all_preds)
    probs = np.vstack(all_probs)
    metrics = {
        "loss": total_loss / len(loader.dataset),
        "activity_accuracy": activity_accuracy(labels, preds),
        "exact_54_accuracy": accuracy_score(labels.astype(int), preds.astype(int)) * 100.0,
        "micro_f1": f1_score(labels.astype(int), preds.astype(int), average="micro", zero_division=0) * 100.0,
        "macro_f1": f1_score(labels.astype(int), preds.astype(int), average="macro", zero_division=0) * 100.0,
        "hamming_loss": hamming_loss(labels.astype(int), preds.astype(int)) * 100.0,
    }
    if include_analysis:
        metrics["activity_analysis"] = analyze_activity_outputs(labels, probs, preds)
    return metrics


def make_loader(dataset, batch_size, shuffle, num_workers):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=torch.cuda.is_available())


def split_data(data_pd_y, args):
    train_val_y, test_y = train_test_split(data_pd_y, test_size=args.test_size, shuffle=True, random_state=args.split_seed)
    train_y, val_y = train_test_split(train_val_y, test_size=args.val_size, shuffle=True, random_state=args.split_seed)
    return train_y, val_y, test_y


def train_once(args, model_name, env_name, band_name, wifi_bands, repeat_idx, input_channels):
    seed = args.seed_start + repeat_idx
    set_seed(seed)
    data_pd_y = load_data_y(args.annotation, var_environment=[env_name], var_wifi_band=wifi_bands, var_num_users=["0", "1", "2", "3", "4", "5"])
    train_y, val_y, test_y = split_data(data_pd_y, args)
    train_dataset = STFTMultiChannelDataset(train_y, args.data_dir, max_len=200, normalize=args.normalize)
    val_dataset = STFTMultiChannelDataset(val_y, args.data_dir, max_len=200, normalize=args.normalize)
    test_dataset = STFTMultiChannelDataset(test_y, args.data_dir, max_len=200, normalize=args.normalize)
    train_loader = make_loader(train_dataset, args.batch_size, True, args.num_workers)
    val_loader = make_loader(val_dataset, args.batch_size, False, args.num_workers)
    test_loader = make_loader(test_dataset, args.batch_size, False, args.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MODEL_FACTORIES[model_name](input_channels=input_channels).to(device)
    pos_weight = calculate_pos_weight(train_dataset.labels).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_val_acc, best_epoch = -1.0, 0
    best_model_wts = copy.deepcopy(model.state_dict())

    print(f"\n[*] Model={model_name} Features={args.features} Env={env_name} Band={band_name} Seed={seed}")
    print(f"    Channels={input_channels} Split train/val/test={len(train_dataset)}/{len(val_dataset)}/{len(test_dataset)}")
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
        val_metrics = evaluate(model, val_loader, criterion, device, args.threshold)
        if val_metrics["activity_accuracy"] > best_val_acc:
            best_val_acc = val_metrics["activity_accuracy"]
            best_epoch = epoch + 1
            best_model_wts = copy.deepcopy(model.state_dict())
        print(f"    Epoch {epoch + 1:03d}/{args.epochs} - {time.time() - start:.1f}s - Train Loss {train_loss:.4f} - Val Loss {val_metrics['loss']:.4f} - Val Activity Acc {val_metrics['activity_accuracy']:.2f}% - Val Micro F1 {val_metrics['micro_f1']:.2f}%")

    model.load_state_dict(best_model_wts)
    test_metrics = evaluate(model, test_loader, criterion, device, args.threshold, include_analysis=True)
    os.makedirs(args.save_dir, exist_ok=True)
    model_path = os.path.join(args.save_dir, f"{model_name}_{args.features}_{env_name}_{band_name}_seed{seed}.pth")
    torch.save(best_model_wts, model_path)
    print(f"    Final Test Only - Best Epoch {best_epoch} - Test Loss {test_metrics['loss']:.4f} - Test Activity Acc {test_metrics['activity_accuracy']:.2f}% - Test Micro F1 {test_metrics['micro_f1']:.2f}%")
    test_metrics.update({"seed": seed, "best_epoch": best_epoch, "model_path": model_path})
    return test_metrics


def summarize(values):
    values = np.array(values, dtype=np.float64)
    return {"avg": float(values.mean()), "std": float(values.std()), "text": f"{values.mean():.2f}+/-{values.std():.2f}"}


def print_table(model_name, results):
    print("\n" + "=" * 96)
    print(f"{model_name} STFT Activity Results - Final Test Only")
    print("=" * 96)
    print(f"{'Band':<12} | {'Classroom':<18} | {'Meeting Room':<18} | {'Empty Room':<18}")
    print("-" * 96)
    for band_name in results:
        row = [band_name] + [results[band_name][env]["activity_accuracy"]["text"] for env in ENVIRONMENTS]
        print(f"{row[0]:<12} | {row[1]:<18} | {row[2]:<18} | {row[3]:<18}")
    print("=" * 96)


def write_excel(model_name, args, results):
    import pandas as pd
    summary_rows, per_activity_rows, confusion_rows, note_rows = [], [], [], []
    for band_name, band_results in results.items():
        for env_name, env_results in band_results.items():
            summary_rows.append({"model": model_name, "features": args.features, "band": band_name, "environment": env_name, "activity_accuracy_avg": env_results["activity_accuracy"]["avg"], "activity_accuracy_std": env_results["activity_accuracy"]["std"], "micro_f1_avg": env_results["micro_f1"]["avg"], "macro_f1_avg": env_results["macro_f1"]["avg"], "loss_avg": env_results["loss"]["avg"]})
            avg_activity = {name: [] for name in ACTIVITY_NAMES}
            avg_cm = np.zeros((9, 9), dtype=np.float64)
            for repeat in env_results["repeats"]:
                for row in repeat["activity_analysis"]["per_activity"]:
                    per_activity_rows.append({"model": model_name, "features": args.features, "band": band_name, "environment": env_name, "seed": repeat["seed"], **row})
                    avg_activity[row["activity"]].append(row)
                avg_cm += np.array(repeat["activity_analysis"]["confusion_matrix"], dtype=np.float64)
            avg_cm /= max(len(env_results["repeats"]), 1)
            for i, true_name in enumerate(ACTIVITY_NAMES):
                for j, pred_name in enumerate(ACTIVITY_NAMES):
                    confusion_rows.append({"model": model_name, "features": args.features, "band": band_name, "environment": env_name, "true_activity": true_name, "predicted_activity": pred_name, "count_avg": avg_cm[i, j]})
            cohort = [{"activity": k, "f1": float(np.mean([x["argmax_f1"] for x in v])), "recall": float(np.mean([x["argmax_recall"] for x in v]))} for k, v in avg_activity.items()]
            best, worst = max(cohort, key=lambda x: x["f1"]), min(cohort, key=lambda x: x["f1"])
            note = f"{model_name} ({args.features}) performed best on {best['activity']} and struggled most on {worst['activity']} for {band_name}/{env_name}."
            note_rows.append({"model": model_name, "features": args.features, "band": band_name, "environment": env_name, "best_activity": best["activity"], "best_f1": best["f1"], "weakest_activity": worst["activity"], "weakest_f1": worst["f1"], "note": note})
            print(note)
    with pd.ExcelWriter(args.result_excel, engine="openpyxl") as writer:
        pd.DataFrame(summary_rows).to_excel(writer, sheet_name="summary", index=False)
        pd.DataFrame(per_activity_rows).to_excel(writer, sheet_name="per_activity", index=False)
        pd.DataFrame(confusion_rows).to_excel(writer, sheet_name="confusion", index=False)
        pd.DataFrame(note_rows).to_excel(writer, sheet_name="cohort_notes", index=False)
        pd.DataFrame([{"key": k, "value": str(v)} for k, v in vars(args).items()]).to_excel(writer, sheet_name="config", index=False)
    print(f"Saved Excel analysis to {args.result_excel}")


def run_experiment(model_name, description):
    args = parse_args(model_name, description)
    args.data_dir = args.data_dir or FEATURE_DIRS[args.features]
    input_channels = infer_input_channels(args.data_dir)
    selected_bands = BAND_GROUPS if args.band == "all" else {args.band: BAND_GROUPS[args.band]}
    results = {}
    for band_name, wifi_bands in selected_bands.items():
        results[band_name] = {}
        for env_name in ENVIRONMENTS:
            repeats = [train_once(args, model_name, env_name, band_name, wifi_bands, idx, input_channels) for idx in range(args.repeat)]
            results[band_name][env_name] = {
                "activity_accuracy": summarize([r["activity_accuracy"] for r in repeats]),
                "exact_54_accuracy": summarize([r["exact_54_accuracy"] for r in repeats]),
                "micro_f1": summarize([r["micro_f1"] for r in repeats]),
                "macro_f1": summarize([r["macro_f1"] for r in repeats]),
                "hamming_loss": summarize([r["hamming_loss"] for r in repeats]),
                "loss": summarize([r["loss"] for r in repeats]),
                "repeats": repeats,
            }
    print_table(model_name, results)
    with open(args.result_json, "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "model": model_name, "results": results}, f, indent=4)
    print(f"Saved JSON results to {args.result_json}")
    write_excel(model_name, args, results)
