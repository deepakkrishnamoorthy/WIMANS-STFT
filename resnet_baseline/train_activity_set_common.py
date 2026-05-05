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
from sklearn.metrics import accuracy_score, f1_score, hamming_loss, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'WiMANS-main', 'benchmark', 'wifi_csi')))
from load_data import load_data_y

from dataset_activity_set import STFTActivitySetDataset
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
    "resnet18_activity_set": lambda input_channels: ResNet18MultiChannel(input_channels=input_channels, num_classes=9),
    "clstm_activity_set": lambda input_channels: ResNetCLSTMMultiChannel(input_channels=input_channels, num_classes=9),
    "slot_attention_activity_set": lambda input_channels: ResNetSlotAttentionMultiChannel(input_channels=input_channels, num_users=1, num_activities=9),
    "that_style_activity_set": lambda input_channels: ResNetTHATStyleMultiChannel(input_channels=input_channels, num_classes=9),
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
    parser.add_argument("--result-json", default=os.path.join(os.path.dirname(__file__), f"{model_name}_holdout_results.json"))
    parser.add_argument("--result-excel", default=os.path.join(os.path.dirname(__file__), f"{model_name}_holdout_analysis.xlsx"))
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
    labels = labels.astype(np.float32)
    pos = labels.sum(axis=0)
    neg = labels.shape[0] - pos
    return torch.tensor(np.clip(neg / np.maximum(pos, 1.0), 1.0, clamp_max), dtype=torch.float32)


def analyze(labels, probs, preds):
    precision, recall, f1, support = precision_recall_fscore_support(labels, preds, average=None, zero_division=0)
    per_activity = []
    for idx, name in enumerate(ACTIVITY_NAMES):
        per_activity.append({
            "activity": name,
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1": float(f1[idx]),
            "support": int(support[idx]),
            "positive_rate_true": float(labels[:, idx].mean() * 100.0),
            "positive_rate_pred": float(preds[:, idx].mean() * 100.0),
            "prob_mean": float(probs[:, idx].mean()),
        })
    return {
        "per_activity": per_activity,
        "samples_with_no_predicted_activity": float((preds.sum(axis=1) == 0).mean() * 100.0),
        "samples_with_multi_predicted_activity": float((preds.sum(axis=1) > 1).mean() * 100.0),
    }


def evaluate(model, loader, criterion, device, threshold, include_analysis=False):
    model.eval()
    total_loss = 0.0
    all_labels, all_probs, all_preds = [], [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            total_loss += criterion(outputs, labels).item() * inputs.size(0)
            probs = torch.sigmoid(outputs)
            all_probs.append(probs.cpu().numpy())
            all_preds.append((probs > threshold).float().cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    labels = np.vstack(all_labels).astype(int)
    probs = np.vstack(all_probs)
    preds = np.vstack(all_preds).astype(int)
    metrics = {
        "loss": total_loss / len(loader.dataset),
        "exact_set_accuracy": accuracy_score(labels, preds) * 100.0,
        "micro_f1": f1_score(labels, preds, average="micro", zero_division=0) * 100.0,
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0) * 100.0,
        "hamming_loss": hamming_loss(labels, preds) * 100.0,
    }
    if include_analysis:
        metrics["activity_analysis"] = analyze(labels, probs, preds)
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
    train_dataset = STFTActivitySetDataset(train_y, args.data_dir, max_len=200, normalize=args.normalize)
    val_dataset = STFTActivitySetDataset(val_y, args.data_dir, max_len=200, normalize=args.normalize)
    test_dataset = STFTActivitySetDataset(test_y, args.data_dir, max_len=200, normalize=args.normalize)

    train_loader = make_loader(train_dataset, args.batch_size, True, args.num_workers)
    val_loader = make_loader(val_dataset, args.batch_size, False, args.num_workers)
    test_loader = make_loader(test_dataset, args.batch_size, False, args.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MODEL_FACTORIES[model_name](input_channels).to(device)
    pos_weight = calculate_pos_weight(train_dataset.labels).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_f1 = -1.0
    best_epoch = 0
    best_model_wts = copy.deepcopy(model.state_dict())
    print(f"\n[*] Model={model_name} Target=activity-set Features={args.features} Env={env_name} Band={band_name} Seed={seed}")
    print(f"    Channels={input_channels} Split train/val/test={len(train_dataset)}/{len(val_dataset)}/{len(test_dataset)} PosWeightMedian={float(pos_weight.median().cpu()):.2f}")

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
        if val_metrics["micro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["micro_f1"]
            best_epoch = epoch + 1
            best_model_wts = copy.deepcopy(model.state_dict())
        print(f"    Epoch {epoch + 1:03d}/{args.epochs} - {time.time() - start:.1f}s - Train Loss {train_loss:.4f} - Val Loss {val_metrics['loss']:.4f} - Val Micro F1 {val_metrics['micro_f1']:.2f}% - Val Exact Set Acc {val_metrics['exact_set_accuracy']:.2f}%")

    model.load_state_dict(best_model_wts)
    test_metrics = evaluate(model, test_loader, criterion, device, args.threshold, include_analysis=True)
    os.makedirs(args.save_dir, exist_ok=True)
    model_path = os.path.join(args.save_dir, f"{model_name}_{args.features}_{env_name}_{band_name}_seed{seed}.pth")
    torch.save(best_model_wts, model_path)
    print(f"    Final Test Only - Best Epoch {best_epoch} - Test Loss {test_metrics['loss']:.4f} - Test Micro F1 {test_metrics['micro_f1']:.2f}% - Test Exact Set Acc {test_metrics['exact_set_accuracy']:.2f}%")
    test_metrics.update({"seed": seed, "best_epoch": best_epoch, "model_path": model_path})
    return test_metrics


def summarize(values):
    values = np.array(values, dtype=np.float64)
    return {"avg": float(values.mean()), "std": float(values.std()), "text": f"{values.mean():.2f}+/-{values.std():.2f}"}


def print_table(model_name, results):
    print("\n" + "=" * 96)
    print(f"{model_name} Activity-Set Results - Final Test Only")
    print("=" * 96)
    print(f"{'Band':<12} | {'Classroom':<18} | {'Meeting Room':<18} | {'Empty Room':<18}")
    print("-" * 96)
    for band_name in results:
        row = [band_name] + [results[band_name][env]["micro_f1"]["text"] for env in ENVIRONMENTS]
        print(f"{row[0]:<12} | {row[1]:<18} | {row[2]:<18} | {row[3]:<18}")
    print("=" * 96)
    print("Metric in table: micro-F1 for 9-label activity-set prediction.")


def write_excel(model_name, args, results):
    import pandas as pd
    summary_rows, repeat_rows, per_activity_rows, notes_rows = [], [], [], []
    for band_name, band_results in results.items():
        for env_name, env_results in band_results.items():
            summary_rows.append({"model": model_name, "target": "activity_set", "features": args.features, "band": band_name, "environment": env_name, "micro_f1_avg": env_results["micro_f1"]["avg"], "micro_f1_std": env_results["micro_f1"]["std"], "macro_f1_avg": env_results["macro_f1"]["avg"], "exact_set_accuracy_avg": env_results["exact_set_accuracy"]["avg"], "hamming_loss_avg": env_results["hamming_loss"]["avg"], "loss_avg": env_results["loss"]["avg"]})
            activity_rows = {name: [] for name in ACTIVITY_NAMES}
            for repeat in env_results["repeats"]:
                repeat_rows.append({"model": model_name, "features": args.features, "band": band_name, "environment": env_name, "seed": repeat["seed"], "best_epoch": repeat["best_epoch"], "micro_f1": repeat["micro_f1"], "macro_f1": repeat["macro_f1"], "exact_set_accuracy": repeat["exact_set_accuracy"], "hamming_loss": repeat["hamming_loss"], "loss": repeat["loss"], "no_predicted_activity_rate": repeat["activity_analysis"]["samples_with_no_predicted_activity"], "multi_predicted_activity_rate": repeat["activity_analysis"]["samples_with_multi_predicted_activity"], "model_path": repeat["model_path"]})
                for row in repeat["activity_analysis"]["per_activity"]:
                    per_activity_rows.append({"model": model_name, "features": args.features, "band": band_name, "environment": env_name, "seed": repeat["seed"], **row})
                    activity_rows[row["activity"]].append(row)
            cohort = [{"activity": k, "f1": float(np.mean([x["f1"] for x in v])), "recall": float(np.mean([x["recall"] for x in v]))} for k, v in activity_rows.items()]
            best, worst = max(cohort, key=lambda x: x["f1"]), min(cohort, key=lambda x: x["f1"])
            note = f"{model_name} detected {best['activity']} best and struggled most with {worst['activity']} for {band_name}/{env_name}."
            notes_rows.append({"model": model_name, "features": args.features, "band": band_name, "environment": env_name, "best_activity": best["activity"], "best_f1": best["f1"], "weakest_activity": worst["activity"], "weakest_f1": worst["f1"], "note": note})
            print(note)
    with pd.ExcelWriter(args.result_excel, engine="openpyxl") as writer:
        pd.DataFrame(summary_rows).to_excel(writer, sheet_name="summary", index=False)
        pd.DataFrame(repeat_rows).to_excel(writer, sheet_name="repeats", index=False)
        pd.DataFrame(per_activity_rows).to_excel(writer, sheet_name="per_activity", index=False)
        pd.DataFrame(notes_rows).to_excel(writer, sheet_name="cohort_notes", index=False)
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
                "micro_f1": summarize([r["micro_f1"] for r in repeats]),
                "macro_f1": summarize([r["macro_f1"] for r in repeats]),
                "exact_set_accuracy": summarize([r["exact_set_accuracy"] for r in repeats]),
                "hamming_loss": summarize([r["hamming_loss"] for r in repeats]),
                "loss": summarize([r["loss"] for r in repeats]),
                "repeats": repeats,
            }
    print_table(model_name, results)
    with open(args.result_json, "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "model": model_name, "target": "activity_set", "results": results}, f, indent=4)
    print(f"Saved JSON results to {args.result_json}")
    write_excel(model_name, args, results)
