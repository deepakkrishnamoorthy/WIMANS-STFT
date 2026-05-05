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
from sklearn.metrics import accuracy_score, f1_score, hamming_loss
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

# Add WiMANS-main to path so the baseline uses the same annotation filtering.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'WiMANS-main', 'benchmark', 'wifi_csi')))
from load_data import load_data_y

from dataset import STFTDataset
from model import ResNetBaseline


ENVIRONMENTS = ["classroom", "meeting_room", "empty_room"]
BAND_GROUPS = {
    "2.4": ["2.4"],
    "5": ["5"],
    "both": ["2.4", "5"],
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run ResNet18 on WiMANS Top-5 STFT activity labels.")
    parser.add_argument("--annotation", default=r"D:\Deepak\wifi_csi\WiMANS-main\dataset\annotation.csv")
    parser.add_argument("--data-dir", default=r"D:\Deepak\wifi_csi\dataset\stft_top5_npy")
    parser.add_argument("--band", choices=["2.4", "5", "both", "all"], default="all",
                        help="WiFi band subset. 'all' runs 2.4, 5, and combined.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--repeat", type=int, default=3,
                        help="Number of model random seeds. WiMANS-style split remains fixed at random_state=39.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--normalize", choices=["log_standard", "log", "standard", "none"], default="log_standard")
    parser.add_argument("--split-seed", type=int, default=39)
    parser.add_argument("--seed-start", type=int, default=39)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--save-dir", default=os.path.join(os.path.dirname(__file__), "saved_models"))
    parser.add_argument("--result-json", default=os.path.join(os.path.dirname(__file__), "resnet_activity_results.json"))
    return parser.parse_args()


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def activity_accuracy(labels, preds):
    """Match WiMANS evaluation: flatten sample/user rows, then score 9-activity vectors."""
    labels_user = labels.reshape(-1, 9).astype(int)
    preds_user = preds.reshape(-1, 9).astype(int)
    return accuracy_score(labels_user, preds_user) * 100.0


def calculate_pos_weight(labels, clamp_max=50.0):
    labels_2d = labels.reshape(labels.shape[0], -1).astype(np.float32)
    pos = labels_2d.sum(axis=0)
    neg = labels_2d.shape[0] - pos
    pos_weight = neg / np.maximum(pos, 1.0)
    return torch.tensor(np.clip(pos_weight, 1.0, clamp_max), dtype=torch.float32)


def evaluate(model, loader, criterion, device, threshold):
    model.eval()
    test_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)

            preds = (torch.sigmoid(outputs) > threshold).float().cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    avg_loss = test_loss / len(loader.dataset)

    return {
        "loss": avg_loss,
        "activity_accuracy": activity_accuracy(all_labels, all_preds),
        "exact_54_accuracy": accuracy_score(all_labels.astype(int), all_preds.astype(int)) * 100.0,
        "micro_f1": f1_score(all_labels.astype(int), all_preds.astype(int), average="micro", zero_division=0) * 100.0,
        "macro_f1": f1_score(all_labels.astype(int), all_preds.astype(int), average="macro", zero_division=0) * 100.0,
        "hamming_loss": hamming_loss(all_labels.astype(int), all_preds.astype(int)) * 100.0,
    }


def train_once(args, env_name, band_name, wifi_bands, repeat_idx):
    seed = args.seed_start + repeat_idx
    set_seed(seed)

    data_pd_y = load_data_y(
        args.annotation,
        var_environment=[env_name],
        var_wifi_band=wifi_bands,
        var_num_users=["0", "1", "2", "3", "4", "5"],
    )

    data_train_y, data_test_y = train_test_split(
        data_pd_y,
        test_size=0.2,
        shuffle=True,
        random_state=args.split_seed,
    )

    train_dataset = STFTDataset(data_train_y, args.data_dir, max_len=200, normalize=args.normalize)
    test_dataset = STFTDataset(data_test_y, args.data_dir, max_len=200, normalize=args.normalize)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNetBaseline(num_classes=54).to(device)

    pos_weight = calculate_pos_weight(train_dataset.labels).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_metric = -1.0
    best_result = None
    best_model_wts = copy.deepcopy(model.state_dict())

    print(f"\n[*] Env={env_name} Band={band_name} Repeat={repeat_idx + 1}/{args.repeat} Seed={seed}")
    print(f"    Samples train/test: {len(train_dataset)}/{len(test_dataset)} | normalize={args.normalize} | lr={args.lr}")
    print(f"    Pos weight median={float(pos_weight.median().cpu()):.2f}, min={float(pos_weight.min().cpu()):.2f}, max={float(pos_weight.max().cpu()):.2f}")

    for epoch in range(args.epochs):
        start = time.time()
        model.train()
        train_loss = 0.0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)

        train_loss /= len(train_loader.dataset)
        metrics = evaluate(model, test_loader, criterion, device, args.threshold)

        if metrics["activity_accuracy"] > best_metric:
            best_metric = metrics["activity_accuracy"]
            best_result = metrics
            best_model_wts = copy.deepcopy(model.state_dict())

        print(
            f"    Epoch {epoch + 1:03d}/{args.epochs}"
            f" - {time.time() - start:.1f}s"
            f" - Train Loss {train_loss:.4f}"
            f" - Test Loss {metrics['loss']:.4f}"
            f" - Activity Acc {metrics['activity_accuracy']:.2f}%"
            f" - Micro F1 {metrics['micro_f1']:.2f}%"
        )

    os.makedirs(args.save_dir, exist_ok=True)
    model_path = os.path.join(args.save_dir, f"resnet_{env_name}_{band_name}_seed{seed}.pth")
    torch.save(best_model_wts, model_path)
    best_result = dict(best_result)
    best_result["seed"] = seed
    best_result["model_path"] = model_path
    return best_result


def summarize(values):
    values = np.array(values, dtype=np.float64)
    return {
        "avg": float(values.mean()),
        "std": float(values.std()),
        "text": f"{values.mean():.2f}+/-{values.std():.2f}",
    }


def print_table(results):
    print("\n" + "=" * 84)
    print("ResNet18 Top-5 STFT Activity Results")
    print("=" * 84)
    print(f"{'Band':<12} | {'Classroom':<18} | {'Meeting Room':<18} | {'Empty Room':<18}")
    print("-" * 84)
    for band_name in results:
        row = [band_name]
        for env_name in ENVIRONMENTS:
            row.append(results[band_name][env_name]["activity_accuracy"]["text"])
        print(f"{row[0]:<12} | {row[1]:<18} | {row[2]:<18} | {row[3]:<18}")
    print("=" * 84)
    print("Metric in table: WiMANS-style activity accuracy, averaged over 6 user activity vectors per sample.")


def main():
    args = parse_args()
    selected_bands = BAND_GROUPS if args.band == "all" else {args.band: BAND_GROUPS[args.band]}
    results = {}

    for band_name, wifi_bands in selected_bands.items():
        results[band_name] = {}
        for env_name in ENVIRONMENTS:
            repeat_results = [
                train_once(args, env_name, band_name, wifi_bands, repeat_idx)
                for repeat_idx in range(args.repeat)
            ]
            results[band_name][env_name] = {
                "activity_accuracy": summarize([r["activity_accuracy"] for r in repeat_results]),
                "exact_54_accuracy": summarize([r["exact_54_accuracy"] for r in repeat_results]),
                "micro_f1": summarize([r["micro_f1"] for r in repeat_results]),
                "macro_f1": summarize([r["macro_f1"] for r in repeat_results]),
                "hamming_loss": summarize([r["hamming_loss"] for r in repeat_results]),
                "loss": summarize([r["loss"] for r in repeat_results]),
                "repeats": repeat_results,
            }

    print_table(results)

    with open(args.result_json, "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "results": results}, f, indent=4)
    print(f"\nSaved JSON results to {args.result_json}")


if __name__ == "__main__":
    main()
