import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, hamming_loss, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
WIMANS_WIFI_DIR = ROOT / "WiMANS-main" / "benchmark" / "wifi_csi"
sys.path.insert(0, str(WIMANS_WIFI_DIR))

from load_data import encode_data_y, load_data_y  # noqa: E402


ENVIRONMENTS = ["classroom", "meeting_room", "empty_room"]
ACTIVITY_NAMES = ["nothing", "walk", "rotation", "jump", "wave", "lie_down", "pick_up", "sit_down", "stand_up"]
FEATURE_DIRS = {
    "amp_top5": ROOT / "dataset" / "amp_top5",
    "amp_og": ROOT / "dataset" / "amp_OG",
}
MODEL_CLASSES = {
    "CLSTM": None,
    "THAT": None,
}


class CNN_LSTM(torch.nn.Module):
    def __init__(self, var_x_shape, var_y_shape):
        super().__init__()
        var_dim_input = var_x_shape[-1]
        var_dim_output = var_y_shape[-1]
        self.layer_norm = torch.nn.BatchNorm1d(var_dim_input)
        self.layer_norm_0 = torch.nn.BatchNorm1d(64)
        self.layer_norm_1 = torch.nn.BatchNorm1d(128)
        self.layer_norm_2 = torch.nn.BatchNorm1d(256)
        self.layer_cnn_1d_0 = torch.nn.Conv1d(var_dim_input, 64, kernel_size=128, stride=8)
        self.layer_cnn_1d_1 = torch.nn.Conv1d(64, 128, kernel_size=64, stride=4)
        self.layer_cnn_1d_2 = torch.nn.Conv1d(128, 256, kernel_size=32, stride=2)
        self.layer_lstm = torch.nn.LSTM(input_size=256, hidden_size=512, batch_first=True)
        self.layer_linear = torch.nn.Linear(512, var_dim_output)
        self.layer_dropout = torch.nn.Dropout(0.5)
        self.layer_leakyrelu = torch.nn.LeakyReLU()
        torch.nn.init.xavier_uniform_(self.layer_cnn_1d_0.weight)
        torch.nn.init.xavier_uniform_(self.layer_cnn_1d_1.weight)
        torch.nn.init.xavier_uniform_(self.layer_cnn_1d_2.weight)
        torch.nn.init.xavier_uniform_(self.layer_linear.weight)

    def forward(self, var_input):
        var_t = torch.permute(var_input, (0, 2, 1))
        var_t = self.layer_norm(var_t)
        var_t = self.layer_leakyrelu(self.layer_cnn_1d_0(var_t))
        var_t = self.layer_norm_0(var_t)
        var_t = self.layer_leakyrelu(self.layer_cnn_1d_1(var_t))
        var_t = self.layer_norm_1(var_t)
        var_t = self.layer_leakyrelu(self.layer_cnn_1d_2(var_t))
        var_t = self.layer_norm_2(var_t)
        var_t = torch.permute(var_t, (0, 2, 1))
        var_t, _ = self.layer_lstm(var_t)
        var_t = self.layer_dropout(var_t[:, -1, :])
        return self.layer_linear(var_t)


class Gaussian_Position(torch.nn.Module):
    def __init__(self, var_dim_feature, var_dim_time, var_num_gaussian=10):
        super().__init__()
        var_embedding = torch.zeros([var_num_gaussian, var_dim_feature], dtype=torch.float)
        self.var_embedding = torch.nn.Parameter(var_embedding, requires_grad=True)
        torch.nn.init.xavier_uniform_(self.var_embedding)
        var_position = torch.arange(0.0, var_dim_time).unsqueeze(1).repeat(1, var_num_gaussian)
        self.var_position = torch.nn.Parameter(var_position, requires_grad=False)
        var_mu = torch.arange(0.0, var_dim_time, var_dim_time / var_num_gaussian).unsqueeze(0)
        self.var_mu = torch.nn.Parameter(var_mu, requires_grad=True)
        var_sigma = torch.tensor([50.0] * var_num_gaussian).unsqueeze(0)
        self.var_sigma = torch.nn.Parameter(var_sigma, requires_grad=True)

    def forward(self, var_input):
        var_pdf = self.var_position - self.var_mu
        var_pdf = -var_pdf * var_pdf
        var_pdf = var_pdf / self.var_sigma / self.var_sigma / 2
        var_pdf = var_pdf - torch.log(self.var_sigma)
        var_pdf = torch.softmax(var_pdf, dim=-1)
        var_position_encoding = torch.matmul(var_pdf, self.var_embedding)
        return var_input + var_position_encoding.unsqueeze(0)


class Encoder(torch.nn.Module):
    def __init__(self, var_dim_feature, var_num_head=10, var_size_cnn=None):
        super().__init__()
        var_size_cnn = var_size_cnn or [1, 3, 5]
        self.layer_norm_0 = torch.nn.LayerNorm(var_dim_feature, eps=1e-6)
        self.layer_attention = torch.nn.MultiheadAttention(var_dim_feature, var_num_head, batch_first=True)
        self.layer_dropout_0 = torch.nn.Dropout(0.1)
        self.layer_norm_1 = torch.nn.LayerNorm(var_dim_feature, 1e-6)
        self.layer_cnn = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Conv1d(var_dim_feature, var_dim_feature, k, padding="same"),
                torch.nn.BatchNorm1d(var_dim_feature),
                torch.nn.Dropout(0.1),
                torch.nn.LeakyReLU(),
            )
            for k in var_size_cnn
        ])
        self.layer_dropout_1 = torch.nn.Dropout(0.1)

    def forward(self, var_input):
        var_t = self.layer_norm_0(var_input)
        var_t, _ = self.layer_attention(var_t, var_t, var_t)
        var_t = self.layer_dropout_0(var_t) + var_input
        var_s = torch.permute(self.layer_norm_1(var_t), (0, 2, 1))
        var_c = torch.stack([layer(var_s) for layer in self.layer_cnn], dim=0)
        var_s = torch.sum(var_c, dim=0) / len(self.layer_cnn)
        var_s = torch.permute(self.layer_dropout_1(var_s), (0, 2, 1))
        return var_s + var_t


class THAT(torch.nn.Module):
    def __init__(self, var_x_shape, var_y_shape):
        super().__init__()
        var_dim_feature = var_x_shape[-1]
        var_dim_time = var_x_shape[-2]
        var_dim_output = var_y_shape[-1]
        self.layer_left_pooling = torch.nn.AvgPool1d(kernel_size=20, stride=20)
        self.layer_left_gaussian = Gaussian_Position(var_dim_feature, var_dim_time // 20)
        self.layer_left_encoder = torch.nn.ModuleList([
            Encoder(var_dim_feature=var_dim_feature, var_num_head=10, var_size_cnn=[1, 3, 5])
            for _ in range(4)
        ])
        self.layer_left_norm = torch.nn.LayerNorm(var_dim_feature, eps=1e-6)
        self.layer_left_cnn_0 = torch.nn.Conv1d(var_dim_feature, 128, kernel_size=8)
        self.layer_left_cnn_1 = torch.nn.Conv1d(var_dim_feature, 128, kernel_size=16)
        self.layer_left_dropout = torch.nn.Dropout(0.5)
        self.layer_right_pooling = torch.nn.AvgPool1d(kernel_size=20, stride=20)
        var_dim_right = var_dim_time // 20
        self.layer_right_encoder = torch.nn.ModuleList([
            Encoder(var_dim_feature=var_dim_right, var_num_head=10, var_size_cnn=[1, 2, 3])
        ])
        self.layer_right_norm = torch.nn.LayerNorm(var_dim_right, eps=1e-6)
        self.layer_right_cnn_0 = torch.nn.Conv1d(var_dim_right, 16, kernel_size=2)
        self.layer_right_cnn_1 = torch.nn.Conv1d(var_dim_right, 16, kernel_size=4)
        self.layer_right_dropout = torch.nn.Dropout(0.5)
        self.layer_leakyrelu = torch.nn.LeakyReLU()
        self.layer_output = torch.nn.Linear(256 + 32, var_dim_output)

    def forward(self, var_input):
        var_left = torch.permute(var_input, (0, 2, 1))
        var_left = torch.permute(self.layer_left_pooling(var_left), (0, 2, 1))
        var_left = self.layer_left_gaussian(var_left)
        for layer in self.layer_left_encoder:
            var_left = layer(var_left)
        var_left = torch.permute(self.layer_left_norm(var_left), (0, 2, 1))
        var_left_0 = torch.sum(self.layer_leakyrelu(self.layer_left_cnn_0(var_left)), dim=-1)
        var_left_1 = torch.sum(self.layer_leakyrelu(self.layer_left_cnn_1(var_left)), dim=-1)
        var_left = self.layer_left_dropout(torch.concat([var_left_0, var_left_1], dim=-1))

        var_right = self.layer_right_pooling(torch.permute(var_input, (0, 2, 1)))
        for layer in self.layer_right_encoder:
            var_right = layer(var_right)
        var_right = torch.permute(self.layer_right_norm(var_right), (0, 2, 1))
        var_right_0 = torch.sum(self.layer_leakyrelu(self.layer_right_cnn_0(var_right)), dim=-1)
        var_right_1 = torch.sum(self.layer_leakyrelu(self.layer_right_cnn_1(var_right)), dim=-1)
        var_right = self.layer_right_dropout(torch.concat([var_right_0, var_right_1], dim=-1))
        return self.layer_output(torch.concat([var_left, var_right], dim=-1))


MODEL_CLASSES = {
    "CLSTM": CNN_LSTM,
    "THAT": THAT,
}


class AmplitudeActivityDataset(Dataset):
    def __init__(self, data_pd_y, data_dir, length=3000, normalize="none"):
        self.data_pd_y = data_pd_y.reset_index(drop=True)
        self.data_dir = Path(data_dir)
        self.length = length
        self.normalize = normalize
        self.file_ids = self.data_pd_y["label"].tolist()
        self.slot_activity = encode_data_y(self.data_pd_y, "activity").astype(np.float32)

    def __len__(self):
        return len(self.file_ids)

    def _pad_or_crop(self, x):
        if x.shape[0] < self.length:
            pad = [(self.length - x.shape[0], 0)] + [(0, 0)] * (x.ndim - 1)
            return np.pad(x, pad, mode="constant")
        if x.shape[0] > self.length:
            return x[-self.length:]
        return x

    def _normalize(self, x):
        if self.normalize in (None, "none"):
            return x
        if self.normalize == "standard":
            mean = x.mean(axis=0, keepdims=True)
            std = x.std(axis=0, keepdims=True)
            return (x - mean) / (std + 1e-6)
        if self.normalize == "log_standard":
            x = np.log1p(np.maximum(x, 0.0))
            mean = x.mean(axis=0, keepdims=True)
            std = x.std(axis=0, keepdims=True)
            return (x - mean) / (std + 1e-6)
        raise ValueError(f"Unknown normalize={self.normalize}")

    def __getitem__(self, idx):
        path = self.data_dir / f"{self.file_ids[idx]}.npy"
        x = np.load(path).astype(np.float32)
        x = self._pad_or_crop(x)
        x = self._normalize(x)
        x = x.reshape(x.shape[0], -1).astype(np.float32)
        y = self.slot_activity[idx].reshape(-1).astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(y)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train WiMANS CLSTM/THAT and evaluate with both WiMANS-style and our decomposed metrics."
    )
    parser.add_argument("--annotation", default=str(ROOT / "WiMANS-main" / "dataset" / "annotation.csv"))
    parser.add_argument("--features", choices=list(FEATURE_DIRS), default="amp_top5")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--model", choices=list(MODEL_CLASSES), default="CLSTM")
    parser.add_argument("--band", choices=["2.4", "5", "both", "all"], default="5")
    parser.add_argument("--environment", choices=ENVIRONMENTS + ["all"], default="all")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--seed-start", type=int, default=39)
    parser.add_argument("--split-seed", type=int, default=39)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--selection", choices=["wimans_test", "val"], default="wimans_test")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--pos-weight", type=float, default=None, help="Default: CLSTM=8, THAT=4, matching WiMANS code.")
    parser.add_argument("--length", type=int, default=3000)
    parser.add_argument("--normalize", choices=["none", "standard", "log_standard"], default="none")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile, as WiMANS does for CLSTM.")
    parser.add_argument("--save-dir", default=str(ROOT / "credibility" / "outputs" / "saved_wimans_models"))
    parser.add_argument("--out-dir", default=str(ROOT / "credibility" / "outputs"))
    parser.add_argument("--tag", default=None)
    return parser.parse_args()


def band_list(band):
    if band == "all":
        return [["2.4"], ["5"], ["2.4", "5"]]
    if band == "both":
        return [["2.4", "5"]]
    return [[band]]


def band_name(bands):
    return "both" if len(bands) > 1 else bands[0]


def split_data(data_pd_y, args):
    train_val_y, test_y = train_test_split(
        data_pd_y,
        test_size=args.test_size,
        shuffle=True,
        random_state=args.split_seed,
    )
    if args.selection == "val":
        train_y, val_y = train_test_split(
            train_val_y,
            test_size=args.val_size,
            shuffle=True,
            random_state=args.split_seed,
        )
        return train_y, val_y, test_y
    return train_val_y, test_y, test_y


def make_loader(dataset, args, shuffle):
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def infer_shapes(dataset):
    x, y = dataset[0]
    return tuple(x.shape), tuple(y.shape)


def make_model(args, x_shape, y_shape, device):
    model = MODEL_CLASSES[args.model](x_shape, y_shape).to(device)
    if args.compile:
        model = torch.compile(model)
    return model


def evaluate_logits(model, loader, device):
    model.eval()
    ys, logits = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            out = model(x)
            ys.append(y.cpu().numpy())
            logits.append(out.detach().cpu().numpy())
    return np.vstack(ys), np.vstack(logits)


def train_one(model, train_loader, select_loader, args, device):
    pos_weight_value = args.pos_weight
    if pos_weight_value is None:
        pos_weight_value = 8.0 if args.model == "CLSTM" else 4.0
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.full((54,), pos_weight_value, device=device))
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_score = -1.0
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        losses = []
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        y_select, logits_select = evaluate_logits(model, select_loader, device)
        pred_select = (1.0 / (1.0 + np.exp(-logits_select)) > args.threshold).astype(int)
        select_rows_true = y_select.reshape(-1, 6, 9).reshape(-1, 9)
        select_rows_pred = pred_select.reshape(-1, 6, 9).reshape(-1, 9)
        select_acc = accuracy_score(select_rows_true, select_rows_pred) * 100.0

        if select_acc > best_score:
            best_score = select_acc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

        print(
            f"    Epoch {epoch:03d}/{args.epochs} - {time.time() - t0:.1f}s "
            f"- Train Loss {np.mean(losses):.4f} - Select WiMANS Acc {select_acc:.2f}%"
        )

    model.load_state_dict(best_state)
    return model, best_epoch, best_score


def compute_metrics(y_true_flat, probs, threshold):
    slot_true = y_true_flat.astype(int)
    slot_pred = (probs > threshold).astype(int)
    true_user = slot_true.reshape(-1, 6, 9)
    pred_user = slot_pred.reshape(-1, 6, 9)
    true_user_rows = true_user.reshape(-1, 9)
    pred_user_rows = pred_user.reshape(-1, 9)
    true_occupancy = (true_user.sum(axis=2) > 0).astype(int)
    pred_occupancy = (pred_user.sum(axis=2) > 0).astype(int)
    true_activity_set = (true_user.sum(axis=1) > 0).astype(int)
    pred_activity_set = (pred_user.sum(axis=1) > 0).astype(int)
    active_mask = true_occupancy.reshape(-1) == 1
    empty_mask = ~active_mask

    return {
        "wimans_slot_activity_accuracy": accuracy_score(true_user_rows, pred_user_rows) * 100.0,
        "wimans_active_slot_activity_accuracy": accuracy_score(true_user_rows[active_mask], pred_user_rows[active_mask]) * 100.0,
        "wimans_empty_slot_accuracy": accuracy_score(true_user_rows[empty_mask], pred_user_rows[empty_mask]) * 100.0,
        "sample_exact_54_accuracy": accuracy_score(slot_true, slot_pred) * 100.0,
        "activity_set_exact_accuracy": accuracy_score(true_activity_set, pred_activity_set) * 100.0,
        "occupancy_exact_accuracy": accuracy_score(true_occupancy, pred_occupancy) * 100.0,
        "activity_set_micro_f1": f1_score(true_activity_set, pred_activity_set, average="micro", zero_division=0) * 100.0,
        "occupancy_micro_f1": f1_score(true_occupancy, pred_occupancy, average="micro", zero_division=0) * 100.0,
        "slot54_micro_f1": f1_score(slot_true, slot_pred, average="micro", zero_division=0) * 100.0,
        "slot54_macro_f1": f1_score(slot_true, slot_pred, average="macro", zero_division=0) * 100.0,
        "active_slot_micro_f1": f1_score(true_user_rows[active_mask], pred_user_rows[active_mask], average="micro", zero_division=0) * 100.0,
        "hamming54": hamming_loss(slot_true, slot_pred) * 100.0,
        "true_active_slots_per_sample": true_occupancy.sum(axis=1).mean(),
        "pred_active_slots_per_sample": pred_occupancy.sum(axis=1).mean(),
    }, slot_pred, pred_activity_set


def per_activity_rows(y_true_flat, slot_pred, metadata):
    slot_true = y_true_flat.astype(int)
    true_user = slot_true.reshape(-1, 6, 9)
    pred_user = slot_pred.reshape(-1, 6, 9)
    true_activity_set = (true_user.sum(axis=1) > 0).astype(int)
    pred_activity_set = (pred_user.sum(axis=1) > 0).astype(int)
    p, r, f, support = precision_recall_fscore_support(
        true_activity_set, pred_activity_set, average=None, zero_division=0
    )
    slot_p, slot_r, slot_f, slot_support = precision_recall_fscore_support(
        slot_true, slot_pred, average=None, zero_division=0
    )
    rows = []
    for idx, name in enumerate(ACTIVITY_NAMES):
        related = list(range(idx, 54, 9))
        rows.append({
            **metadata,
            "activity": name,
            "activity_set_precision": p[idx] * 100.0,
            "activity_set_recall": r[idx] * 100.0,
            "activity_set_f1": f[idx] * 100.0,
            "activity_set_support": int(support[idx]),
            "slot54_mean_f1_for_activity": float(np.mean(slot_f[related]) * 100.0),
            "slot54_total_support_for_activity": int(np.sum(slot_support[related])),
        })
    return rows


def summarize(values):
    return {"avg": float(np.mean(values)), "std": float(np.std(values))}


def main():
    args = parse_args()
    data_dir = Path(args.data_dir) if args.data_dir else FEATURE_DIRS[args.features]
    envs = ENVIRONMENTS if args.environment == "all" else [args.environment]
    out_dir = Path(args.out_dir)
    save_dir = Path(args.save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or f"wimans_{args.model.lower()}_{args.features}_band{args.band}_{args.selection}"
    device = torch.device(args.device)

    summary_rows = []
    per_activity = []
    for bands in band_list(args.band):
        bname = band_name(bands)
        for env in envs:
            data_pd_y = load_data_y(args.annotation, [env], bands, ["0", "1", "2", "3", "4", "5"])
            train_y, select_y, test_y = split_data(data_pd_y, args)
            train_dataset = AmplitudeActivityDataset(train_y, data_dir, length=args.length, normalize=args.normalize)
            select_dataset = AmplitudeActivityDataset(select_y, data_dir, length=args.length, normalize=args.normalize)
            test_dataset = AmplitudeActivityDataset(test_y, data_dir, length=args.length, normalize=args.normalize)
            x_shape, y_shape = infer_shapes(train_dataset)
            train_loader = make_loader(train_dataset, args, shuffle=True)
            select_loader = make_loader(select_dataset, args, shuffle=False)
            test_loader = make_loader(test_dataset, args, shuffle=False)

            print(
                f"\n[*] WiMANS {args.model} Features={args.features} Env={env} Band={bname} "
                f"Split train/select/test={len(train_dataset)}/{len(select_dataset)}/{len(test_dataset)} "
                f"Selection={args.selection}"
            )
            for repeat_idx in range(args.repeat):
                seed = args.seed_start + repeat_idx
                np.random.seed(seed)
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                model = make_model(args, x_shape, y_shape, device)
                model, best_epoch, best_select_score = train_one(model, train_loader, select_loader, args, device)
                y_test, logits_test = evaluate_logits(model, test_loader, device)
                probs = 1.0 / (1.0 + np.exp(-logits_test))
                metrics, slot_pred, _ = compute_metrics(y_test, probs, args.threshold)
                ckpt_path = save_dir / f"wimans_{args.model.lower()}_{args.features}_{env}_{bname}_seed{seed}.pth"
                torch.save(model.state_dict(), ckpt_path)
                metadata = {
                    "model": args.model,
                    "features": args.features,
                    "band": bname,
                    "environment": env,
                    "seed": seed,
                    "selection": args.selection,
                    "best_epoch": best_epoch,
                    "best_select_wimans_accuracy": best_select_score,
                    "threshold": args.threshold,
                    "checkpoint": str(ckpt_path),
                }
                row = {**metadata, **metrics}
                summary_rows.append(row)
                per_activity.extend(per_activity_rows(y_test, slot_pred, metadata))
                print(
                    f"    Final Test - Best Epoch {best_epoch} "
                    f"- WiMANS Acc {metrics['wimans_slot_activity_accuracy']:.2f}% "
                    f"- ActSet F1 {metrics['activity_set_micro_f1']:.2f}% "
                    f"- Occ F1 {metrics['occupancy_micro_f1']:.2f}% "
                    f"- ActiveSlot F1 {metrics['active_slot_micro_f1']:.2f}% "
                    f"- 54-F1 {metrics['slot54_micro_f1']:.2f}%"
                )

    summary_df = pd.DataFrame(summary_rows)
    per_activity_df = pd.DataFrame(per_activity)
    metric_cols = [
        "wimans_slot_activity_accuracy",
        "wimans_active_slot_activity_accuracy",
        "wimans_empty_slot_accuracy",
        "activity_set_micro_f1",
        "occupancy_micro_f1",
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

    base = out_dir / tag
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
    print("Credibility B: WiMANS Model Evaluated With Our Metrics")
    print("================================================================================================================")
    print(grouped_df[[
        "band",
        "environment",
        "wimans_slot_activity_accuracy_text",
        "activity_set_micro_f1_text",
        "occupancy_micro_f1_text",
        "active_slot_micro_f1_text",
        "slot54_micro_f1_text",
    ]].to_string(index=False))
    print("================================================================================================================")
    print(f"Saved Excel: {excel_path}")
    print(f"Saved CSV: {summary_path}")
    print(f"Saved JSON: {json_path}")


if __name__ == "__main__":
    main()
