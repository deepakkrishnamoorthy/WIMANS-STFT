import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MULTI_HEAD_DIR = os.path.join(ROOT, "multi_head_experiments")
if MULTI_HEAD_DIR not in sys.path:
    sys.path.insert(0, MULTI_HEAD_DIR)

from dataset_multi_head import MultiHeadSTFTDataset
from train_multi_head_top5 import (
    ACTIVITY_NAMES,
    BAND_GROUPS,
    ENVIRONMENTS,
    FEATURE_DIRS,
    MODEL_FACTORIES,
    SELECT_METRICS,
    apply_mixup,
    batch_to_device,
    collect_outputs,
    combined_loss,
    compute_metrics,
    format_threshold,
    infer_input_channels,
    make_criteria,
    make_default_thresholds,
    print_table,
    set_seed,
    split_data,
    summarize,
    tune_threshold,
    write_excel,
)

sys.path.append(os.path.join(ROOT, "WiMANS-main", "benchmark", "wifi_csi"))
from load_data import load_data_y


def parse_args():
    parser = argparse.ArgumentParser(
        description="Curriculum multi-head Top-5 STFT experiment with comparable final holdout testing."
    )
    parser.add_argument("--annotation", default=r"D:\Deepak\wifi_csi\WiMANS-main\dataset\annotation.csv")
    parser.add_argument("--features", choices=["multichannel", "pca"], default="multichannel")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--model", choices=list(MODEL_FACTORIES), default="clstm")
    parser.add_argument("--environment", choices=["all", *ENVIRONMENTS], default="all")
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

    parser.add_argument("--stage1-epochs", type=int, default=10)
    parser.add_argument("--stage2-epochs", type=int, default=20)
    parser.add_argument("--stage1-max-active", type=int, default=2)
    parser.add_argument("--stage2-max-active", type=int, default=3)
    parser.add_argument("--min-stage-samples", type=int, default=128)
    parser.add_argument("--stage1-loss-activity", type=float, default=1.0)
    parser.add_argument("--stage1-loss-occupancy", type=float, default=1.0)
    parser.add_argument("--stage1-loss-slot", type=float, default=0.2)
    parser.add_argument("--stage2-loss-activity", type=float, default=0.8)
    parser.add_argument("--stage2-loss-occupancy", type=float, default=0.8)
    parser.add_argument("--stage2-loss-slot", type=float, default=0.7)

    args = parser.parse_args()
    out_dir = os.path.join(ROOT, "curriculum_experiments", "outputs")
    tag = f"curriculum_{args.model}_{args.features}_band{args.band}".replace(".", "p")
    args.save_dir = args.save_dir or os.path.join(out_dir, f"saved_models_{tag}")
    args.result_json = args.result_json or os.path.join(out_dir, f"{tag}_results.json")
    args.result_excel = args.result_excel or os.path.join(out_dir, f"{tag}_analysis.xlsx")
    return args


def make_loader(dataset, args, shuffle):
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def density_subset(dataset, max_active_users, min_samples):
    if max_active_users is None:
        return dataset, len(dataset), "all"
    active_counts = dataset.occupancy.sum(axis=1)
    indices = np.where(active_counts <= max_active_users)[0].tolist()
    if len(indices) < min_samples:
        return dataset, len(dataset), f"all_fallback_for_le_{max_active_users}"
    return Subset(dataset, indices), len(indices), f"active_users_le_{max_active_users}"


def stage_args(args, stage):
    staged = copy.copy(args)
    if stage == 1:
        staged.loss_activity_set = args.stage1_loss_activity
        staged.loss_occupancy = args.stage1_loss_occupancy
        staged.loss_slot_activity = args.stage1_loss_slot
    elif stage == 2:
        staged.loss_activity_set = args.stage2_loss_activity
        staged.loss_occupancy = args.stage2_loss_occupancy
        staged.loss_slot_activity = args.stage2_loss_slot
    else:
        staged.loss_activity_set = args.loss_activity_set
        staged.loss_occupancy = args.loss_occupancy
        staged.loss_slot_activity = args.loss_slot_activity
    return staged


def build_curriculum_stages(args, train_dataset):
    stage1_epochs = min(args.stage1_epochs, args.epochs)
    stage2_epochs = min(args.stage2_epochs, max(0, args.epochs - stage1_epochs))
    stage3_epochs = max(0, args.epochs - stage1_epochs - stage2_epochs)
    raw = [
        (1, stage1_epochs, args.stage1_max_active),
        (2, stage2_epochs, args.stage2_max_active),
        (3, stage3_epochs, None),
    ]
    stages = []
    for stage_id, epochs, max_active in raw:
        if epochs <= 0:
            continue
        subset, sample_count, density_rule = density_subset(train_dataset, max_active, args.min_stage_samples)
        stages.append({
            "stage": stage_id,
            "epochs": epochs,
            "dataset": subset,
            "sample_count": sample_count,
            "density_rule": density_rule,
            "args": stage_args(args, stage_id),
        })
    return stages


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

    val_loader = make_loader(val_dataset, args, shuffle=False)
    test_loader = make_loader(test_dataset, args, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MODEL_FACTORIES[args.model](input_channels).to(device)
    criteria = make_criteria(train_dataset, args, device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    stages = build_curriculum_stages(args, train_dataset)

    best_score = -1.0
    best_epoch = 0
    best_threshold = make_default_thresholds(args)
    best_model_wts = copy.deepcopy(model.state_dict())
    epoch_history = []
    global_epoch = 0

    print(f"\n[*] Curriculum MultiHead Model={args.model} Features={args.features} Env={env_name} Band={band_name} Seed={seed}")
    print(
        f"    Channels={input_channels} Split train/val/test={len(train_dataset)}/{len(val_dataset)}/{len(test_dataset)}"
        f" Select={args.select_metric} Tune={args.tune_threshold} ThresholdStrategy={args.threshold_strategy}"
        f" OccupancyGate={args.occupancy_gate} GatePower={args.occupancy_gate_power}"
    )
    for stage in stages:
        sargs = stage["args"]
        print(
            f"    Stage {stage['stage']}: epochs={stage['epochs']} samples={stage['sample_count']}"
            f" density={stage['density_rule']}"
            f" loss(activity/occupancy/slot)={sargs.loss_activity_set}/{sargs.loss_occupancy}/{sargs.loss_slot_activity}"
        )
    if args.augment or args.mixup_alpha > 0.0:
        print(
            f"    Augment={args.augment} TimeMask={args.time_mask_width} FreqMask={args.freq_mask_width}"
            f" ChannelDrop={args.channel_drop_prob} NoiseStd={args.noise_std} TimeShift={args.time_shift}"
            f" MixupAlpha={args.mixup_alpha}"
        )

    for stage in stages:
        stage_loader = make_loader(stage["dataset"], args, shuffle=True)
        sargs = stage["args"]
        for stage_epoch in range(stage["epochs"]):
            global_epoch += 1
            start = time.time()
            model.train()
            train_loss = 0.0
            for batch in stage_loader:
                batch = batch_to_device(batch, device)
                batch = apply_mixup(batch, args.mixup_alpha)
                optimizer.zero_grad()
                outputs = model(batch["x"])
                loss = combined_loss(outputs, batch, criteria, sargs)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * batch["x"].size(0)
            train_loss /= len(stage_loader.dataset)

            val_labels, val_probs, val_loss = collect_outputs(model, val_loader, criteria, args, device)
            threshold = tune_threshold(val_labels, val_probs, args)
            val_metrics = compute_metrics(val_labels, val_probs, threshold, args.occupancy_gate, args.occupancy_gate_power)
            score = val_metrics[args.select_metric]
            if score > best_score:
                best_score = score
                best_epoch = global_epoch
                best_threshold = threshold
                best_model_wts = copy.deepcopy(model.state_dict())

            epoch_history.append({
                "epoch": global_epoch,
                "stage": stage["stage"],
                "stage_epoch": stage_epoch + 1,
                "stage_density_rule": stage["density_rule"],
                "stage_sample_count": stage["sample_count"],
                "loss_activity_set_weight": sargs.loss_activity_set,
                "loss_occupancy_weight": sargs.loss_occupancy,
                "loss_slot_activity_weight": sargs.loss_slot_activity,
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
                f"    Epoch {global_epoch:03d}/{args.epochs} S{stage['stage']} - {time.time() - start:.1f}s"
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
    model_path = os.path.join(args.save_dir, f"curriculum_{args.model}_{args.features}_{env_name}_{band_name}_seed{seed}.pth")
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
        "curriculum": "task_density",
    })
    return test_metrics


def main():
    args = parse_args()
    args.data_dir = args.data_dir or FEATURE_DIRS[args.features]
    input_channels = infer_input_channels(args.data_dir)
    selected_bands = BAND_GROUPS if args.band == "all" else {args.band: BAND_GROUPS[args.band]}
    selected_environments = ENVIRONMENTS if args.environment == "all" else [args.environment]

    results = {}
    for band_name, wifi_bands in selected_bands.items():
        results[band_name] = {}
        for env_name in selected_environments:
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
