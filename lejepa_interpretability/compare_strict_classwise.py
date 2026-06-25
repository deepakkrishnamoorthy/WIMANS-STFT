import argparse
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ACTIVITY_NAMES = ["nothing", "walk", "rotation", "jump", "wave", "lie_down", "pick_up", "sit_down", "stand_up"]
ENVIRONMENTS = ["classroom", "meeting_room", "empty_room"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Class-wise comparison between augmented multi-head CLSTM and strict-holdout LeJEPA fine-tuning."
    )
    parser.add_argument(
        "--baseline-json",
        default=r"D:\Deepak\wifi_csi\multi_head_experiments\outputs\multi_head_clstm_multichannel_band5_augmix_slotpos0p5_results.json",
    )
    parser.add_argument(
        "--lejepa-json",
        default=r"D:\Deepak\wifi_csi\lejepa_experiments\strict_holdout\outputs\finetune_clstm_band5_strict_results.json",
    )
    parser.add_argument("--band", default="5")
    parser.add_argument("--include-nothing", action="store_true")
    parser.add_argument(
        "--out-dir",
        default=r"D:\Deepak\wifi_csi\lejepa_interpretability\outputs\strict_classwise_comparison",
    )
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def collect_per_activity(data, band, model_name):
    rows = []
    for env in ENVIRONMENTS:
        repeats = data["results"][band][env]["repeats"]
        for repeat in repeats:
            for activity_row in repeat["per_activity"]:
                rows.append({
                    "model": model_name,
                    "environment": env,
                    "seed": repeat["seed"],
                    "activity": activity_row["activity"],
                    "direct_precision": activity_row["direct_activity_precision"],
                    "direct_recall": activity_row["direct_activity_recall"],
                    "direct_f1": activity_row["direct_activity_f1"],
                    "slot_precision": activity_row["slot_collapsed_precision"],
                    "slot_recall": activity_row["slot_collapsed_recall"],
                    "slot_f1": activity_row["slot_collapsed_f1"],
                    "support": activity_row["activity_set_support"],
                    "direct_mean_probability": activity_row["direct_mean_probability"],
                    "slot_mean_probability": activity_row["slot_mean_probability"],
                })
    return pd.DataFrame(rows)


def aggregate(df):
    metric_cols = [
        "direct_precision",
        "direct_recall",
        "direct_f1",
        "slot_precision",
        "slot_recall",
        "slot_f1",
        "support",
        "direct_mean_probability",
        "slot_mean_probability",
    ]
    grouped = (
        df.groupby(["model", "environment", "activity"], as_index=False)[metric_cols]
        .mean()
    )
    return grouped


def make_comparison(agg):
    base = agg[agg["model"] == "baseline"].drop(columns=["model"])
    lej = agg[agg["model"] == "strict_lejepa"].drop(columns=["model"])
    merged = base.merge(
        lej,
        on=["environment", "activity"],
        suffixes=("_baseline", "_strict_lejepa"),
    )
    for metric in ["direct_precision", "direct_recall", "direct_f1", "slot_precision", "slot_recall", "slot_f1"]:
        merged[f"{metric}_delta"] = merged[f"{metric}_strict_lejepa"] - merged[f"{metric}_baseline"]
    return merged


def print_summary(comp):
    print("\nClass-wise direct activity F1, excluding nothing unless requested")
    table = comp.pivot(index="activity", columns="environment", values="direct_f1_delta")
    table = table[[env for env in ENVIRONMENTS if env in table.columns]]
    print("Delta = strict LeJEPA - baseline multi-head CLSTM")
    print(table.round(2).to_string())

    print("\nLargest direct activity F1 gains")
    gains = comp.sort_values("direct_f1_delta", ascending=False).head(8)
    print(gains[["environment", "activity", "direct_f1_baseline", "direct_f1_strict_lejepa", "direct_f1_delta"]].round(2).to_string(index=False))

    print("\nLargest direct activity F1 drops")
    drops = comp.sort_values("direct_f1_delta", ascending=True).head(8)
    print(drops[["environment", "activity", "direct_f1_baseline", "direct_f1_strict_lejepa", "direct_f1_delta"]].round(2).to_string(index=False))

    slot = comp.pivot(index="activity", columns="environment", values="slot_f1_delta")
    slot = slot[[env for env in ENVIRONMENTS if env in slot.columns]]
    print("\nSlot-collapsed activity F1 delta")
    print(slot.round(2).to_string())


def plot_heatmap(comp, value_col, title, out_path):
    pivot = comp.pivot(index="activity", columns="environment", values=value_col)
    pivot = pivot.reindex([a for a in ACTIVITY_NAMES if a in pivot.index])
    pivot = pivot[[env for env in ENVIRONMENTS if env in pivot.columns]]
    values = pivot.values
    vmax = max(1e-6, abs(pd.Series(values.ravel()).dropna()).max())
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    im = ax.imshow(values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([c.replace("_", " ").title() for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title(title)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, f"{values[i, j]:+.1f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="F1 delta")
    fig.tight_layout()
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    baseline = collect_per_activity(load_json(args.baseline_json), args.band, "baseline")
    lejepa = collect_per_activity(load_json(args.lejepa_json), args.band, "strict_lejepa")
    combined = pd.concat([baseline, lejepa], ignore_index=True)
    if not args.include_nothing:
        combined = combined[combined["activity"] != "nothing"].reset_index(drop=True)

    agg = aggregate(combined)
    comp = make_comparison(agg)
    combined.to_csv(os.path.join(args.out_dir, "classwise_repeat_rows.csv"), index=False)
    agg.to_csv(os.path.join(args.out_dir, "classwise_average_rows.csv"), index=False)
    comp.to_csv(os.path.join(args.out_dir, "strict_lejepa_vs_baseline_classwise.csv"), index=False)

    plot_heatmap(
        comp,
        "direct_f1_delta",
        "Direct activity-head F1 delta: strict LeJEPA - baseline",
        os.path.join(args.out_dir, "direct_activity_f1_delta_heatmap.png"),
    )
    plot_heatmap(
        comp,
        "slot_f1_delta",
        "Slot-collapsed activity F1 delta: strict LeJEPA - baseline",
        os.path.join(args.out_dir, "slot_activity_f1_delta_heatmap.png"),
    )
    print_summary(comp)
    print(f"\nSaved class-wise comparison outputs to: {args.out_dir}")


if __name__ == "__main__":
    main()
