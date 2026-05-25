import os
import re
import sys
import argparse
import time
from functools import lru_cache
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import plotly.express as px
import torch
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
MULTI_HEAD_DIR = ROOT / "multi_head_experiments"
WIMANS_CSI_DIR = ROOT / "WiMANS-main" / "benchmark" / "wifi_csi"
for path in (MULTI_HEAD_DIR, WIMANS_CSI_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from model_multi_head import CLSTMMultiHead, ResNet18MultiHead, THATStyleMultiHead


ACTIVITY_NAMES = ["nothing", "walk", "rotation", "jump", "wave", "lie_down", "pick_up", "sit_down", "stand_up"]
ENVIRONMENTS = ["classroom", "meeting_room", "empty_room"]
USER_NAMES = [f"user_{idx}" for idx in range(1, 7)]
FEATURE_DIR = ROOT / "dataset" / "stft_top5_multichannel_npy"
ANNOTATION_PATH = ROOT / "WiMANS-main" / "dataset" / "annotation.csv"
DEFAULT_WEIGHTS_DIR = ROOT / "multi_head_experiments" / "outputs" / "saved_models_clstm_augmix_slotpos0p5"
DEFAULT_ANALYSIS = ROOT / "multi_head_experiments" / "outputs" / "multi_head_clstm_multichannel_band5_augmix_slotpos0p5_analysis.xlsx"
WEIGHT_RE = re.compile(
    r"^multi_head_(?P<model>resnet18|clstm|that_style)_(?P<features>multichannel|pca)_"
    r"(?P<environment>classroom|meeting_room|empty_room)_(?P<band>2\.4|5|both)_seed(?P<seed>\d+)\.pth$"
)
MODEL_FACTORIES = {
    "resnet18": lambda input_channels: ResNet18MultiHead(input_channels=input_channels),
    "clstm": lambda input_channels: CLSTMMultiHead(input_channels=input_channels),
    "that_style": lambda input_channels: THATStyleMultiHead(input_channels=input_channels),
}


@lru_cache(maxsize=1)
def annotation_df():
    return pd.read_csv(ANNOTATION_PATH)


@lru_cache(maxsize=1)
def thresholds():
    if not DEFAULT_ANALYSIS.exists():
        return {}
    try:
        repeats = pd.read_excel(DEFAULT_ANALYSIS, sheet_name="repeats")
    except Exception:
        return {}
    values = {}
    for _, row in repeats.iterrows():
        values[(str(row["environment"]), int(row["seed"]))] = float(row.get("threshold", 0.5))
    return values


@lru_cache(maxsize=1)
def weights_index():
    rows = []
    for path in sorted(DEFAULT_WEIGHTS_DIR.glob("*.pth")):
        match = WEIGHT_RE.match(path.name)
        if not match:
            continue
        row = match.groupdict()
        row["seed"] = int(row["seed"])
        row["path"] = str(path)
        rows.append(row)
    return rows


def find_weight(environment, seed):
    matches = [row for row in weights_index() if row["environment"] == environment and row["seed"] == int(seed)]
    if not matches:
        raise FileNotFoundError(f"No checkpoint found for environment={environment}, seed={seed}")
    return matches[0]


@lru_cache(maxsize=8)
def load_model(model_name, weight_path, input_channels, device_name):
    device = torch.device(device_name)
    model = MODEL_FACTORIES[model_name](input_channels)
    state = torch.load(weight_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def split_holdout(environment, band="5", split_seed=39, test_size=0.2, val_size=0.2):
    df = annotation_df()
    band_value = float(band)
    scoped = df[(df["environment"] == environment) & (df["wifi_band"].astype(float) == band_value)].copy()
    train_val, test = train_test_split(scoped, test_size=test_size, shuffle=True, random_state=split_seed)
    train_test_split(train_val, test_size=val_size, shuffle=True, random_state=split_seed)
    return test.sort_values("label")


def holdout_labels(environment):
    holdout = split_holdout(environment)
    return [label for label in holdout["label"].tolist() if (FEATURE_DIR / f"{label}.npy").exists()]


def update_sample_choices(environment):
    labels = holdout_labels(environment)
    return gr.Dropdown(choices=labels, value=labels[0] if labels else None)


def default_threshold(environment, seed):
    return thresholds().get((environment, int(seed)), 0.5)


def update_threshold(environment, seed):
    return gr.Slider(value=float(default_threshold(environment, seed)))


def pad_or_crop(spec, max_len=200):
    time_len = spec.shape[-1]
    if time_len < max_len:
        pad_width = [(0, 0)] * spec.ndim
        pad_width[-1] = (0, max_len - time_len)
        return np.pad(spec, pad_width, mode="constant")
    if time_len > max_len:
        return spec[..., :max_len]
    return spec


def normalize_spec(spec, mode="log_standard"):
    spec = spec.astype(np.float32)
    if mode == "log_standard":
        spec = np.log1p(spec)
        mean = spec.mean(axis=(-2, -1), keepdims=True)
        std = spec.std(axis=(-2, -1), keepdims=True)
        return (spec - mean) / (std + 1e-6)
    if mode == "log":
        return np.log1p(spec)
    if mode == "standard":
        mean = spec.mean(axis=(-2, -1), keepdims=True)
        std = spec.std(axis=(-2, -1), keepdims=True)
        return (spec - mean) / (std + 1e-6)
    return spec


def prepare_spec(spec, normalize="log_standard", max_len=200):
    if spec.ndim == 2:
        spec = spec[np.newaxis, ...]
    if spec.ndim != 3:
        raise ValueError(f"Expected shape (channels, freq, time) or (freq, time), got {spec.shape}")
    raw = pad_or_crop(spec.astype(np.float32), max_len=max_len)
    model_input = normalize_spec(raw, normalize)
    return raw, model_input.astype(np.float32)


def apply_occupancy_gate(slot_probs, occupancy_probs, mode, threshold, power):
    slot = slot_probs.copy().reshape(6, 9)
    if mode == "binary":
        slot[occupancy_probs <= threshold, :] = 0.0
    elif mode == "prob":
        slot = slot * np.power(np.clip(occupancy_probs.reshape(6, 1), 0.0, 1.0), power)
    return slot


def annotation_for_label(label):
    rows = annotation_df()[annotation_df()["label"] == label]
    if rows.empty:
        return None
    return rows.iloc[0].to_dict()


def ground_truth(row):
    if row is None:
        return [], pd.DataFrame(columns=["user_slot", "true_active", "true_activity", "true_location"])
    rows = []
    scene = []
    for idx in range(1, 7):
        activity = row.get(f"user_{idx}_activity")
        location = row.get(f"user_{idx}_location")
        active = isinstance(activity, str) and activity in ACTIVITY_NAMES
        if active:
            scene.append(activity)
        rows.append({
            "user_slot": f"user_{idx}",
            "true_active": active,
            "true_activity": activity if active else "-",
            "true_location": "-" if pd.isna(location) else location,
        })
    scene = sorted(set(scene), key=ACTIVITY_NAMES.index)
    return scene, pd.DataFrame(rows)


def bar_plot(names, values, title, threshold):
    df = pd.DataFrame({"label": names, "probability": values})
    fig = px.bar(df, x="label", y="probability", range_y=[0, 1], title=title)
    fig.add_hline(y=threshold, line_dash="dash", line_color="#444")
    fig.update_layout(height=320, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def slot_heatmap(slot_probs, threshold):
    fig = px.imshow(
        slot_probs,
        x=ACTIVITY_NAMES,
        y=USER_NAMES,
        zmin=0,
        zmax=1,
        color_continuous_scale="Viridis",
        text_auto=".2f",
        title=f"User-slot activity probabilities, threshold={threshold:.2f}",
    )
    fig.update_layout(height=430, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def stft_plot(raw_spec, channel):
    if channel == "mean":
        view = raw_spec.mean(axis=0)
        title = "STFT log-energy averaged across channels"
    else:
        idx = int(channel)
        view = raw_spec[idx]
        title = f"STFT log-energy channel {idx}"
    fig = px.imshow(
        np.log1p(view),
        labels=dict(x="time bin", y="frequency bin", color="log energy"),
        color_continuous_scale="Magma",
        aspect="auto",
        title=title,
    )
    fig.update_layout(height=430, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def predict_sample(environment, seed, sample_label, uploaded_file, threshold, gate_mode, gate_power, normalize, channel_view, device_name):
    selected = find_weight(environment, int(seed))

    if uploaded_file is not None:
        sample_path = Path(uploaded_file)
        sample_name = sample_path.stem
        spec = np.load(sample_path)
    else:
        sample_name = sample_label
        spec = np.load(FEATURE_DIR / f"{sample_name}.npy")

    raw_spec, model_input = prepare_spec(spec, normalize=normalize)
    model = load_model(selected["model"], selected["path"], raw_spec.shape[0], device_name)
    with torch.no_grad():
        x = torch.from_numpy(model_input).unsqueeze(0).to(torch.device(device_name))
        outputs = model(x)
    activity_probs = torch.sigmoid(outputs["activity_set"]).cpu().numpy()[0]
    occupancy_probs = torch.sigmoid(outputs["occupancy"]).cpu().numpy()[0]
    slot_probs = torch.sigmoid(outputs["slot_activity"]).cpu().numpy()[0].reshape(6, 9)
    slot_probs = apply_occupancy_gate(slot_probs, occupancy_probs, gate_mode, threshold, gate_power)

    pred_scene = [ACTIVITY_NAMES[idx] for idx, value in enumerate(activity_probs) if value > threshold]
    pred_rows = []
    for user_idx, user in enumerate(USER_NAMES):
        user_probs = slot_probs[user_idx]
        top_idx = int(np.argmax(user_probs))
        active_activities = [ACTIVITY_NAMES[idx] for idx, value in enumerate(user_probs) if value > threshold]
        pred_rows.append({
            "user_slot": user,
            "occupancy_prob": round(float(occupancy_probs[user_idx]), 4),
            "occupancy_pred": bool(occupancy_probs[user_idx] > threshold),
            "top_slot_activity": ACTIVITY_NAMES[top_idx],
            "top_slot_prob": round(float(user_probs[top_idx]), 4),
            "slot_activities_over_threshold": ", ".join(active_activities) if active_activities else "-",
        })

    row = annotation_for_label(sample_name)
    true_scene, truth_df = ground_truth(row)
    summary = (
        f"### {sample_name}\n"
        f"- checkpoint: `{Path(selected['path']).name}`\n"
        f"- STFT shape: `{tuple(raw_spec.shape)}`\n"
        f"- predicted scene activities: **{', '.join(pred_scene) if pred_scene else 'none over threshold'}**\n"
        f"- true scene activities: **{', '.join(true_scene) if true_scene else 'unknown / none'}**\n\n"
        "Confidence values are probabilities from the model, not calibrated real-world accuracy."
    )

    return (
        summary,
        bar_plot(ACTIVITY_NAMES, activity_probs, "Scene activity probabilities", threshold),
        bar_plot(USER_NAMES, occupancy_probs, "User occupancy probabilities", threshold),
        slot_heatmap(slot_probs, threshold),
        stft_plot(raw_spec, channel_view),
        pd.DataFrame(pred_rows),
        truth_df,
    )


def build_app():
    weights = weights_index()
    if not weights:
        raise FileNotFoundError(f"No checkpoints found in {DEFAULT_WEIGHTS_DIR}")
    seeds = sorted({row["seed"] for row in weights})
    initial_env = "classroom"
    initial_samples = holdout_labels(initial_env)
    device_choices = ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]

    with gr.Blocks(title="WiMANS STFT Multi-Head Viewer") as demo:
        gr.Markdown(
            "# WiMANS Top-5 STFT Multi-Head Viewer\n"
            "Offline inference viewer for saved CLSTM multi-head checkpoints. "
            "Upload/select a Top-5 STFT `.npy` clip and inspect activity, occupancy, user-slot predictions, and STFT heatmaps."
        )
        with gr.Row():
            with gr.Column(scale=1):
                environment = gr.Dropdown(ENVIRONMENTS, value=initial_env, label="Environment")
                seed = gr.Dropdown(seeds, value=39 if 39 in seeds else seeds[0], label="Seed")
                sample = gr.Dropdown(initial_samples, value=initial_samples[0] if initial_samples else None, label="Holdout test clip")
                uploaded = gr.File(label="Optional upload: .npy STFT clip", file_types=[".npy"], type="filepath")
                threshold = gr.Slider(0.05, 0.95, value=float(default_threshold(initial_env, 39)), step=0.05, label="Decision threshold")
                gate_mode = gr.Dropdown(["none", "binary", "prob"], value="none", label="Occupancy gate")
                gate_power = gr.Slider(0.25, 3.0, value=1.0, step=0.25, label="Gate power")
                normalize = gr.Dropdown(["log_standard", "log", "standard", "none"], value="log_standard", label="Normalization")
                channel_view = gr.Dropdown(["mean"] + [str(idx) for idx in range(45)], value="mean", label="STFT channel view")
                device = gr.Dropdown(device_choices, value=device_choices[0], label="Device")
                run = gr.Button("Run inference", variant="primary")
            with gr.Column(scale=2):
                summary = gr.Markdown()
                activity_plot = gr.Plot(label="Scene activity")
                occupancy_plot = gr.Plot(label="Occupancy")
        slot_plot = gr.Plot(label="User-slot activity heatmap")
        stft_heat = gr.Plot(label="STFT heatmap")
        pred_table = gr.Dataframe(label="Predictions", interactive=False)
        truth_table = gr.Dataframe(label="Ground truth", interactive=False)

        environment.change(update_sample_choices, inputs=environment, outputs=sample)
        environment.change(update_threshold, inputs=[environment, seed], outputs=threshold)
        seed.change(update_threshold, inputs=[environment, seed], outputs=threshold)
        run.click(
            predict_sample,
            inputs=[environment, seed, sample, uploaded, threshold, gate_mode, gate_power, normalize, channel_view, device],
            outputs=[summary, activity_plot, occupancy_plot, slot_plot, stft_heat, pred_table, truth_table],
        )
    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the WiMANS STFT multi-head Gradio viewer.")
    parser.add_argument("--port", type=int, default=int(os.environ.get("WIMANS_APP_PORT", "7860")))
    args = parser.parse_args()
    app = build_app()
    app.launch(server_name="127.0.0.1", server_port=args.port, show_error=True, prevent_thread_lock=True)
    while True:
        time.sleep(3600)
