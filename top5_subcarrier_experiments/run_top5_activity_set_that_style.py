import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "top5_subcarrier_experiments", "outputs")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, os.path.join(ROOT, "resnet_baseline"))

from train_activity_set_common import run_experiment


if __name__ == "__main__":
    sys.argv.extend([
        "--result-json", os.path.join(OUT, "that_style_activity_set_top5_results.json"),
        "--result-excel", os.path.join(OUT, "that_style_activity_set_top5_analysis.xlsx"),
        "--save-dir", os.path.join(OUT, "saved_models_that_style_activity_set"),
    ])
    run_experiment(
        "that_style_activity_set",
        "Top-5 subcarrier activity-set experiment with ResNet18 + THAT-style head.",
    )
