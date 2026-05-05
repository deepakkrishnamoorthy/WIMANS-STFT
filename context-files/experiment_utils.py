import os
import json
import time
import random
import subprocess
import numpy as np
import torch

def setup_experiment(experiment_name, key_vars, config_dict, base_dir="."):
    """
    Sets up the environment for a new experiment run, enforcing reproducibility.
    """
    # 1. Generate Run ID
    timestamp = time.strftime("%Y%m%d_%H%M")
    run_id = f"{timestamp}_{experiment_name}_{key_vars}"
    
    # 2. Create directories
    outputs_dir = os.path.join(base_dir, "outputs", run_id)
    data_cache_dir = os.path.join(base_dir, "data_cache")
    models_dir = os.path.join(base_dir, "saved_models", run_id)
    
    os.makedirs(outputs_dir, exist_ok=True)
    os.makedirs(data_cache_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    # 3. Enforce global seed
    seed = config_dict.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    # 4. Gather Reproducibility Info
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
        git_diff = subprocess.check_output(["git", "status", "--porcelain"]).decode("utf-8").strip()
    except Exception:
        git_commit = "unknown"
        git_diff = "unknown"
        
    hardware_info = {
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
        "torch_version": torch.__version__
    }
    
    # 5. Save Config
    final_config = {
        "run_id": run_id,
        "experiment_name": experiment_name,
        "key_vars": key_vars,
        "seed": seed,
        "git_commit": git_commit,
        "has_uncommitted_changes": len(git_diff) > 0,
        "hardware": hardware_info,
        "parameters": config_dict
    }
    
    with open(os.path.join(outputs_dir, "config.json"), "w") as f:
        json.dump(final_config, f, indent=4)
        
    print(f"[*] Experiment {run_id} initialized.")
    print(f"    Outputs: {outputs_dir}")
    print(f"    Seed:    {seed}")
    
    return {
        "run_id": run_id,
        "outputs_dir": outputs_dir,
        "data_cache_dir": data_cache_dir,
        "models_dir": models_dir
    }

def finalize_experiment(run_id, outputs_dir, metrics_dict, summary_text=""):
    """
    Saves the final report for the experiment.
    """
    report_path = os.path.join(outputs_dir, "report.md")
    
    with open(report_path, "w") as f:
        f.write(f"# Experiment Report: {run_id}\n\n")
        f.write("## Summary\n")
        f.write(f"{summary_text}\n\n")
        f.write("## Final Metrics\n")
        f.write("```json\n")
        f.write(json.dumps(metrics_dict, indent=4))
        f.write("\n```\n")
        
    print(f"[*] Experiment finalized. Report saved to {report_path}")
