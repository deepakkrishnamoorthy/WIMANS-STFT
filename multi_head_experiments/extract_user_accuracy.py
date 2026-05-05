import os
import sys
import torch
import numpy as np
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# Add paths for WiMANS imports
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(ROOT, "WiMANS-main", "benchmark", "wifi_csi"))

from dataset_multi_head import MultiHeadSTFTDataset
from load_data import load_data_y
from model_multi_head import ResNet18MultiHead

# Configuration
ENVIRONMENTS = ["classroom", "meeting_room", "empty_room"]
ANNOTATION = r"D:\Deepak\wifi_csi\WiMANS-main\dataset\annotation.csv"
DATA_DIR = r"D:\Deepak\wifi_csi\dataset\stft_top5_multichannel_npy"
MODEL_DIR = r"D:\Deepak\wifi_csi\multi_head_experiments\outputs\saved_models_multi_head_resnet18_multichannel_band5"
OUTPUT_FILE = r"D:\Deepak\wifi_csi\multi_head_experiments\user_accuracy_results.txt"

def split_data(data_pd_y, test_size=0.2, split_seed=39):
    train_val_y, test_y = train_test_split(
        data_pd_y,
        test_size=test_size,
        shuffle=True,
        random_state=split_seed,
    )
    return test_y

def extract_metrics():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_text = []
    
    header = f"{'Environment':<15} | {'User Accuracy (%)':<20}"
    results_text.append(header)
    results_text.append("-" * len(header))
    print(header)
    print("-" * len(header))

    for env in ENVIRONMENTS:
        # Load data
        data_pd_y = load_data_y(ANNOTATION, var_environment=[env], var_wifi_band=["5"])
        test_y = split_data(data_pd_y)
        
        dataset = MultiHeadSTFTDataset(test_y, DATA_DIR, max_len=200, normalize="log_standard")
        loader = DataLoader(dataset, batch_size=64, shuffle=False)
        
        # Initialize Model (ResNet18 as per the folder name)
        model = ResNet18MultiHead(input_channels=45).to(device)
        model_path = os.path.join(MODEL_DIR, f"multi_head_resnet18_multichannel_{env}_5_seed39.pth")
        
        if not os.path.exists(model_path):
            print(f"Model not found: {model_path}")
            continue
            
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        all_labels = []
        all_preds = []
        
        with torch.no_grad():
            for batch in loader:
                x = batch["x"].to(device)
                labels = batch["slot_activity"].cpu().numpy() # (Batch, 54)
                
                outputs = model(x)
                probs = torch.sigmoid(outputs["slot_activity"]).cpu().numpy() # (Batch, 54)
                preds = (probs > 0.5).astype(int)
                
                all_labels.append(labels)
                all_preds.append(preds)
        
        all_labels = np.vstack(all_labels) # (N, 54)
        all_preds = np.vstack(all_preds)   # (N, 54)
        
        # Reshape to per-user: (N * 6, 9)
        labels_user = all_labels.reshape(-1, 9)
        preds_user = all_preds.reshape(-1, 9)
        
        # Calculate User Accuracy (as defined in baseline)
        user_acc = accuracy_score(labels_user, preds_user) * 100.0
        
        line = f"{env:<15} | {user_acc:<20.4f}"
        results_text.append(line)
        print(line)

    # Save to file
    with open(OUTPUT_FILE, "w") as f:
        f.write("\n".join(results_text))
    print(f"\nResults saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    extract_metrics()
