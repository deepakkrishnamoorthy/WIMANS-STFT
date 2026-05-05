import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Add WiMANS-main to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'WiMANS-main', 'benchmark', 'wifi_csi')))
from load_data import load_data_y
from preset import preset

from dataset import STFTDataset
from model import ResNetBaseline

def run_environment(env_name):
    # Override annotation path to the actual location
    preset["path"]["data_y"] = r"d:\Deepak\wifi_csi\WiMANS-main\dataset\annotation.csv"
    
    # Load dataset labels just for the selected environment
    data_pd_y = load_data_y(preset["path"]["data_y"], var_environment=[env_name])
    
    # EXACT split logic used in the paper: 80/20, shuffle=True, random_state=39
    data_train_y, data_test_y = train_test_split(data_pd_y, test_size=0.2, shuffle=True, random_state=39)
    
    # Datasets
    # We use max_len=200 since the STFT extracted files are roughly ~179 long.
    data_dir = r"d:\Deepak\wifi_csi\dataset\stft_top5_npy"
    train_dataset = STFTDataset(data_train_y, data_dir, max_len=200)
    test_dataset = STFTDataset(data_test_y, data_dir, max_len=200)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNetBaseline().to(device)
    
    # Loss and optimizer
    # Pos weight = 6 to balance the 0s and 1s in multi-user matrix, exactly as in cnn_2d.py
    pos_weight = torch.tensor([6] * 54).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    # Train for 1 epoch (Smoke Test)
    model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
    # Evaluate
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            # Threshold the Sigmoid output at 0.5
            preds = (torch.sigmoid(outputs) > 0.5).float().cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.numpy())
            
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    
    # Calculate exact match accuracy
    acc = accuracy_score(all_labels.astype(int), all_preds.astype(int))
    return acc * 100.0

if __name__ == '__main__':
    print("Running Smoke Test (1 Epoch) for ResNet Baseline...\n")
    envs = ['classroom', 'meeting_room', 'empty_room']
    results = {}
    
    for env in envs:
        print(f"[*] Training on environment: {env}...")
        acc = run_environment(env)
        results[env] = acc
        print(f"    -> Test Accuracy: {acc:.2f}%\n")
        
    print("="*60)
    print("ResNet Smoke Test Results (Accuracy %)")
    print("="*60)
    # Replicating the table formatting from the image
    print(f"Model          | {'Classroom':<12} | {'Meeting Room':<12} | {'Empty Room':<12}")
    print("-" * 60)
    print(f"ResNet (STFT)  | {results['classroom']:<12.2f} | {results['meeting_room']:<12.2f} | {results['empty_room']:<12.2f}")
    print("="*60)
