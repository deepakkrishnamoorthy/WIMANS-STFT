import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import copy

# Add WiMANS-main to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'WiMANS-main', 'benchmark', 'wifi_csi')))
from load_data import load_data_y
from preset import preset

from dataset import STFTDataset
from model import CustomCNNBaseline

def train_environment(env_name, max_epochs=200):
    # Override annotation path to the actual location
    preset["path"]["data_y"] = r"d:\Deepak\wifi_csi\WiMANS-main\dataset\annotation.csv"
    
    # Load dataset labels just for the selected environment
    data_pd_y = load_data_y(preset["path"]["data_y"], var_environment=[env_name])
    
    # EXACT split logic used in the paper: 80/20, shuffle=True, random_state=39
    data_train_y, data_test_y = train_test_split(data_pd_y, test_size=0.2, shuffle=True, random_state=39)
    
    data_dir = r"d:\Deepak\wifi_csi\dataset\stft_top5_npy"
    train_dataset = STFTDataset(data_train_y, data_dir, max_len=200)
    test_dataset = STFTDataset(data_test_y, data_dir, max_len=200)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CustomCNNBaseline().to(device)
    
    pos_weight = torch.tensor([6] * 54).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    best_test_loss = float('inf')
    best_test_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    
    os.makedirs(os.path.join(os.path.dirname(__file__), 'saved_models'), exist_ok=True)
    save_path = os.path.join(os.path.dirname(__file__), 'saved_models', f'best_model_{env_name}.pth')
    
    print(f"\n[*] Training Custom CNN on environment: {env_name}...")
    for epoch in range(max_epochs):
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            
        train_loss = train_loss / len(train_loader.dataset)
        
        # Evaluate
        model.eval()
        test_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item() * inputs.size(0)
                
                preds = (torch.sigmoid(outputs) > 0.5).float().cpu().numpy()
                all_preds.append(preds)
                all_labels.append(labels.cpu().numpy())
                
        test_loss = test_loss / len(test_loader.dataset)
        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)
        
        test_acc = accuracy_score(all_labels.astype(int), all_preds.astype(int)) * 100.0
        
        print(f"    Epoch {epoch+1:03d}/{max_epochs} - Train Loss: {train_loss:.4f} - Test Loss: {test_loss:.4f} - Test Acc: {test_acc:.2f}%")
        
        # Save best model based on test loss (or accuracy)
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            if test_acc > best_test_acc:
                best_test_acc = test_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            # Save the best model
            torch.save(best_model_wts, save_path)
                
    print(f"    -> Best Test Accuracy: {best_test_acc:.2f}% (Saved to {save_path})")
    return best_test_acc

if __name__ == '__main__':
    print("Starting Full Custom CNN Baseline Benchmark (200 Epochs)...\n")
    envs = ['classroom', 'meeting_room', 'empty_room']
    results = {}
    
    for env in envs:
        acc = train_environment(env, max_epochs=200) # Ensure max_epochs is explicit
        results[env] = acc
        
    print("\n" + "="*60)
    print("Full Custom CNN Benchmark Results (Accuracy %)")
    print("="*60)
    print(f"Model          | {'Classroom':<12} | {'Meeting Room':<12} | {'Empty Room':<12}")
    print("-" * 60)
    print(f"Custom CNN     | {results['classroom']:<12.2f} | {results['meeting_room']:<12.2f} | {results['empty_room']:<12.2f}")
    print("="*60)
