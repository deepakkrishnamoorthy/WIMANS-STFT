import torch
import torch.nn as nn

class CustomCNNBaseline(nn.Module):
    def __init__(self, num_classes=54):
        """
        num_classes = 54 (6 users * 9 activities)
        Input: (1, 129, 200) spectrogram
        """
        super(CustomCNNBaseline, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1
            # Input: (1, 129, 200)
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output: (32, 64, 100)
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output: (64, 32, 50)
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output: (128, 16, 25)
            
            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Output: (256, 8, 12)
        )
        
        # 256 channels * 8 height * 12 width = 24576
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 12, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
