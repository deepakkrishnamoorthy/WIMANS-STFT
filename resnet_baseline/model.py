import torch
import torch.nn as nn
from torchvision.models import resnet18

class ResNetBaseline(nn.Module):
    def __init__(self, num_classes=54):
        """
        num_classes = 54 (6 users * 9 activities)
        """
        super(ResNetBaseline, self).__init__()
        
        # Load a base ResNet18 without downloading pretrained weights.
        self.resnet = resnet18(weights=None)
        
        # The STFT spectrogram is a single-channel (grayscale) image.
        # We replace the first convolutional layer (which expects 3 RGB channels)
        # to accept 1 input channel.
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Replace the final fully connected layer to output the 54 activities vector
        # Added a Dropout layer (p=0.5) to combat severe overfitting observed in the first run.
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features, num_classes)
        )
        
    def forward(self, x):
        # We output the raw logits. 
        # The BCEWithLogitsLoss in the training loop will apply the Sigmoid activation.
        return self.resnet(x)
