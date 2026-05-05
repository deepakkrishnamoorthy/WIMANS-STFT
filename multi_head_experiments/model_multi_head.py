import torch
import torch.nn as nn
from torchvision.models import resnet18


class ResNetBackbone(nn.Module):
    def __init__(self, input_channels=45):
        super().__init__()
        resnet = resnet18(weights=None)
        resnet.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.features = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )

    def forward(self, x):
        return self.features(x)


class MultiHeadOutputMixin:
    def build_heads(self, feature_dim, dropout):
        self.activity_set_head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 9),
        )
        self.occupancy_head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 6),
        )
        self.slot_activity_head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 54),
        )

    def heads_forward(self, features):
        return {
            "activity_set": self.activity_set_head(features),
            "occupancy": self.occupancy_head(features),
            "slot_activity": self.slot_activity_head(features),
        }


class ResNet18MultiHead(nn.Module, MultiHeadOutputMixin):
    def __init__(self, input_channels=45, dropout=0.5):
        super().__init__()
        self.backbone = ResNetBackbone(input_channels=input_channels)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.build_heads(feature_dim=512, dropout=dropout)

    def forward(self, x):
        fmap = self.backbone(x)
        features = self.pool(fmap).flatten(1)
        return self.heads_forward(features)


class CLSTMMultiHead(nn.Module, MultiHeadOutputMixin):
    def __init__(self, input_channels=45, hidden_size=256, num_layers=2, dropout=0.3):
        super().__init__()
        self.backbone = ResNetBackbone(input_channels=input_channels)
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.proj = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Dropout(dropout),
        )
        self.build_heads(feature_dim=hidden_size * 2, dropout=dropout)

    def forward(self, x):
        fmap = self.backbone(x)
        seq = fmap.mean(dim=2).permute(0, 2, 1)
        seq, _ = self.lstm(seq)
        features = self.proj(seq.mean(dim=1))
        return self.heads_forward(features)


class TemporalAttentionBlock(nn.Module):
    def __init__(self, dim=256, heads=8, dropout=0.1):
        super().__init__()
        self.norm_attn = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True, dropout=dropout)
        self.drop_attn = nn.Dropout(dropout)
        self.norm_conv = nn.LayerNorm(dim)
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(dim, dim, kernel_size=k, padding=k // 2),
                nn.BatchNorm1d(dim),
                nn.Dropout(dropout),
                nn.LeakyReLU(inplace=True),
            )
            for k in (1, 3, 5)
        ])

    def forward(self, x):
        attn_in = self.norm_attn(x)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in)
        x = x + self.drop_attn(attn_out)
        conv_in = self.norm_conv(x).permute(0, 2, 1)
        conv_out = torch.stack([conv(conv_in) for conv in self.convs], dim=0).mean(dim=0)
        return x + conv_out.permute(0, 2, 1)


class THATStyleMultiHead(nn.Module, MultiHeadOutputMixin):
    def __init__(self, input_channels=45, dim=256, depth=3, dropout=0.3):
        super().__init__()
        self.backbone = ResNetBackbone(input_channels=input_channels)
        self.proj = nn.Sequential(
            nn.Linear(512, dim),
            nn.LayerNorm(dim),
            nn.LeakyReLU(inplace=True),
        )
        self.blocks = nn.ModuleList([TemporalAttentionBlock(dim=dim) for _ in range(depth)])
        self.summary = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Dropout(dropout),
        )
        self.build_heads(feature_dim=dim, dropout=dropout)

    def forward(self, x):
        fmap = self.backbone(x)
        seq = fmap.mean(dim=2).permute(0, 2, 1)
        seq = self.proj(seq)
        for block in self.blocks:
            seq = block(seq)
        features = self.summary(seq.mean(dim=1))
        return self.heads_forward(features)
