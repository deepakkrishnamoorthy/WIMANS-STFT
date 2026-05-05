import math

import torch
import torch.nn as nn
from torchvision.models import resnet18


class ResNetFeatureBackbone(nn.Module):
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


class ResNet18MultiChannel(nn.Module):
    def __init__(self, input_channels=45, num_classes=54, dropout=0.5):
        super().__init__()
        self.backbone = ResNetFeatureBackbone(input_channels=input_channels)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.head(self.pool(self.backbone(x)))


class ResNetCLSTMMultiChannel(nn.Module):
    def __init__(self, input_channels=45, num_classes=54, hidden_size=256, num_layers=2, dropout=0.3):
        super().__init__()
        self.backbone = ResNetFeatureBackbone(input_channels=input_channels)
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden_size * 2)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        fmap = self.backbone(x)
        seq = fmap.mean(dim=2).permute(0, 2, 1)
        seq, _ = self.lstm(seq)
        pooled = seq.mean(dim=1)
        return self.classifier(self.dropout(self.norm(pooled)))


class SlotAttention(nn.Module):
    def __init__(self, num_slots=6, dim=256, iters=3, eps=1e-8):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5
        self.slots = nn.Parameter(torch.randn(1, num_slots, dim) / math.sqrt(dim))
        self.norm_inputs = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_mlp = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.gru = nn.GRUCell(dim, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 2), nn.ReLU(inplace=True), nn.Linear(dim * 2, dim))

    def forward(self, inputs):
        batch = inputs.shape[0]
        slots = self.slots.expand(batch, -1, -1)
        inputs = self.norm_inputs(inputs)
        k = self.to_k(inputs)
        v = self.to_v(inputs)
        for _ in range(self.iters):
            old_slots = slots
            q = self.to_q(self.norm_slots(slots))
            attn_logits = torch.einsum("bid,bjd->bij", q, k) * self.scale
            attn = attn_logits.softmax(dim=1) + self.eps
            attn = attn / attn.sum(dim=-1, keepdim=True)
            updates = torch.einsum("bjd,bij->bid", v, attn)
            slots = self.gru(updates.reshape(-1, updates.shape[-1]), old_slots.reshape(-1, old_slots.shape[-1]))
            slots = slots.reshape(batch, self.num_slots, -1)
            slots = slots + self.mlp(self.norm_mlp(slots))
        return slots


class ResNetSlotAttentionMultiChannel(nn.Module):
    def __init__(self, input_channels=45, num_users=6, num_activities=9, slot_dim=256, slot_iters=3, dropout=0.3):
        super().__init__()
        self.num_users = num_users
        self.num_activities = num_activities
        self.backbone = ResNetFeatureBackbone(input_channels=input_channels)
        self.token_proj = nn.Sequential(nn.Linear(512, slot_dim), nn.LayerNorm(slot_dim), nn.ReLU(inplace=True))
        self.slot_attention = SlotAttention(num_slots=num_users, dim=slot_dim, iters=slot_iters)
        self.head = nn.Sequential(nn.LayerNorm(slot_dim), nn.Dropout(dropout), nn.Linear(slot_dim, num_activities))

    def forward(self, x):
        fmap = self.backbone(x)
        tokens = fmap.flatten(2).permute(0, 2, 1)
        slots = self.slot_attention(self.token_proj(tokens))
        return self.head(slots).reshape(x.shape[0], self.num_users * self.num_activities)


class GaussianPosition(nn.Module):
    def __init__(self, dim, max_len=32, num_gaussian=8):
        super().__init__()
        self.embedding = nn.Parameter(torch.empty(num_gaussian, dim))
        nn.init.xavier_uniform_(self.embedding)
        position = torch.arange(0.0, max_len).unsqueeze(1).repeat(1, num_gaussian)
        mu = torch.linspace(0.0, max_len - 1.0, num_gaussian).unsqueeze(0)
        sigma = torch.full((1, num_gaussian), max(1.0, max_len / num_gaussian))
        self.register_buffer("position", position)
        self.mu = nn.Parameter(mu)
        self.sigma = nn.Parameter(sigma)

    def forward(self, x):
        seq_len = x.shape[1]
        position = self.position[:seq_len]
        sigma = self.sigma.clamp_min(1e-3)
        pdf = -((position - self.mu) ** 2) / (2 * sigma * sigma) - torch.log(sigma)
        encoding = pdf.softmax(dim=-1) @ self.embedding
        return x + encoding.unsqueeze(0)


class ThatEncoderBlock(nn.Module):
    def __init__(self, dim=256, heads=8, kernels=(1, 3, 5), dropout=0.1):
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
            for k in kernels
        ])
        self.drop_conv = nn.Dropout(dropout)

    def forward(self, x):
        attn_in = self.norm_attn(x)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in)
        x = x + self.drop_attn(attn_out)
        conv_in = self.norm_conv(x).permute(0, 2, 1)
        conv_out = torch.stack([conv(conv_in) for conv in self.convs], dim=0).mean(dim=0)
        return x + self.drop_conv(conv_out.permute(0, 2, 1))


class ResNetTHATStyleMultiChannel(nn.Module):
    def __init__(self, input_channels=45, num_classes=54, dim=256, depth=3, heads=8, dropout=0.3):
        super().__init__()
        self.backbone = ResNetFeatureBackbone(input_channels=input_channels)
        self.proj = nn.Sequential(nn.Linear(512, dim), nn.LayerNorm(dim), nn.LeakyReLU(inplace=True))
        self.position = GaussianPosition(dim=dim, max_len=32, num_gaussian=8)
        self.blocks = nn.ModuleList([ThatEncoderBlock(dim=dim, heads=heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.summary_convs = nn.ModuleList([
            nn.Conv1d(dim, 128, kernel_size=1),
            nn.Conv1d(dim, 128, kernel_size=3, padding=1),
            nn.Conv1d(dim, 128, kernel_size=5, padding=2),
        ])
        self.head = nn.Sequential(nn.LeakyReLU(inplace=True), nn.Dropout(dropout), nn.Linear(384, num_classes))

    def forward(self, x):
        fmap = self.backbone(x)
        seq = fmap.mean(dim=2).permute(0, 2, 1)
        seq = self.position(self.proj(seq))
        for block in self.blocks:
            seq = block(seq)
        seq = self.norm(seq).permute(0, 2, 1)
        pooled = torch.cat([conv(seq).sum(dim=-1) for conv in self.summary_convs], dim=-1)
        return self.head(pooled)
