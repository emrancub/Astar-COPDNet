from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tvm

from dg_copdnet.config import ModelConfig
from dg_copdnet.models.blocks import AttentionPooling
from dg_copdnet.models.hybrid_model import HybridDGRespNet, EfficientNetSpecBranch, CRNNSpecBranch, HandcraftedBranch


class SpectrogramCNN(nn.Module):
    def __init__(self, num_classes: int, cfg: ModelConfig):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.GELU(), nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Sequential(nn.Flatten(), nn.Dropout(cfg.dropout), nn.Linear(256, cfg.embed_dim), nn.GELU())
        self.head = nn.Linear(cfg.embed_dim, num_classes)

    def forward(self, spectrogram, handcrafted=None):
        emb = self.fc(self.encoder(spectrogram))
        logits = self.head(emb)
        return {'logits': logits.squeeze(1) if logits.shape[1]==1 else logits, 'embedding': emb}


class AeroCOPDNetBaseline(nn.Module):
    """Recent SOTA comparator: MobileNetV2 backbone with Squeeze-and-Excitation and Attentive Statistics Pooling (Hasan et al., 2026)."""
    def __init__(self, num_classes: int, cfg: ModelConfig):
        super().__init__()
        weights = tvm.MobileNet_V2_Weights.DEFAULT if cfg.pretrained_effnet else None
        mnet = tvm.mobilenet_v2(weights=weights)
        first = mnet.features[0][0]
        mnet.features[0][0] = nn.Conv2d(1, first.out_channels, kernel_size=first.kernel_size, stride=first.stride, padding=first.padding, bias=False)
        self.features = mnet.features
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1280, 320),
            nn.ReLU(inplace=True),
            nn.Linear(320, 1280),
            nn.Sigmoid(),
        )
        self.attn_pool = AttentionPooling(1280)
        self.proj = nn.Sequential(
            nn.Linear(1280, cfg.embed_dim),
            nn.LayerNorm(cfg.embed_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )
        self.head = nn.Linear(cfg.embed_dim, num_classes)

    def forward(self, spectrogram, handcrafted=None):
        feat = self.features(spectrogram)  # (B, 1280, F', T')
        se_w = self.se(feat).unsqueeze(-1).unsqueeze(-1)
        feat = feat * se_w
        seq = feat.mean(dim=2).permute(0, 2, 1)  # (B, T', 1280)
        pooled, att = self.attn_pool(seq)
        emb = self.proj(pooled)
        logits = self.head(emb)
        return {'logits': logits.squeeze(1) if logits.shape[1]==1 else logits, 'embedding': emb, 'attention_weights': att}


class EfficientNetOnly(nn.Module):
    def __init__(self, num_classes: int, cfg: ModelConfig):
        super().__init__()
        self.branch = EfficientNetSpecBranch(cfg.embed_dim, pretrained=cfg.pretrained_effnet)
        self.head = nn.Linear(cfg.embed_dim, num_classes)

    def forward(self, spectrogram, handcrafted=None):
        emb = self.branch(spectrogram)
        logits = self.head(emb)
        return {'logits': logits.squeeze(1) if logits.shape[1]==1 else logits, 'embedding': emb}


class CRNNOnly(nn.Module):
    def __init__(self, num_classes: int, cfg: ModelConfig):
        super().__init__()
        self.branch = CRNNSpecBranch(cfg)
        self.head = nn.Linear(cfg.embed_dim, num_classes)

    def forward(self, spectrogram, handcrafted=None):
        emb, att = self.branch(spectrogram)
        logits = self.head(emb)
        return {'logits': logits.squeeze(1) if logits.shape[1]==1 else logits, 'embedding': emb, 'attention_weights': att}


class HandcraftedMLP(nn.Module):
    def __init__(self, handcrafted_dim: int, num_classes: int, cfg: ModelConfig):
        super().__init__()
        self.branch = HandcraftedBranch(handcrafted_dim, cfg)
        self.head = nn.Linear(cfg.embed_dim, num_classes)

    def forward(self, spectrogram, handcrafted):
        emb = self.branch(handcrafted)
        logits = self.head(emb)
        return {'logits': logits.squeeze(1) if logits.shape[1]==1 else logits, 'embedding': emb}


def create_model(model_name: str, handcrafted_dim: int, cfg: ModelConfig, num_domains: int, num_classes: int = 1):
    name = model_name.lower()
    if name in {'hybrid_dg_respnet', 'proposed', 'hybrid'}:
        return HybridDGRespNet(handcrafted_dim, cfg, num_domains=num_domains) if num_classes == 1 else HybridDGRespNetMulti(handcrafted_dim, cfg, num_domains, num_classes)
    if name == 'aerocopdnet':
        return AeroCOPDNetBaseline(num_classes, cfg)
    if name == 'spec_cnn':
        return SpectrogramCNN(num_classes, cfg)
    if name == 'effnet_only':
        return EfficientNetOnly(num_classes, cfg)
    if name == 'crnn_only':
        return CRNNOnly(num_classes, cfg)
    if name == 'handcrafted_mlp':
        return HandcraftedMLP(handcrafted_dim, num_classes, cfg)
    raise ValueError(f'Unsupported model name: {model_name}')


class HybridDGRespNetMulti(nn.Module):
    def __init__(self, handcrafted_input_dim: int, cfg: ModelConfig, num_domains: int, num_classes: int):
        super().__init__()
        self.base = HybridDGRespNet(handcrafted_input_dim, cfg, num_domains)
        self.base.classifier = nn.Linear(cfg.embed_dim, num_classes)

    def forward(self, spectrogram, handcrafted):
        out = self.base(spectrogram, handcrafted)
        if out['logits'].ndim == 1:
            out['logits'] = out['logits'].unsqueeze(1)
        return out
