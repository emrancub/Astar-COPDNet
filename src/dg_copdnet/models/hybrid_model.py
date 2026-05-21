from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tvm

from dg_copdnet.config import ModelConfig
from dg_copdnet.models.blocks import AttentionPooling, GradientReversal


class EfficientNetSpecBranch(nn.Module):
    def __init__(self, embed_dim: int, pretrained: bool):
        super().__init__()
        weights = tvm.EfficientNet_B2_Weights.DEFAULT if pretrained else None
        model = tvm.efficientnet_b2(weights=weights)
        first = model.features[0][0]
        model.features[0][0] = nn.Conv2d(1, first.out_channels, kernel_size=first.kernel_size, stride=first.stride, padding=first.padding, bias=False)
        self.features = model.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(nn.Linear(1408, embed_dim), nn.GELU())

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.proj(x)


class CRNNSpecBranch(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        layers = []
        in_ch = 1
        for out_ch in cfg.crnn_channels:
            layers.extend([
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.GELU(),
                nn.MaxPool2d(2),
            ])
            in_ch = out_ch
        self.cnn = nn.Sequential(*layers)
        self.rnn = nn.GRU(
            input_size=cfg.crnn_channels[-1] * 8,
            hidden_size=cfg.crnn_hidden,
            num_layers=cfg.crnn_layers,
            batch_first=True,
            bidirectional=True,
            dropout=cfg.dropout if cfg.crnn_layers > 1 else 0.0,
        )
        self.attn = AttentionPooling(cfg.crnn_hidden * 2)
        self.proj = nn.Sequential(nn.Linear(cfg.crnn_hidden * 2, cfg.embed_dim), nn.GELU())

    def forward(self, x):
        x = self.cnn(x)
        b, c, f, t = x.shape
        x = x.permute(0, 3, 1, 2).reshape(b, t, -1)
        x, _ = self.rnn(x)
        pooled, weights = self.attn(x)
        return self.proj(pooled), weights


class HandcraftedBranch(nn.Module):
    def __init__(self, input_dim: int, cfg: ModelConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, cfg.handcrafted_dim),
            nn.BatchNorm1d(cfg.handcrafted_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.handcrafted_dim, cfg.embed_dim),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class HybridDGRespNet(nn.Module):
    def __init__(self, handcrafted_input_dim: int, cfg: ModelConfig, num_domains: int):
        super().__init__()
        self.effnet_branch = EfficientNetSpecBranch(cfg.embed_dim, pretrained=cfg.pretrained_effnet)
        self.crnn_branch = CRNNSpecBranch(cfg)
        self.handcrafted_branch = HandcraftedBranch(handcrafted_input_dim, cfg)

        self.gate = nn.Sequential(
            nn.Linear(cfg.embed_dim * 3, 3),
            nn.Softmax(dim=1),
        )
        self.fusion = nn.Sequential(
            nn.Linear(cfg.embed_dim, cfg.embed_dim),
            nn.BatchNorm1d(cfg.embed_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )
        self.classifier = nn.Linear(cfg.embed_dim, 1)
        self.projection = nn.Sequential(
            nn.Linear(cfg.embed_dim, cfg.projection_dim),
            nn.GELU(),
            nn.Linear(cfg.projection_dim, cfg.projection_dim),
        )
        self.grl = GradientReversal(cfg.grl_lambda)
        self.domain_classifier = nn.Sequential(
            nn.Linear(cfg.embed_dim, cfg.domain_hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.domain_hidden_dim, num_domains),
        )

    def forward(self, spectrogram: torch.Tensor, handcrafted: torch.Tensor):
        eff = self.effnet_branch(spectrogram)
        crnn, att = self.crnn_branch(spectrogram)
        hand = self.handcrafted_branch(handcrafted)
        concat = torch.cat([eff, crnn, hand], dim=1)
        w = self.gate(concat)
        fused = w[:, 0:1] * eff + w[:, 1:2] * crnn + w[:, 2:3] * hand
        fused = self.fusion(fused)
        logits = self.classifier(fused).squeeze(1)
        proj = self.projection(fused)
        domain_logits = self.domain_classifier(self.grl(fused))
        return {
            "logits": logits,
            "projection": proj,
            "embedding": fused,
            "domain_logits": domain_logits,
            "attention_weights": att,
            "branch_weights": w,
        }
