"""DARE-COPDNet: reviewer-oriented domain-robust respiratory model.

Design goals:
1) binary COPD-vs-non-COPD and multiclass disease/GOLD-stage tasks;
2) explicit domain robustness rather than hidden mixed-source CV;
3) interpretable branch weights and attention maps;
4) optional self-supervised/contrastive representation head.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torchvision.models as tvm
except Exception:  # pragma: no cover
    tvm = None

from dg_copdnet.models.blocks import AttentionPooling, GradientReversal

TaskType = Literal["binary", "multiclass"]


class MixStyle(nn.Module):
    """MixStyle regularization for domain generalization.

    Randomly mixes feature means/stds between samples. This discourages source/style
    shortcuts such as stethoscope/device/background signatures.
    """
    def __init__(self, p: float = 0.5, alpha: float = 0.3, eps: float = 1e-6):
        super().__init__()
        self.p, self.alpha, self.eps = p, alpha, eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or torch.rand(1, device=x.device).item() > self.p or x.size(0) < 2:
            return x
        dims = tuple(range(2, x.dim()))
        mu = x.mean(dim=dims, keepdim=True)
        sig = (x.var(dim=dims, keepdim=True) + self.eps).sqrt()
        x_normed = (x - mu) / sig
        perm = torch.randperm(x.size(0), device=x.device)
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample((x.size(0), 1, 1, 1)).to(x.device)
        mu_mix = mu * lam + mu[perm] * (1 - lam)
        sig_mix = sig * lam + sig[perm] * (1 - lam)
        return x_normed * sig_mix + mu_mix


class LightSpecEncoder(nn.Module):
    def __init__(self, embed_dim: int = 256, mixstyle_p: float = 0.5):
        super().__init__()
        chans = [1, 32, 64, 128, 256]
        blocks = []
        for i in range(len(chans) - 1):
            blocks += [
                nn.Conv2d(chans[i], chans[i + 1], 3, padding=1, bias=False),
                nn.BatchNorm2d(chans[i + 1]), nn.GELU(),
                MixStyle(p=mixstyle_p) if i in (1, 2) else nn.Identity(),
                nn.Conv2d(chans[i + 1], chans[i + 1], 3, padding=1, bias=False),
                nn.BatchNorm2d(chans[i + 1]), nn.GELU(),
                nn.MaxPool2d(2),
            ]
        self.net = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(nn.Linear(256, embed_dim), nn.LayerNorm(embed_dim), nn.GELU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        return self.proj(self.pool(z).flatten(1))


class EfficientNetEncoder(nn.Module):
    def __init__(self, embed_dim: int = 256, pretrained: bool = True, mixstyle_p: float = 0.25):
        super().__init__()
        if tvm is None:
            raise ImportError("torchvision is required for EfficientNetEncoder")
        weights = tvm.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = tvm.efficientnet_b0(weights=weights)
        first = model.features[0][0]
        model.features[0][0] = nn.Conv2d(1, first.out_channels, first.kernel_size, first.stride, first.padding, bias=False)
        self.features = model.features
        self.mixstyle = MixStyle(p=mixstyle_p)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(nn.Linear(1280, embed_dim), nn.LayerNorm(embed_dim), nn.GELU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.features(x)
        z = self.mixstyle(z)
        return self.proj(self.pool(z).flatten(1))


class TemporalEncoder(nn.Module):
    def __init__(self, embed_dim: int = 256, hidden: int = 128, dropout: float = 0.25):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.GELU(), nn.MaxPool2d((2, 2)),
            nn.Conv2d(64, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.GELU(), nn.MaxPool2d((2, 2)),
            nn.Conv2d(128, 192, 3, padding=1, bias=False), nn.BatchNorm2d(192), nn.GELU(), nn.MaxPool2d((2, 1)),
        )
        self.rnn = nn.GRU(192 * 16, hidden, num_layers=2, batch_first=True, bidirectional=True, dropout=dropout)
        self.attn = AttentionPooling(hidden * 2)
        self.proj = nn.Sequential(nn.Linear(hidden * 2, embed_dim), nn.LayerNorm(embed_dim), nn.GELU())

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.cnn(x)
        b, c, f, t = z.shape
        z = z.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)
        z, _ = self.rnn(z)
        pooled, weights = self.attn(z)
        return self.proj(pooled), weights


class HandcraftedEncoder(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int = 256, dropout: float = 0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, embed_dim), nn.LayerNorm(embed_dim), nn.GELU(),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DARECOPDNet(nn.Module):
    """Domain-Adaptive Robust Ensemble COPD Network.

    Contributions implemented for manuscript:
    - Tri-view representation: EfficientNet spectrogram, temporal attention, acoustic handcrafted.
    - Domain shortcut suppression: MixStyle + gradient reversal domain head.
    - Source-balanced robust objective hooks: embeddings returned for CORAL/MMD/IRM/GroupDRO.
    - Same backbone supports binary and multiclass classification.
    - Uncertainty/calibration compatible: logits exposed for temperature scaling.
    """
    def __init__(
        self,
        handcrafted_input_dim: int,
        num_classes: int,
        num_domains: int,
        task: TaskType = "binary",
        embed_dim: int = 256,
        projection_dim: int = 128,
        dropout: float = 0.30,
        pretrained_effnet: bool = True,
        grl_lambda: float = 1.0,
        use_effnet: bool = True,
    ):
        super().__init__()
        self.task, self.num_classes = task, int(num_classes)
        self.spec = EfficientNetEncoder(embed_dim, pretrained_effnet) if use_effnet else LightSpecEncoder(embed_dim)
        self.temp = TemporalEncoder(embed_dim, dropout=dropout)
        self.hand = HandcraftedEncoder(handcrafted_input_dim, embed_dim, dropout=dropout)
        self.branch_gate = nn.Sequential(nn.Linear(embed_dim * 3, embed_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(embed_dim, 3))
        self.fusion = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.LayerNorm(embed_dim), nn.GELU(), nn.Dropout(dropout))
        out_dim = 1 if task == "binary" else num_classes
        self.classifier = nn.Linear(embed_dim, out_dim)
        self.projection = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, projection_dim))
        self.grl = GradientReversal(grl_lambda)
        self.domain_classifier = nn.Sequential(nn.Linear(embed_dim, 128), nn.GELU(), nn.Dropout(dropout), nn.Linear(128, num_domains))

    def forward(self, spectrogram: torch.Tensor, handcrafted: torch.Tensor) -> dict[str, torch.Tensor]:
        z1 = self.spec(spectrogram)
        z2, att = self.temp(spectrogram)
        z3 = self.hand(handcrafted)
        gate = torch.softmax(self.branch_gate(torch.cat([z1, z2, z3], dim=1)), dim=1)
        z = gate[:, 0:1] * z1 + gate[:, 1:2] * z2 + gate[:, 2:3] * z3
        z = self.fusion(z)
        logits = self.classifier(z)
        if self.task == "binary":
            logits = logits.squeeze(1)
        return {
            "logits": logits,
            "embedding": z,
            "projection": F.normalize(self.projection(z), dim=1),
            "domain_logits": self.domain_classifier(self.grl(z)),
            "attention_weights": att,
            "branch_weights": gate,
            "branch_embeddings": torch.stack([z1, z2, z3], dim=1),
        }
