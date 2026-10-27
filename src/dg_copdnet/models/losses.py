from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if len(features) < 2:
            return features.new_tensor(0.0)
        features = F.normalize(features, dim=1)
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(features.device)
        logits = torch.div(torch.matmul(features, features.T), self.temperature)
        logits = logits - torch.max(logits, dim=1, keepdim=True)[0].detach()
        logits_mask = torch.ones_like(mask) - torch.eye(mask.shape[0], device=mask.device)
        mask = mask * logits_mask
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-12)
        loss = -mean_log_prob_pos.mean()
        return loss
