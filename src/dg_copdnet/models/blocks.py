from __future__ import annotations

import torch
import torch.nn as nn


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class GradientReversal(nn.Module):
    def __init__(self, lambda_: float = 1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


class AttentionPooling(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.attn = nn.Linear(input_dim, 1)

    def forward(self, x):
        score = self.attn(x).squeeze(-1)
        weight = torch.softmax(score, dim=1)
        pooled = torch.sum(x * weight.unsqueeze(-1), dim=1)
        return pooled, weight
