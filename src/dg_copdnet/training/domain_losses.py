from __future__ import annotations
import torch
import torch.nn.functional as F


def supervised_contrastive_loss(z: torch.Tensor, y: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    z = F.normalize(z, dim=1)
    y = y.view(-1, 1)
    mask = torch.eq(y, y.T).float().to(z.device)
    logits = torch.matmul(z, z.T) / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    logits_mask = torch.ones_like(mask) - torch.eye(mask.size(0), device=z.device)
    mask = mask * logits_mask
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
    denom = mask.sum(1).clamp_min(1.0)
    loss = -(mask * log_prob).sum(1) / denom
    return loss.mean()


def coral_loss(z: torch.Tensor, domains: torch.Tensor) -> torch.Tensor:
    """Multi-source CORAL alignment on embeddings."""
    unique = domains.unique()
    if unique.numel() < 2:
        return z.new_tensor(0.0)
    covs = []
    for d in unique:
        xd = z[domains == d]
        if xd.size(0) < 2:
            continue
        xc = xd - xd.mean(0, keepdim=True)
        covs.append((xc.T @ xc) / (xd.size(0) - 1))
    if len(covs) < 2:
        return z.new_tensor(0.0)
    loss, n = z.new_tensor(0.0), 0
    for i in range(len(covs)):
        for j in range(i + 1, len(covs)):
            loss = loss + (covs[i] - covs[j]).pow(2).mean()
            n += 1
    return loss / max(n, 1)


def irm_penalty(logits: torch.Tensor, y: torch.Tensor, task: str) -> torch.Tensor:
    """Invariant risk minimization penalty for binary/multiclass logits."""
    scale = torch.tensor(1.0, device=logits.device, requires_grad=True)
    if task == "binary":
        loss = F.binary_cross_entropy_with_logits(logits * scale, y.float())
    else:
        loss = F.cross_entropy(logits * scale, y.long())
    grad = torch.autograd.grad(loss, [scale], create_graph=True)[0]
    return grad.pow(2)


def groupdro_loss(per_sample_loss: torch.Tensor, domains: torch.Tensor, num_domains: int, eta: float = 0.05) -> torch.Tensor:
    """Smooth max over source-specific losses. Stateless version for simplicity."""
    vals = []
    for d in range(num_domains):
        m = domains == d
        if m.any():
            vals.append(per_sample_loss[m].mean())
    if not vals:
        return per_sample_loss.mean()
    v = torch.stack(vals)
    w = torch.softmax(v.detach() / eta, dim=0)
    return (w * v).sum()
