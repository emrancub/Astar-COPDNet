from __future__ import annotations

from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from dg_copdnet.utils.metrics import compute_metrics


def run_epoch(model, loader, optimizer, scaler, device, loss_fns, cfg, train: bool):
    model.train(train)
    losses = []
    y_true, y_prob = [], []
    branch_weights = []
    for batch in tqdm(loader, leave=False):
        x_spec = batch["spectrogram"].to(device, non_blocking=True)
        x_hand = batch["handcrafted"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)
        domain = batch["domain"].to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            with torch.amp.autocast(device_type="cuda", enabled=(cfg.training.mixed_precision and device.type == "cuda")):
                out = model(x_spec, x_hand)
                out["logits"] = torch.nan_to_num(out["logits"], nan=0.0, posinf=0.0, neginf=0.0)
                out["projection"] = torch.nan_to_num(out["projection"], nan=0.0, posinf=0.0, neginf=0.0)
                out["domain_logits"] = torch.nan_to_num(out["domain_logits"], nan=0.0, posinf=0.0, neginf=0.0)
                cls_loss = loss_fns["bce"](out["logits"], y)
                supcon_loss = loss_fns["supcon"](out["projection"], y.long())
                dom_loss = loss_fns["domain"](out["domain_logits"], domain)
                loss = cls_loss + cfg.loss.lambda_supcon * supcon_loss + cfg.loss.lambda_domain * dom_loss
                loss = torch.nan_to_num(loss, nan=0.0, posinf=1e6, neginf=1e6)

            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.gradient_clip_norm)
                scaler.step(optimizer)
                scaler.update()

        loss_value = float(loss.detach().cpu())
        if not np.isfinite(loss_value):
            continue
        losses.append(loss_value)
        batch_prob = torch.sigmoid(out["logits"]).detach().cpu().numpy()
        batch_prob = np.nan_to_num(batch_prob, nan=0.5, posinf=1.0, neginf=0.0)
        batch_prob = np.clip(batch_prob, 0.0, 1.0)
        y_true.extend(y.detach().cpu().numpy().tolist())
        y_prob.extend(batch_prob.tolist())
        branch_weights.append(out["branch_weights"].detach().cpu().numpy())

    metrics = compute_metrics(y_true, y_prob)
    metrics["loss"] = float(np.mean(losses)) if losses else 0.0
    if branch_weights:
        bw = np.concatenate(branch_weights, axis=0)
        for i in range(bw.shape[1]):
            metrics[f"branch_weight_{i+1}"] = float(np.mean(bw[:, i]))
    return metrics
