from __future__ import annotations
import json
from pathlib import Path
from typing import Literal
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.utils.class_weight import compute_class_weight

from dg_copdnet.config import AppConfig
from dg_copdnet.data.dataset import RespiratoryAudioDataset
from dg_copdnet.models.dare_copdnet import DARECOPDNet
from dg_copdnet.training.domain_losses import supervised_contrastive_loss, coral_loss, irm_penalty, groupdro_loss
from dg_copdnet.utils.metrics import compute_metrics, compute_multiclass_metrics
from dg_copdnet.utils.io import ensure_dir, save_json

Task = Literal["binary", "multiclass"]


def _feature_dim(df: pd.DataFrame, cfg: AppConfig, feature_names: list[str]) -> int:
    from dg_copdnet.data.features import cached_feature_vector
    return int(len(cached_feature_vector(df.iloc[0].file_path, cfg.audio, cfg.features, feature_names)))


def _loader(df: pd.DataFrame, cfg: AppConfig, feature_names: list[str], source_to_idx: dict[str, int], label_col: str, is_train: bool):
    ds = RespiratoryAudioDataset(df, cfg.audio, cfg.features, cfg.augmentation, feature_names, is_train, source_to_idx, label_col=label_col)
    sampler = None
    shuffle = is_train
    if is_train and len(df[label_col].unique()) > 1:
        counts = df[label_col].value_counts().to_dict()
        weights = df[label_col].map(lambda x: 1.0 / counts[x]).to_numpy(dtype=np.float64)
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        shuffle = False
    return DataLoader(ds, batch_size=cfg.training.batch_size, shuffle=shuffle, sampler=sampler,
                      num_workers=cfg.training.num_workers, pin_memory=torch.cuda.is_available())


def _losses(out, batch, task: Task, num_domains: int, class_weights, lambdas: dict):
    y = batch["label_long"] if task == "multiclass" else batch["label"]
    domains = batch["domain"]
    logits = out["logits"]
    if task == "binary":
        per = F.binary_cross_entropy_with_logits(logits, y.float(), reduction="none")
    else:
        per = F.cross_entropy(logits, y.long(), weight=class_weights, reduction="none")
    cls = groupdro_loss(per, domains, num_domains, eta=lambdas.get("groupdro_eta", 0.05)) if lambdas.get("groupdro", 1) else per.mean()
    dom = F.cross_entropy(out["domain_logits"], domains.long()) if num_domains > 1 else logits.new_tensor(0.0)
    supcon = supervised_contrastive_loss(out["projection"], y.long(), lambdas.get("temperature", 0.07)) if out["projection"].size(0) > 2 else logits.new_tensor(0.0)
    coral = coral_loss(out["embedding"], domains)
    irm = irm_penalty(logits, y, task)
    total = cls + lambdas.get("domain", 0.2) * dom + lambdas.get("supcon", 0.1) * supcon + lambdas.get("coral", 0.1) * coral + lambdas.get("irm", 0.01) * irm
    return total, {"loss": float(total.detach().cpu()), "cls": float(cls.detach().cpu()), "domain": float(dom.detach().cpu()), "supcon": float(supcon.detach().cpu()), "coral": float(coral.detach().cpu()), "irm": float(irm.detach().cpu())}


@torch.no_grad()
def predict(model, loader, task: Task, device):
    model.eval(); ys=[]; probs=[]; domains=[]; patients=[]; sources=[]; branch=[]
    for batch in loader:
        spec=batch["spectrogram"].to(device); hand=batch["handcrafted"].to(device)
        out=model(spec, hand)
        if task == "binary":
            p=torch.sigmoid(out["logits"]).detach().cpu().numpy()
            y=batch["label"].numpy().astype(int)
        else:
            p=torch.softmax(out["logits"], dim=1).detach().cpu().numpy()
            y=batch["label_long"].numpy().astype(int)
        ys.append(y); probs.append(p); domains += batch["domain"].numpy().astype(int).tolist(); patients += list(batch["patient_id"]); sources += list(batch["source"])
        branch.append(out["branch_weights"].detach().cpu().numpy())
    return {"y": np.concatenate(ys), "prob": np.concatenate(probs), "domains": domains, "patients": patients, "sources": sources, "branch_weights": np.concatenate(branch) if branch else np.empty((0,3))}


def _select_threshold(y, prob):
    # Choose threshold only on validation data by max Youden/balanced accuracy.
    candidates = np.linspace(0.05, 0.95, 181)
    best_t, best = 0.5, -1
    for t in candidates:
        m = compute_metrics(y, prob, threshold=float(t))
        if m["balanced_accuracy"] > best:
            best, best_t = m["balanced_accuracy"], float(t)
    return best_t


def train_one_split(cfg: AppConfig, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame | None, out_dir: str | Path, task: Task, label_col: str, num_classes: int, seed: int = 42):
    torch.manual_seed(seed); np.random.seed(seed)
    out_dir = ensure_dir(out_dir)
    feature_names = cfg.experiments.selected_feature_triplet or cfg.features.handcrafted_sets
    sources = sorted(pd.concat([train_df, val_df] + ([] if test_df is None else [test_df]))["source"].astype(str).unique())
    source_to_idx = {s:i for i,s in enumerate(sources)}
    device = torch.device("cuda" if cfg.training.use_cuda and torch.cuda.is_available() else "cpu")
    hdim = _feature_dim(train_df, cfg, feature_names)
    model = DARECOPDNet(hdim, num_classes=num_classes, num_domains=len(source_to_idx), task=task,
                        embed_dim=cfg.model.embed_dim, projection_dim=cfg.model.projection_dim, dropout=cfg.model.dropout,
                        pretrained_effnet=cfg.model.pretrained_effnet, grl_lambda=cfg.model.grl_lambda).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)
    train_loader = _loader(train_df, cfg, feature_names, source_to_idx, label_col, True)
    val_loader = _loader(val_df, cfg, feature_names, source_to_idx, label_col, False)
    test_loader = _loader(test_df, cfg, feature_names, source_to_idx, label_col, False) if test_df is not None else None
    class_weights = None
    if task == "multiclass":
        classes = np.arange(num_classes)
        present = np.unique(train_df[label_col].astype(int).to_numpy())
        cw = np.ones(num_classes, dtype=np.float32)
        if len(present) > 1:
            vals = compute_class_weight("balanced", classes=present, y=train_df[label_col].astype(int).to_numpy())
            for c, v in zip(present, vals): cw[int(c)] = float(v)
        class_weights = torch.tensor(cw, dtype=torch.float32, device=device)
    lambdas = {"domain": cfg.loss.lambda_domain, "supcon": cfg.loss.lambda_supcon, "temperature": cfg.loss.temperature, "coral": 0.10, "irm": 0.01, "groupdro": 1}
    best_score = -np.inf; best_epoch = -1; patience = 0; history=[]; best_path = Path(out_dir) / "best_model.pt"
    for epoch in range(int(cfg.training.epochs)):
        model.train(); tr_losses=[]
        for batch in train_loader:
            batch = {k:(v.to(device) if torch.is_tensor(v) else v) for k,v in batch.items()}
            opt.zero_grad(set_to_none=True)
            out = model(batch["spectrogram"], batch["handcrafted"])
            loss, parts = _losses(out, batch, task, len(source_to_idx), class_weights, lambdas)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.gradient_clip_norm); opt.step()
            tr_losses.append(parts)
        val_pred = predict(model, val_loader, task, device)
        if task == "binary":
            thr = _select_threshold(val_pred["y"], val_pred["prob"])
            val_metrics = compute_metrics(val_pred["y"], val_pred["prob"], threshold=thr)
            score = val_metrics.get("auc", np.nan)
        else:
            thr = None; val_metrics = compute_multiclass_metrics(val_pred["y"], val_pred["prob"]); score = val_metrics.get("macro_ovr_auc", val_metrics["macro_f1"])
        row = {"epoch": epoch, "threshold": thr, "val_score": float(score), **{f"val_{k}": v for k,v in val_metrics.items() if isinstance(v,(int,float))}}
        history.append(row)
        if np.isfinite(score) and score > best_score:
            best_score = float(score); best_epoch = epoch; patience = 0
            torch.save({"model": model.state_dict(), "threshold": thr, "source_to_idx": source_to_idx, "feature_names": feature_names, "task": task, "label_col": label_col, "num_classes": num_classes}, best_path)
        else:
            patience += 1
            if patience >= cfg.training.patience: break
    pd.DataFrame(history).to_csv(Path(out_dir) / "history.csv", index=False)
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    results = {"best_epoch": best_epoch, "best_validation_score": best_score, "validation_threshold": ckpt.get("threshold")}
    for name, loader in [("validation", val_loader), ("external_test", test_loader)]:
        if loader is None: continue
        pred = predict(model, loader, task, device)
        if task == "binary":
            metrics = compute_metrics(pred["y"], pred["prob"], threshold=float(ckpt.get("threshold") or 0.5))
        else:
            metrics = compute_multiclass_metrics(pred["y"], pred["prob"])
        results[name] = metrics
        np.savez(Path(out_dir) / f"{name}_predictions.npz", y=pred["y"], prob=pred["prob"], branch_weights=pred["branch_weights"], sources=np.array(pred["sources"]), patients=np.array(pred["patients"]))
    save_json(results, Path(out_dir) / "metrics.json")
    return results
