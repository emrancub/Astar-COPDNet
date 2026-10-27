from __future__ import annotations

from pathlib import Path
from typing import Optional
import numbers
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dg_copdnet.config import AppConfig
from dg_copdnet.data.dataset import RespiratoryAudioDataset
from dg_copdnet.data.metadata_builder import build_metadata_from_real_paths
from dg_copdnet.data.splits import make_grouped_folds, make_grouped_holdout
from dg_copdnet.models.hybrid_model import HybridDGRespNet
from dg_copdnet.models.losses import SupervisedContrastiveLoss
from dg_copdnet.training.engine import run_epoch
from dg_copdnet.utils.device import get_device
from dg_copdnet.utils.io import ensure_dir, save_json
from dg_copdnet.utils.modeling import count_parameters
from dg_copdnet.utils.seed import seed_everything


def _maybe_build_metadata(cfg: AppConfig):
    if cfg.dataset_build and cfg.dataset_build.auto_build:
        internal_csv = Path(cfg.internal_csv)
        external_csv = Path(cfg.external_csv)
        all_csv = Path(cfg.all_csv)
        if not (internal_csv.exists() and external_csv.exists() and all_csv.exists()):
            build_metadata_from_real_paths(cfg)


def _make_optimizer(name: str, model, lr: float, weight_decay: float):
    name = name.lower()
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "rmsprop":
        return torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9, nesterov=True)
    raise ValueError(f"Unsupported optimizer: {name}")


def _scalar_metrics_only(metrics: dict) -> dict:
    out = {}
    for k, v in metrics.items():
        if isinstance(v, numbers.Number):
            out[k] = float(v) if isinstance(v, float) else v
    return out


def _make_summary_stats(df: pd.DataFrame, prefix: str) -> dict:
    stats = {}
    for col in [c for c in df.columns if c.startswith(prefix)]:
        s = pd.to_numeric(df[col], errors="coerce")
        if s.notna().any():
            stats[col] = {
                "mean": float(s.mean()),
                "std": float(s.std(ddof=1) if len(s.dropna()) > 1 else 0.0),
            }
    return stats


def _dataset_and_loader(df, cfg, source_to_idx, is_train: bool):
    dataset = RespiratoryAudioDataset(
        df=df,
        audio_cfg=cfg.audio,
        feat_cfg=cfg.features,
        aug_cfg=cfg.augmentation,
        feature_names=cfg.features.handcrafted_sets,
        is_train=is_train,
        source_to_idx=source_to_idx,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=is_train,
        num_workers=cfg.training.num_workers,
        pin_memory=cfg.training.use_cuda,
        drop_last=is_train,
    )
    return dataset, loader


def train_standard_cv(cfg: AppConfig, optimizer_name: str = "adamw", subset_df: Optional[pd.DataFrame] = None, save_prefix: str = "standard_cv"):
    seed_everything(cfg.seed)
    _maybe_build_metadata(cfg)
    device = get_device(cfg.training.use_cuda)
    df = subset_df.copy() if subset_df is not None else pd.read_csv(cfg.internal_csv)
    source_to_idx = {s: i for i, s in enumerate(sorted(df["source"].astype(str).unique().tolist()))}
    out_dir = ensure_dir(Path(cfg.output_dir) / save_prefix)
    cv_rows = []
    hand_dim = None

    for fold, tr_idx, va_idx in make_grouped_folds(df, cfg.training.num_folds, cfg.seed):
        tr_df = df.iloc[tr_idx].reset_index(drop=True)
        va_df = df.iloc[va_idx].reset_index(drop=True)
        train_set, train_loader = _dataset_and_loader(tr_df, cfg, source_to_idx, is_train=True)
        val_set, val_loader = _dataset_and_loader(va_df, cfg, source_to_idx, is_train=False)
        if hand_dim is None:
            hand_dim = int(train_set[0]["handcrafted"].shape[0])

        model = HybridDGRespNet(hand_dim, cfg.model, num_domains=len(source_to_idx)).to(device)
        optimizer = _make_optimizer(optimizer_name, model, cfg.training.learning_rate, cfg.training.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.5)
        scaler = torch.amp.GradScaler("cuda", enabled=(cfg.training.mixed_precision and device.type == "cuda"))
        loss_fns = {
            "bce": nn.BCEWithLogitsLoss(),
            "supcon": SupervisedContrastiveLoss(cfg.loss.temperature),
            "domain": nn.CrossEntropyLoss(),
        }
        best_score = float("-inf")
        best_state = None
        wait = 0
        history = []

        for epoch in range(1, cfg.training.epochs + 1):
            train_metrics = run_epoch(model, train_loader, optimizer, scaler, device, loss_fns, cfg, train=True)
            val_metrics = run_epoch(model, val_loader, optimizer=None, scaler=scaler, device=device, loss_fns=loss_fns, cfg=cfg, train=False)
            scheduler.step(float(val_metrics[cfg.training.early_stop_metric]))

            row = {"fold": fold, "epoch": epoch}
            row.update({f"train_{k}": v for k, v in _scalar_metrics_only(train_metrics).items()})
            row.update({f"val_{k}": v for k, v in _scalar_metrics_only(val_metrics).items()})
            history.append(row)

            score = float(val_metrics[cfg.training.early_stop_metric])
            if score > best_score:
                best_score = score
                wait = 0
                best_state = {
                    "model_state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "fold": fold,
                    "source_to_idx": source_to_idx,
                    "handcrafted_dim": hand_dim,
                    "param_count": count_parameters(model),
                    "best_val_metrics": val_metrics,
                }
            else:
                wait += 1
                if wait >= cfg.training.patience:
                    break

        fold_dir = ensure_dir(out_dir / f"fold_{fold}")
        hist_df = pd.DataFrame(history)
        hist_df.to_csv(fold_dir / "history.csv", index=False)
        if best_state is not None:
            torch.save(best_state, fold_dir / "best.pt")
            save_json(best_state["best_val_metrics"], fold_dir / "best_val_metrics.json")

        best_idx = hist_df[f"val_{cfg.training.early_stop_metric}"].astype(float).idxmax()
        best_row = hist_df.loc[best_idx].to_dict()
        cv_rows.append(best_row)

    cv_df = pd.DataFrame(cv_rows)
    cv_df.to_csv(out_dir / "cv_summary.csv", index=False)
    stats = _make_summary_stats(cv_df, "val_")
    save_json(stats, out_dir / "cv_summary_stats.json")
    return cv_df, stats


def evaluate_checkpoint_on_df(cfg: AppConfig, checkpoint_path: str, eval_df: pd.DataFrame, save_dir_name: str):
    device = get_device(cfg.training.use_cuda)
    source_to_idx = {s: i for i, s in enumerate(sorted(eval_df["source"].astype(str).unique().tolist()))}
    temp_dataset, temp_loader = _dataset_and_loader(eval_df.reset_index(drop=True), cfg, source_to_idx, is_train=False)
    hand_dim = int(temp_dataset[0]["handcrafted"].shape[0])
    ckpt = torch.load(checkpoint_path, map_location=device)
    domain_out = ckpt["model_state_dict"]["domain_classifier.3.weight"].shape[0] if "domain_classifier.3.weight" in ckpt["model_state_dict"] else max(1, len(source_to_idx))
    model = HybridDGRespNet(hand_dim, cfg.model, num_domains=int(domain_out)).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    loss_fns = {"bce": nn.BCEWithLogitsLoss(), "supcon": SupervisedContrastiveLoss(cfg.loss.temperature), "domain": nn.CrossEntropyLoss()}
    metrics = run_epoch(model, temp_loader, optimizer=None, scaler=scaler, device=device, loss_fns=loss_fns, cfg=cfg, train=False)
    out_dir = ensure_dir(Path(cfg.output_dir) / save_dir_name)
    save_json(metrics, out_dir / "metrics.json")
    return metrics


def grouped_holdout_experiment(cfg: AppConfig, subset_df: pd.DataFrame, optimizer_name: str = "adamw", test_size: float = 0.2, save_prefix: str = "grouped_holdout"):
    seed_everything(cfg.seed)
    device = get_device(cfg.training.use_cuda)
    tr_idx, te_idx = make_grouped_holdout(subset_df, test_size=test_size, seed=cfg.seed)
    tr_df = subset_df.iloc[tr_idx].reset_index(drop=True)
    te_df = subset_df.iloc[te_idx].reset_index(drop=True)
    source_to_idx = {s: i for i, s in enumerate(sorted(subset_df["source"].astype(str).unique().tolist()))}
    train_set, train_loader = _dataset_and_loader(tr_df, cfg, source_to_idx, is_train=True)
    test_set, test_loader = _dataset_and_loader(te_df, cfg, source_to_idx, is_train=False)

    hand_dim = int(train_set[0]["handcrafted"].shape[0])
    model = HybridDGRespNet(hand_dim, cfg.model, num_domains=len(source_to_idx)).to(device)
    optimizer = _make_optimizer(optimizer_name, model, cfg.training.learning_rate, cfg.training.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.5)
    scaler = torch.amp.GradScaler("cuda", enabled=(cfg.training.mixed_precision and device.type == "cuda"))
    loss_fns = {"bce": nn.BCEWithLogitsLoss(), "supcon": SupervisedContrastiveLoss(cfg.loss.temperature), "domain": nn.CrossEntropyLoss()}

    best_score = float("-inf")
    best_state = None
    wait = 0
    history = []
    for epoch in range(1, cfg.training.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, scaler, device, loss_fns, cfg, train=True)
        test_metrics = run_epoch(model, test_loader, optimizer=None, scaler=scaler, device=device, loss_fns=loss_fns, cfg=cfg, train=False)
        scheduler.step(float(test_metrics[cfg.training.early_stop_metric]))
        row = {"epoch": epoch}
        row.update({f"train_{k}": v for k, v in _scalar_metrics_only(train_metrics).items()})
        row.update({f"test_{k}": v for k, v in _scalar_metrics_only(test_metrics).items()})
        history.append(row)
        score = float(test_metrics[cfg.training.early_stop_metric])
        if score > best_score:
            best_score = score
            wait = 0
            best_state = {"model_state_dict": model.state_dict(), "param_count": count_parameters(model), "best_test_metrics": test_metrics}
        else:
            wait += 1
            if wait >= cfg.training.patience:
                break
    out_dir = ensure_dir(Path(cfg.output_dir) / save_prefix)
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
    if best_state is not None:
        torch.save(best_state, out_dir / "best.pt")
        save_json(best_state["best_test_metrics"], out_dir / "best_test_metrics.json")
    return pd.DataFrame(history)
