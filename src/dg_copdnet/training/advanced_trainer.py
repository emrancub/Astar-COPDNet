
from __future__ import annotations

from pathlib import Path
from typing import Optional
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dg_copdnet.config import AppConfig
from dg_copdnet.data.dataset import RespiratoryAudioDataset
from dg_copdnet.data.metadata_builder import build_metadata_from_real_paths
from dg_copdnet.data.splits import make_grouped_folds
from dg_copdnet.models.baselines import create_model
from dg_copdnet.models.losses import SupervisedContrastiveLoss
from dg_copdnet.training.trainer import _make_optimizer
from dg_copdnet.utils.device import get_device
from dg_copdnet.utils.io import ensure_dir, save_json
from dg_copdnet.utils.metrics import compute_metrics, compute_multiclass_metrics
from dg_copdnet.utils.modeling import count_parameters
from dg_copdnet.utils.seed import seed_everything


def _maybe_build_metadata(cfg: AppConfig):
    if cfg.dataset_build and cfg.dataset_build.auto_build:
        internal_csv = Path(cfg.internal_csv)
        external_csv = Path(cfg.external_csv)
        all_csv = Path(cfg.all_csv)
        if not (internal_csv.exists() and external_csv.exists() and all_csv.exists()):
            build_metadata_from_real_paths(cfg)


def _dataset_loader(df, cfg, source_to_idx, is_train, label_col, feature_names=None):
    ds = RespiratoryAudioDataset(df, cfg.audio, cfg.features, cfg.augmentation, feature_names=feature_names or cfg.features.handcrafted_sets, is_train=is_train, source_to_idx=source_to_idx, label_col=label_col)
    return ds, DataLoader(ds, batch_size=cfg.training.batch_size, shuffle=is_train, num_workers=cfg.training.num_workers, pin_memory=cfg.training.use_cuda, drop_last=is_train)


def _run_epoch(model, loader, optimizer, scaler, device, cfg, train: bool, multiclass: bool):
    model.train(train)
    losses=[]; y_true=[]; probs=[]; preds=[]; embeds=[]; branch_weights=[]
    use_domain = hasattr(model, 'base') or hasattr(model, 'domain_classifier')
    for batch in loader:
        x_spec=batch['spectrogram'].to(device, non_blocking=True)
        x_hand=batch['handcrafted'].to(device, non_blocking=True)
        y_float=batch['label'].to(device, non_blocking=True)
        y_long=batch['label_long'].to(device, non_blocking=True)
        domain=batch['domain'].to(device, non_blocking=True)
        with torch.set_grad_enabled(train):
            with torch.amp.autocast(device_type='cuda', enabled=(cfg.training.mixed_precision and device.type=='cuda')):
                out=model(x_spec, x_hand)
                logits=torch.nan_to_num(out['logits'], nan=0.0, posinf=0.0, neginf=0.0)
                if multiclass:
                    cls_loss=nn.CrossEntropyLoss()(logits, y_long)
                    prob=torch.softmax(logits, dim=1)
                    pred=prob.argmax(dim=1)
                else:
                    logits=logits.view(-1)
                    cls_loss=nn.BCEWithLogitsLoss()(logits, y_float)
                    prob=torch.sigmoid(logits)
                    pred=(prob>=0.5).long()
                loss=cls_loss
                if 'projection' in out and not multiclass:
                    try:
                        loss = loss + cfg.loss.lambda_supcon * SupervisedContrastiveLoss(cfg.loss.temperature)(torch.nan_to_num(out['projection'], nan=0.0), y_long)
                    except Exception:
                        pass
                if use_domain and 'domain_logits' in out:
                    try:
                        loss = loss + cfg.loss.lambda_domain * nn.CrossEntropyLoss()(torch.nan_to_num(out['domain_logits'], nan=0.0), domain)
                    except Exception:
                        pass
            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.gradient_clip_norm)
                scaler.step(optimizer); scaler.update()
        losses.append(float(loss.detach().cpu()))
        y_true.extend(y_long.detach().cpu().numpy().tolist())
        probs.append(prob.detach().cpu().numpy())
        preds.extend(pred.detach().cpu().numpy().tolist())
        embeds.append(out['embedding'].detach().cpu().numpy())
        if 'branch_weights' in out:
            branch_weights.append(out['branch_weights'].detach().cpu().numpy())
    prob_arr=np.concatenate(probs, axis=0) if probs else np.array([])
    emb_arr=np.concatenate(embeds, axis=0) if embeds else np.empty((0,))
    if multiclass:
        metrics=compute_multiclass_metrics(y_true, prob_arr)
    else:
        metrics=compute_metrics(y_true, prob_arr)
    metrics['loss']=float(np.mean(losses)) if losses else 0.0
    if branch_weights:
        bw=np.concatenate(branch_weights, axis=0)
        for i in range(bw.shape[1]):
            metrics[f'branch_weight_{i+1}']=float(np.mean(bw[:,i]))
    return metrics, {'y_true': np.asarray(y_true), 'probs': prob_arr, 'preds': np.asarray(preds), 'embeddings': emb_arr}


def train_cv_generic(cfg: AppConfig, df: pd.DataFrame, model_name: str, label_col: str = 'label', multiclass: bool = False, feature_names: Optional[list[str]] = None, optimizer_name: str = 'adamw', output_subdir: str = 'advanced'):
    seed_everything(cfg.seed)
    device=get_device(cfg.training.use_cuda)
    source_to_idx={s:i for i,s in enumerate(sorted(df['source'].astype(str).unique().tolist()))}
    out_dir=ensure_dir(Path(cfg.output_dir)/output_subdir/model_name)
    results=[]
    for fold, tr_idx, va_idx in make_grouped_folds(df, cfg.training.num_folds, cfg.seed):
        tr_df=df.iloc[tr_idx].reset_index(drop=True); va_df=df.iloc[va_idx].reset_index(drop=True)
        tr_ds, tr_loader=_dataset_loader(tr_df, cfg, source_to_idx, True, label_col, feature_names)
        va_ds, va_loader=_dataset_loader(va_df, cfg, source_to_idx, False, label_col, feature_names)
        hand_dim=int(tr_ds[0]['handcrafted'].shape[0])
        n_classes=int(df[label_col].nunique()) if multiclass else 1
        model=create_model(model_name, hand_dim, cfg.model, len(source_to_idx), num_classes=n_classes).to(device)
        optimizer=_make_optimizer(optimizer_name, model, cfg.training.learning_rate, cfg.training.weight_decay)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.training.epochs)
        scaler=torch.amp.GradScaler('cuda', enabled=(cfg.training.mixed_precision and device.type=='cuda'))
        best_score=-1e9; best=None; history=[]
        for epoch in range(1, cfg.training.epochs+1):
            tr_metrics,_=_run_epoch(model,tr_loader,optimizer,scaler,device,cfg,True,multiclass)
            va_metrics,payload=_run_epoch(model,va_loader,None,scaler,device,cfg,False,multiclass)
            scheduler.step()
            row={'fold':fold,'epoch':epoch, **{f'train_{k}':v for k,v in tr_metrics.items() if isinstance(v, (int, float, np.integer, np.floating))}, **{f'val_{k}':v for k,v in va_metrics.items() if isinstance(v, (int, float, np.integer, np.floating))}}
            history.append(row)
            score = va_metrics['macro_ovr_auc'] if multiclass else va_metrics[cfg.training.early_stop_metric]
            if score > best_score:
                best_score=score
                best={'state_dict':model.state_dict(),'payload':payload,'metrics':va_metrics,'param_count':count_parameters(model),'hand_dim':hand_dim,'n_classes':n_classes,'source_to_idx':source_to_idx}
        fold_dir=ensure_dir(out_dir/f'fold_{fold}')
        pd.DataFrame(history).to_csv(fold_dir/'history.csv', index=False)
        torch.save(best, fold_dir/'best.pt')
        np.save(fold_dir/'val_embeddings.npy', best['payload']['embeddings'])
        np.save(fold_dir/'val_probs.npy', best['payload']['probs'])
        np.save(fold_dir/'val_y_true.npy', best['payload']['y_true'])
        results.append({'fold':fold, **{k:v for k,v in best['metrics'].items() if isinstance(v, (int, float, np.integer, np.floating))}, 'param_count':best['param_count']})
        save_json(best['metrics'], fold_dir/'best_metrics.json')
    dfres=pd.DataFrame(results)
    dfres.to_csv(out_dir/'summary.csv', index=False)
    save_json({c:{'mean':float(pd.to_numeric(dfres[c], errors='coerce').mean()),'std':float(pd.to_numeric(dfres[c], errors='coerce').std(ddof=1) if pd.to_numeric(dfres[c], errors='coerce').notna().sum()>1 else 0.0)} for c in dfres.columns if c!='fold' and pd.to_numeric(dfres[c], errors='coerce').notna().any()}, out_dir/'summary_stats.json')
    return dfres
