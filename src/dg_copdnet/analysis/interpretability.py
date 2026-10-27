
from __future__ import annotations

from pathlib import Path
from typing import Optional
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import librosa.display

from dg_copdnet.config import AppConfig
from dg_copdnet.data.dataset import RespiratoryAudioDataset
from dg_copdnet.models.baselines import create_model
from dg_copdnet.utils.device import get_device
from dg_copdnet.utils.io import ensure_dir
from dg_copdnet.utils.plots import savefig


def _load_best_checkpoint(ckpt_path: str | Path, cfg: AppConfig, df: pd.DataFrame, label_col: str = 'label'):
    device = get_device(cfg.training.use_cuda)
    ckpt = torch.load(ckpt_path, map_location=device)
    source_to_idx = ckpt.get('source_to_idx', {s: i for i, s in enumerate(sorted(df['source'].astype(str).unique().tolist()))})
    hand_dim = ckpt.get('handcrafted_dim') or ckpt.get('hand_dim')
    if hand_dim is None:
        ds = RespiratoryAudioDataset(df.reset_index(drop=True), cfg.audio, cfg.features, cfg.augmentation, feature_names=cfg.features.handcrafted_sets, is_train=False, source_to_idx=source_to_idx, label_col=label_col)
        hand_dim = int(ds[0]['handcrafted'].shape[0])
    n_classes = ckpt.get('n_classes', 1)
    state = ckpt.get('model_state_dict') or ckpt.get('state_dict')
    # infer model type from path
    model_name = 'hybrid_dg_respnet'
    pstr = str(ckpt_path).lower()
    for cand in ['spec_cnn', 'effnet_only', 'crnn_only', 'handcrafted_mlp']:
        if cand in pstr:
            model_name = cand
            break
    num_domains = len(source_to_idx)
    model = create_model(model_name, hand_dim, cfg.model, num_domains, num_classes=n_classes).to(device)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model, device, source_to_idx, n_classes


def generate_saliency_examples(cfg: AppConfig, df: pd.DataFrame, ckpt_path: str | Path, output_dir: str | Path, label_col: str = 'label', max_examples_per_class: int = 2):
    outdir = ensure_dir(output_dir)
    model, device, source_to_idx, n_classes = _load_best_checkpoint(ckpt_path, cfg, df, label_col=label_col)
    ds = RespiratoryAudioDataset(df.reset_index(drop=True), cfg.audio, cfg.features, cfg.augmentation, feature_names=cfg.features.handcrafted_sets, is_train=False, source_to_idx=source_to_idx, label_col=label_col)
    classes = sorted(df[label_col].astype(int).unique().tolist())
    saved = []
    for cls in classes:
        idxs = df.index[df[label_col].astype(int) == cls].tolist()[:max_examples_per_class]
        for j, idx in enumerate(idxs, start=1):
            batch = ds[idx]
            spec = batch['spectrogram'].unsqueeze(0).to(device)
            hand = batch['handcrafted'].unsqueeze(0).to(device)
            spec.requires_grad_(True)
            out = model(spec, hand)
            logits = out['logits']
            if logits.ndim == 1 or (logits.ndim == 2 and logits.shape[1] == 1):
                score = logits.view(-1)[0]
            else:
                pred_class = int(batch['label_long'].item())
                score = logits[0, pred_class]
            model.zero_grad(set_to_none=True)
            if spec.grad is not None:
                spec.grad.zero_()
            score.backward(retain_graph=False)
            grad = spec.grad.detach().cpu().numpy()[0, 0]
            mel = spec.detach().cpu().numpy()[0, 0]
            heat = np.abs(grad)
            heat = heat / (heat.max() + 1e-8)
            plt.figure(figsize=(8, 6))
            plt.subplot(2, 1, 1)
            librosa.display.specshow(mel, x_axis='time', y_axis='mel', sr=cfg.audio.sample_rate, hop_length=cfg.audio.hop_length)
            plt.title(f'Input log-Mel | class={cls}')
            plt.colorbar()
            plt.subplot(2, 1, 2)
            librosa.display.specshow(heat, x_axis='time', y_axis='mel', sr=cfg.audio.sample_rate, hop_length=cfg.audio.hop_length, cmap='jet')
            plt.title('Saliency / gradient heatmap')
            plt.colorbar()
            fname = f'saliency_class_{cls}_example_{j}.png'
            savefig(Path(outdir) / fname)
            saved.append(fname)
            if 'attention_weights' in out and out['attention_weights'] is not None:
                att = out['attention_weights'].detach().cpu().numpy()[0]
                plt.figure(figsize=(8, 3))
                plt.plot(att)
                plt.xlabel('Time frames')
                plt.ylabel('Attention weight')
                plt.title(f'CRNN attention weights | class={cls}')
                fname2 = f'attention_class_{cls}_example_{j}.png'
                savefig(Path(outdir) / fname2)
                saved.append(fname2)
    return saved
