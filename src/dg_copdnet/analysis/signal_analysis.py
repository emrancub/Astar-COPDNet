
from __future__ import annotations

from pathlib import Path
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import librosa.display

from dg_copdnet.config import AppConfig
from dg_copdnet.data.features import load_audio, logmel
from dg_copdnet.utils.io import ensure_dir
from dg_copdnet.utils.plots import savefig


def _signal_stats(y, sr):
    duration = len(y) / sr
    rms = librosa.feature.rms(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    return {
        'duration_s': float(duration),
        'mean_abs_amp': float(np.mean(np.abs(y))),
        'std_amp': float(np.std(y)),
        'rms_mean': float(np.mean(rms)),
        'rms_std': float(np.std(rms)),
        'zcr_mean': float(np.mean(zcr)),
        'spectral_centroid_mean': float(np.mean(centroid)),
        'spectral_bandwidth_mean': float(np.mean(bandwidth)),
        'spectral_rolloff_mean': float(np.mean(rolloff)),
    }


def _safe_group_summary(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # keep only actual feature/stat columns, not duplicate group identifiers if numeric
    numeric_cols = [c for c in numeric_cols if c not in group_cols]
    if not numeric_cols:
        return pd.DataFrame()
    return df.groupby(group_cols, dropna=False)[numeric_cols].agg(['mean', 'std', 'median', 'min', 'max']).reset_index()


def _plot_dataset_examples(cfg: AppConfig, all_df: pd.DataFrame, outdir: Path):
    random.seed(cfg.seed)
    for source in ['ICBHI', 'KAUH', 'TR']:
        sdf = all_df[all_df['source'].astype(str).str.upper() == source].copy()
        if sdf.empty:
            continue
        # Prefer one negative and two positives when available.
        sampled_rows = []
        for label_value in [0, 1]:
            part = sdf[sdf['label'] == label_value]
            if not part.empty:
                sampled_rows.extend(list(part.sample(n=min(1 if label_value == 0 else 2, len(part)), random_state=cfg.seed).itertuples(index=False)))
        if not sampled_rows:
            sampled_rows = list(sdf.sample(n=min(3, len(sdf)), random_state=cfg.seed).itertuples(index=False))
        for i, r in enumerate(sampled_rows[:3], start=1):
            y, sr = load_audio(getattr(r, 'file_path'), cfg.audio)
            mel = logmel(y, sr, cfg.audio)
            dx = getattr(r, 'diagnosis_raw', '')
            plt.figure(figsize=(10, 3))
            librosa.display.waveshow(y, sr=sr)
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude')
            plt.title(f'{source} waveform example {i} | {dx}')
            savefig(outdir / f'{source.lower()}_waveform_example_{i}.png')

            plt.figure(figsize=(8, 4))
            librosa.display.specshow(mel, sr=sr, hop_length=cfg.audio.hop_length, x_axis='time', y_axis='mel')
            plt.colorbar()
            plt.title(f'{source} log-Mel example {i} | {dx}')
            savefig(outdir / f'{source.lower()}_logmel_example_{i}.png')


def _plot_summary_boxplots(stats_df: pd.DataFrame, outdir: Path):
    numeric_cols = [c for c in ['rms_mean', 'zcr_mean', 'spectral_centroid_mean', 'spectral_bandwidth_mean', 'spectral_rolloff_mean'] if c in stats_df.columns]
    for col in numeric_cols:
        plt.figure(figsize=(8, 4))
        stats_df.boxplot(column=col, by='source', grid=False)
        plt.title(f'{col} by dataset')
        plt.suptitle('')
        plt.ylabel(col)
        savefig(outdir / f'{col}_by_dataset.png')
        plt.figure(figsize=(8, 4))
        stats_df.boxplot(column=col, by='label', grid=False)
        plt.title(f'{col} by binary label')
        plt.suptitle('')
        plt.ylabel(col)
        savefig(outdir / f'{col}_by_binary_label.png')


def build_signal_report(cfg: AppConfig, all_df: pd.DataFrame):
    outdir = ensure_dir(Path(cfg.output_dir) / 'signal_analysis')
    rows = []
    for _, row in all_df.iterrows():
        try:
            y, sr = load_audio(row['file_path'], cfg.audio)
            rec = {
                'source': row['source'],
                'label': row['label'],
                'binary_label_name': 'COPD' if int(row['label']) == 1 else 'non-COPD',
                'diagnosis_raw': row.get('diagnosis_raw', ''),
            }
            if 'multiclass_label' in row.index:
                rec['multiclass_label'] = row['multiclass_label']
            rec.update(_signal_stats(y, sr))
            rows.append(rec)
        except Exception:
            continue
    stats_df = pd.DataFrame(rows)
    stats_df.to_csv(outdir / 'signal_stats_per_record.csv', index=False)

    # Numeric-only grouped summaries to avoid pandas aggregation errors on string columns.
    summary_source_label = _safe_group_summary(stats_df, ['source', 'label', 'binary_label_name'])
    summary_source_label.to_csv(outdir / 'signal_stats_summary_by_source_label.csv', index=False)

    if 'diagnosis_raw' in stats_df.columns:
        dx_summary = _safe_group_summary(stats_df, ['source', 'diagnosis_raw'])
        dx_summary.to_csv(outdir / 'signal_stats_summary_by_diagnosis.csv', index=False)

    _plot_dataset_examples(cfg, all_df, outdir)
    if not stats_df.empty:
        _plot_summary_boxplots(stats_df, outdir)
    return outdir
