from __future__ import annotations

from pathlib import Path
import hashlib
import json
import math
import os
import warnings

import librosa
import numpy as np
import soundfile as sf

from dg_copdnet.config import AudioConfig, FeatureConfig
from dg_copdnet.utils.io import ensure_dir


def load_audio(path: str | Path, audio_cfg: AudioConfig):
    y, sr = librosa.load(path, sr=audio_cfg.sample_rate, mono=True)
    y = np.asarray(y, dtype=np.float32)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    target = audio_cfg.num_samples
    if len(y) < target:
        y = np.pad(y, (0, target - len(y)))
    else:
        y = y[:target]
    if audio_cfg.normalize_waveform:
        max_abs = np.max(np.abs(y)) + 1e-8
        y = y / max_abs
    y = np.clip(y, -1.0, 1.0)
    return y.astype(np.float32), audio_cfg.sample_rate


def logmel(y: np.ndarray, sr: int, cfg: AudioConfig) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=cfg.n_fft, hop_length=cfg.hop_length,
        n_mels=cfg.n_mels, fmin=cfg.f_min, fmax=cfg.f_max, power=2.0
    )
    mel = np.maximum(mel, 1e-10)
    mel_db = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
    mel_db = np.nan_to_num(mel_db, nan=0.0, posinf=0.0, neginf=0.0)
    if cfg.normalize_spectrogram:
        mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-8)
    mel_db = np.nan_to_num(mel_db, nan=0.0, posinf=0.0, neginf=0.0)
    return mel_db


def mfcc_default(y: np.ndarray, sr: int, n_mfcc: int = 20) -> np.ndarray:
    m = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return _stats_pool(m)


def mfcc_optimized(y: np.ndarray, sr: int, n_mfcc: int = 40) -> np.ndarray:
    m = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=1024, hop_length=256)
    delta = librosa.feature.delta(m)
    delta2 = librosa.feature.delta(m, order=2)
    feat = np.vstack([m, delta, delta2])
    return _stats_pool(feat)


def lpcc(y: np.ndarray, order: int = 16) -> np.ndarray:
    try:
        # Use float64 internally for better numerical stability.
        y64 = np.asarray(y, dtype=np.float64)
        if not np.all(np.isfinite(y64)) or np.max(np.abs(y64)) < 1e-8:
            return np.zeros(order * 2, dtype=np.float32)
        a = np.asarray(librosa.lpc(y64, order=order), dtype=np.float64)
        if not np.all(np.isfinite(a)) or abs(a[0]) < 1e-8:
            return np.zeros(order * 2, dtype=np.float32)
    except Exception:
        return np.zeros(order * 2, dtype=np.float32)

    cep = np.zeros(order, dtype=np.float64)
    cep[0] = -np.log(np.clip(np.abs(a[0]), 1e-8, None))
    try:
        for n in range(1, order):
            acc = 0.0
            for k in range(1, n):
                acc += (k / n) * cep[k] * a[n - k]
            cep[n] = -a[n] - acc
            if not np.isfinite(cep[n]):
                return np.zeros(order * 2, dtype=np.float32)
    except Exception:
        return np.zeros(order * 2, dtype=np.float32)

    cep = np.clip(np.nan_to_num(cep, nan=0.0, posinf=0.0, neginf=0.0), -1e3, 1e3)
    out = np.concatenate([cep, np.abs(cep)])
    return out.astype(np.float32)


def phonation(y: np.ndarray, sr: int) -> np.ndarray:
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    rms = librosa.feature.rms(y=y)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    f0, voiced_flag, _ = librosa.pyin(y, sr=sr, fmin=50, fmax=500)
    valid_f0 = f0[~np.isnan(f0)] if f0 is not None else np.array([])
    pitch_stats = np.array([
        float(np.mean(valid_f0)) if len(valid_f0) else 0.0,
        float(np.std(valid_f0)) if len(valid_f0) else 0.0,
        float(len(valid_f0) / max(1, len(f0))) if f0 is not None else 0.0,
    ], dtype=np.float32)
    pooled = np.concatenate([_basic_stats(zcr), _basic_stats(rms), _basic_stats(centroid), _basic_stats(rolloff), pitch_stats])
    return pooled.astype(np.float32)


def logmel_stats(y: np.ndarray, sr: int, cfg: AudioConfig) -> np.ndarray:
    mel = logmel(y, sr, cfg)
    return _stats_pool(mel)


def egemaps(y: np.ndarray, sr: int, enabled: bool = True) -> np.ndarray:
    if not enabled:
        return np.zeros(88, dtype=np.float32)
    try:
        import opensmile
        smile = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02, feature_level=opensmile.FeatureLevel.Functionals)
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, y, sr)
            df = smile.process_file(tmp.name)
        arr = df.to_numpy(dtype=np.float32).reshape(-1)
        return arr
    except Exception:
        return np.zeros(88, dtype=np.float32)


def vggish_features(y: np.ndarray, sr: int, enabled: bool = False) -> np.ndarray:
    if not enabled:
        return np.zeros(128, dtype=np.float32)
    try:
        import torch
        from torchvggish import vggish
        model = vggish()
        x = torch.from_numpy(y).float().unsqueeze(0)
        with torch.no_grad():
            emb = model.forward_audio(x, sr=sr)
        return emb.mean(dim=0).cpu().numpy().astype(np.float32)
    except Exception:
        return np.zeros(128, dtype=np.float32)


def get_feature_vector(y: np.ndarray, sr: int, audio_cfg: AudioConfig, feat_cfg: FeatureConfig, feature_names: list[str]) -> np.ndarray:
    chunks = []
    for name in feature_names:
        name = name.lower()
        if name == "mfcc_default":
            chunks.append(mfcc_default(y, sr))
        elif name == "mfcc_optimized":
            chunks.append(mfcc_optimized(y, sr))
        elif name == "lpcc":
            chunks.append(lpcc(y))
        elif name == "phonation":
            chunks.append(phonation(y, sr))
        elif name == "logmel_stats":
            chunks.append(logmel_stats(y, sr, audio_cfg))
        elif name == "egemaps":
            chunks.append(egemaps(y, sr, feat_cfg.use_opensmile_if_available))
        elif name == "vggish":
            chunks.append(vggish_features(y, sr, feat_cfg.use_vggish_if_available))
        else:
            raise ValueError(f"Unsupported feature set: {name}")
    if not chunks:
        return np.zeros(1, dtype=np.float32)
    feats = np.concatenate(chunks).astype(np.float32)
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    feats = np.clip(feats, -1e4, 1e4)
    return feats


def cached_feature_vector(audio_path: str | Path, audio_cfg: AudioConfig, feat_cfg: FeatureConfig, feature_names: list[str]) -> np.ndarray:
    cache_dir = ensure_dir(feat_cfg.cache_dir)
    key = {
        "path": str(audio_path),
        "feature_names": feature_names,
        "sr": audio_cfg.sample_rate,
        "duration": audio_cfg.duration_seconds,
    }
    cache_name = hashlib.md5(json.dumps(key, sort_keys=True).encode("utf-8")).hexdigest() + ".npy"
    cache_path = cache_dir / cache_name
    if cache_path.exists():
        cached = np.load(cache_path)
        if np.all(np.isfinite(cached)):
            return cached.astype(np.float32)
        try:
            cache_path.unlink()
        except OSError:
            pass
    y, sr = load_audio(audio_path, audio_cfg)
    feats = get_feature_vector(y, sr, audio_cfg, feat_cfg, feature_names)
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    np.save(cache_path, feats)
    return feats


def _basic_stats(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return np.array([float(np.mean(x)), float(np.std(x)), float(np.min(x)), float(np.max(x))], dtype=np.float32)


def _stats_pool(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float32)
    mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
    mean = mat.mean(axis=1)
    std = mat.std(axis=1)
    mn = mat.min(axis=1)
    mx = mat.max(axis=1)
    out = np.concatenate([mean, std, mn, mx]).astype(np.float32)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out
