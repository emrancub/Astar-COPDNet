from __future__ import annotations

import random
import librosa
import numpy as np
import torch
import torchaudio

from dg_copdnet.config import AugmentationConfig


class HybridAugment:
    def __init__(self, cfg: AugmentationConfig):
        self.cfg = cfg
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=24)
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=16)

    def waveform(self, y: np.ndarray, sr: int) -> np.ndarray:
        if not self.cfg.enable_waveform_aug:
            return y
        x = y.copy()
        if random.random() < self.cfg.gaussian_noise_prob:
            sigma = np.random.uniform(0.001, 0.015)
            x = x + np.random.normal(0.0, sigma, size=x.shape).astype(np.float32)
        if random.random() < self.cfg.gain_prob:
            gain = np.random.uniform(0.8, 1.2)
            x = x * gain
        if random.random() < self.cfg.shift_prob:
            shift = np.random.randint(-int(0.1 * len(x)), int(0.1 * len(x)))
            x = np.roll(x, shift)
        if random.random() < self.cfg.time_stretch_prob:
            rate = np.random.uniform(0.9, 1.1)
            stretched = librosa.effects.time_stretch(x, rate=rate)
            x = _fit_length(stretched, len(y))
        if random.random() < self.cfg.pitch_shift_prob:
            steps = np.random.uniform(-1.0, 1.0)
            x = librosa.effects.pitch_shift(x, sr=sr, n_steps=steps)
        return x.astype(np.float32)

    def spectrogram(self, spec: torch.Tensor) -> torch.Tensor:
        if not self.cfg.enable_spec_aug:
            return spec
        x = spec
        if random.random() < self.cfg.time_mask_prob:
            x = self.time_mask(x)
        if random.random() < self.cfg.freq_mask_prob:
            x = self.freq_mask(x)
        return x


def _fit_length(x: np.ndarray, target: int) -> np.ndarray:
    if len(x) < target:
        return np.pad(x, (0, target - len(x)))
    return x[:target]
