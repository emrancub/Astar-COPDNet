
from __future__ import annotations

import pandas as pd
import torch
from torch.utils.data import Dataset

from dg_copdnet.config import AudioConfig, FeatureConfig, AugmentationConfig
from dg_copdnet.data.augmentations import HybridAugment
from dg_copdnet.data.features import cached_feature_vector, load_audio, logmel


class RespiratoryAudioDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        audio_cfg: AudioConfig,
        feat_cfg: FeatureConfig,
        aug_cfg: AugmentationConfig,
        feature_names: list[str] | None = None,
        is_train: bool = False,
        source_to_idx: dict[str, int] | None = None,
        label_col: str = 'label',
    ):
        self.df = df.reset_index(drop=True)
        self.audio_cfg = audio_cfg
        self.feat_cfg = feat_cfg
        self.aug = HybridAugment(aug_cfg)
        self.feature_names = feature_names or feat_cfg.handcrafted_sets
        self.is_train = is_train
        self.source_to_idx = source_to_idx or {}
        self.label_col = label_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        y, sr = load_audio(row['file_path'], self.audio_cfg)
        if self.is_train:
            y = self.aug.waveform(y, sr)
        mel = logmel(y, sr, self.audio_cfg)
        mel_tensor = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)
        mel_tensor = torch.nan_to_num(mel_tensor, nan=0.0, posinf=0.0, neginf=0.0)
        if self.is_train:
            mel_tensor = self.aug.spectrogram(mel_tensor)
        hand = cached_feature_vector(row['file_path'], self.audio_cfg, self.feat_cfg, self.feature_names)
        hand_tensor = torch.tensor(hand, dtype=torch.float32)
        hand_tensor = torch.nan_to_num(hand_tensor, nan=0.0, posinf=0.0, neginf=0.0)
        source = str(row['source'])
        domain_idx = self.source_to_idx.get(source, 0)
        label_value = row[self.label_col] if self.label_col in row.index else row['label']
        return {
            'spectrogram': mel_tensor,
            'handcrafted': hand_tensor,
            'label': torch.tensor(float(label_value), dtype=torch.float32),
            'label_long': torch.tensor(int(label_value), dtype=torch.long),
            'domain': torch.tensor(domain_idx, dtype=torch.long),
            'file_path': str(row['file_path']), 'patient_id': str(row['patient_id']), 'source': source,
            'diagnosis_raw': str(row.get('diagnosis_raw', '')),
        }
