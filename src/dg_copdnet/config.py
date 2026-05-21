
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
import json


@dataclass
class AudioConfig:
    sample_rate: int
    duration_seconds: float
    n_mels: int
    n_fft: int
    hop_length: int
    f_min: int
    f_max: int
    normalize_waveform: bool
    normalize_spectrogram: bool

    @property
    def num_samples(self) -> int:
        return int(self.sample_rate * self.duration_seconds)


@dataclass
class FeatureConfig:
    handcrafted_sets: list[str]
    cache_dir: str
    use_opensmile_if_available: bool
    use_vggish_if_available: bool


@dataclass
class AugmentationConfig:
    enable_waveform_aug: bool
    enable_spec_aug: bool
    gaussian_noise_prob: float
    gain_prob: float
    shift_prob: float
    time_stretch_prob: float
    pitch_shift_prob: float
    time_mask_prob: float
    freq_mask_prob: float


@dataclass
class TrainingConfig:
    num_folds: int
    epochs: int
    batch_size: int
    num_workers: int
    learning_rate: float
    weight_decay: float
    patience: int
    mixed_precision: bool
    gradient_clip_norm: float
    use_cuda: bool
    early_stop_metric: str


@dataclass
class ModelConfig:
    embed_dim: int
    handcrafted_dim: int
    projection_dim: int
    crnn_channels: list[int]
    crnn_hidden: int
    crnn_layers: int
    domain_hidden_dim: int
    dropout: float
    grl_lambda: float
    pretrained_effnet: bool


@dataclass
class LossConfig:
    lambda_supcon: float
    lambda_domain: float
    temperature: float


@dataclass
class DatasetPathsConfig:
    icbhi_root: str
    icbhi_audio_dir: str
    icbhi_demographic_txt: str
    icbhi_filename_differences_txt: str
    icbhi_filename_format_txt: str
    icbhi_patient_diagnosis_csv: str
    kauh_root: str
    kauh_audio_dir: str
    kauh_stethoscope_dir: str
    kauh_annotation_xlsx: str
    tr_root: str
    tr_audio_dir: str
    tr_labels_xlsx: str


@dataclass
class DatasetBuildConfig:
    auto_build: bool
    internal_sources: list[str]
    external_source: str
    all_sources_for_blind_experiments: list[str]
    metadata_output_dir: str
    real_paths: Optional[DatasetPathsConfig] = None


@dataclass
class ExperimentConfig:
    standard_model_name: str
    classical_models: list[str]
    optimizer_candidates: list[str]
    validation_strategies: list[str]
    feature_sets_for_benchmark: list[str]
    blind_protocols: list[dict[str, Any]]
    deep_baselines: list[str] | None = None
    selected_feature_triplet: list[str] | None = None


@dataclass
class AppConfig:
    seed: int
    internal_csv: str
    external_csv: str
    all_csv: str
    output_dir: str
    dataset_build: Optional[DatasetBuildConfig]
    audio: AudioConfig
    features: FeatureConfig
    augmentation: AugmentationConfig
    training: TrainingConfig
    model: ModelConfig
    loss: LossConfig
    experiments: ExperimentConfig

    @classmethod
    def from_json(cls, path: str | Path) -> "AppConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        dataset_build = None
        if data.get("dataset_build") is not None:
            db = data["dataset_build"]
            real_paths = None
            if db.get("real_paths") is not None:
                real_paths = DatasetPathsConfig(**db["real_paths"])
            dataset_build = DatasetBuildConfig(
                auto_build=db["auto_build"],
                internal_sources=db["internal_sources"],
                external_source=db["external_source"],
                all_sources_for_blind_experiments=db["all_sources_for_blind_experiments"],
                metadata_output_dir=db["metadata_output_dir"],
                real_paths=real_paths,
            )
        exp = data["experiments"]
        return cls(
            seed=data["seed"],
            internal_csv=data["internal_csv"],
            external_csv=data["external_csv"],
            all_csv=data["all_csv"],
            output_dir=data["output_dir"],
            dataset_build=dataset_build,
            audio=AudioConfig(**data["audio"]),
            features=FeatureConfig(**data["features"]),
            augmentation=AugmentationConfig(**data["augmentation"]),
            training=TrainingConfig(**data["training"]),
            model=ModelConfig(**data["model"]),
            loss=LossConfig(**data["loss"]),
            experiments=ExperimentConfig(
                standard_model_name=exp["standard_model_name"],
                classical_models=exp["classical_models"],
                optimizer_candidates=exp["optimizer_candidates"],
                validation_strategies=exp["validation_strategies"],
                feature_sets_for_benchmark=exp["feature_sets_for_benchmark"],
                blind_protocols=exp["blind_protocols"],
                deep_baselines=exp.get("deep_baselines", []),
                selected_feature_triplet=exp.get("selected_feature_triplet", []),
            ),
        )
