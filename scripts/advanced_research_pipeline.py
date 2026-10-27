
from __future__ import annotations
import argparse, sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / 'src'
if str(SRC_ROOT) not in sys.path: sys.path.insert(0, str(SRC_ROOT))
import pandas as pd
from dg_copdnet.config import AppConfig
from dg_copdnet.data.metadata_builder import build_metadata_from_real_paths
from dg_copdnet.training.advanced_trainer import train_cv_generic


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args=ap.parse_args()
    cfg=AppConfig.from_json(args.config)
    build_metadata_from_real_paths(cfg)
    df=pd.read_csv(cfg.internal_csv)
    # Proposed model binary CV
    train_cv_generic(cfg, df, model_name='hybrid_dg_respnet', label_col='label', multiclass=False, feature_names=cfg.experiments.selected_feature_triplet or cfg.features.handcrafted_sets, optimizer_name='adamw', output_subdir='proposed_binary_cv')
    # Deep baseline benchmark
    for model_name in cfg.experiments.deep_baselines or []:
        train_cv_generic(cfg, df, model_name=model_name, label_col='label', multiclass=False, feature_names=cfg.experiments.selected_feature_triplet or cfg.features.handcrafted_sets, optimizer_name='adamw', output_subdir='deep_baseline_benchmark')
