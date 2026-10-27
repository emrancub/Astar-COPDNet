
from __future__ import annotations
import argparse, sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / 'src'
if str(SRC_ROOT) not in sys.path: sys.path.insert(0, str(SRC_ROOT))
import pandas as pd
from dg_copdnet.config import AppConfig
from dg_copdnet.training.advanced_trainer import train_cv_generic


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config', required=True); args=ap.parse_args()
    cfg=AppConfig.from_json(args.config)
    all_df=pd.read_csv(cfg.all_csv)
    for source in ['ICBHI','KAUH','TR']:
        sdf=all_df[all_df['source'].str.upper()==source].copy()
        if sdf.empty or sdf['multiclass_label'].nunique()<2: continue
        print('Multiclass analysis for', source, 'classes=', sdf['multiclass_label'].nunique())
        train_cv_generic(cfg, sdf, model_name='hybrid_dg_respnet', label_col='multiclass_label', multiclass=True, feature_names=cfg.experiments.selected_feature_triplet or cfg.features.handcrafted_sets, optimizer_name='adamw', output_subdir=f'multiclass_{source.lower()}')

if __name__=='__main__':
    main()
