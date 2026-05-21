
from __future__ import annotations
import argparse, itertools, sys
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
    df=pd.read_csv(cfg.internal_csv)
    candidate=['mfcc_optimized','logmel_stats','phonation','mfcc_default']
    rows=[]
    for combo in itertools.combinations(candidate, 3):
        out=train_cv_generic(cfg, df, model_name='hybrid_dg_respnet', label_col='label', multiclass=False, feature_names=list(combo), optimizer_name='adamw', output_subdir=f'feature_combo_{"_".join(combo)}')
        rows.append({'combo':'+'.join(combo), 'auc_mean':out['auc'].mean(), 'f1_mean':out['f1'].mean(), 'mcc_mean':out['mcc'].mean()})
    pd.DataFrame(rows).to_csv(Path(cfg.output_dir)/'feature_combo_benchmark.csv', index=False)

if __name__=='__main__':
    main()
