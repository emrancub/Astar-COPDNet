
from __future__ import annotations
import argparse, sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / 'src'
if str(SRC_ROOT) not in sys.path: sys.path.insert(0, str(SRC_ROOT))
import pandas as pd
from dg_copdnet.analysis.signal_analysis import build_signal_report
from dg_copdnet.config import AppConfig


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config', required=True); args=ap.parse_args()
    cfg=AppConfig.from_json(args.config)
    all_df=pd.read_csv(cfg.all_csv)
    build_signal_report(cfg, all_df)

if __name__=='__main__':
    main()
