
from __future__ import annotations
import argparse, sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / 'src'
if str(SRC_ROOT) not in sys.path: sys.path.insert(0, str(SRC_ROOT))
import pandas as pd
from dg_copdnet.analysis.figure_factory import build_advanced_figure_pack
from dg_copdnet.config import AppConfig


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config', required=True); args=ap.parse_args()
    cfg=AppConfig.from_json(args.config)
    build_advanced_figure_pack(cfg)

if __name__=='__main__':
    main()
