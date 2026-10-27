from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import pandas as pd

from dg_copdnet.config import AppConfig
from dg_copdnet.analysis.signal_analysis import run_signal_analysis
from dg_copdnet.analysis.figure_factory import generate_summary_figures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = AppConfig.from_json(args.config)
    all_df = pd.read_csv(cfg.all_csv)
    run_signal_analysis(all_df, cfg, Path(cfg.output_dir) / "signal_analysis")
    manifest = generate_summary_figures(cfg.output_dir)
    print(manifest)


if __name__ == "__main__":
    main()
