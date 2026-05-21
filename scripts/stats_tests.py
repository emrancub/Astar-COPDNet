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
from dg_copdnet.utils.stats import paired_stats
from dg_copdnet.utils.io import ensure_dir, save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--reference", default="optimizer_adamw")
    parser.add_argument("--comparison", default="optimizer_adam")
    args = parser.parse_args()

    cfg = AppConfig.from_json(args.config)
    ref_csv = Path(cfg.output_dir) / args.reference / "cv_summary.csv"
    cmp_csv = Path(cfg.output_dir) / args.comparison / "cv_summary.csv"
    if not (ref_csv.exists() and cmp_csv.exists()):
        raise FileNotFoundError("Run both experiments before statistical testing.")
    ref_df = pd.read_csv(ref_csv)
    cmp_df = pd.read_csv(cmp_csv)
    result = paired_stats(ref_df["val_auc"], cmp_df["val_auc"])
    out_dir = ensure_dir(Path(cfg.output_dir) / "stats_tests")
    save_json(result, out_dir / f"{args.reference}_vs_{args.comparison}.json")
    print(result)


if __name__ == "__main__":
    main()
