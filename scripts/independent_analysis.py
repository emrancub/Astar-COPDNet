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
from dg_copdnet.training.trainer import train_standard_cv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = AppConfig.from_json(args.config)
    all_df = pd.read_csv(cfg.all_csv) if Path(cfg.all_csv).exists() else pd.read_csv(cfg.internal_csv)
    out_rows = []
    for source in sorted(all_df["source"].unique()):
        sdf = all_df[all_df["source"] == source].reset_index(drop=True)
        if sdf["patient_id"].nunique() < cfg.training.num_folds:
            print(f"Skipping {source}: insufficient patients for {cfg.training.num_folds}-fold CV")
            continue
        _, stats = train_standard_cv(cfg, subset_df=sdf, save_prefix=f"independent_{source.lower()}")
        row = {"dataset": source}
        for k, v in stats.items():
            if k.startswith("val_"):
                row[k] = v["mean"]
        out_rows.append(row)
    out = pd.DataFrame(out_rows)
    out_dir = Path(cfg.output_dir) / "independent_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / "independent_summary.csv", index=False)
    print(out)


if __name__ == "__main__":
    main()
