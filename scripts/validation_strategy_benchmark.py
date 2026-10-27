from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pathlib import Path
import pandas as pd

from dg_copdnet.config import AppConfig
from dg_copdnet.training.trainer import train_standard_cv, grouped_holdout_experiment
from dg_copdnet.utils.io import ensure_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = AppConfig.from_json(args.config)
    df = pd.read_csv(cfg.internal_csv)
    rows = []

    if "grouped_stratified_kfold" in cfg.experiments.validation_strategies:
        _, stats = train_standard_cv(cfg, subset_df=df, save_prefix="validation_grouped_kfold")
        rows.append({"strategy": "grouped_stratified_kfold", "auc": stats["val_auc"]["mean"], "accuracy": stats["val_accuracy"]["mean"], "f1": stats["val_f1"]["mean"]})

    if "grouped_holdout" in cfg.experiments.validation_strategies:
        hist = grouped_holdout_experiment(cfg, subset_df=df, save_prefix="validation_grouped_holdout")
        best_row = hist.loc[hist["test_auc"].idxmax()]
        rows.append({"strategy": "grouped_holdout", "auc": float(best_row["test_auc"]), "accuracy": float(best_row["test_accuracy"]), "f1": float(best_row["test_f1"])})

    out = pd.DataFrame(rows)
    out_dir = ensure_dir(Path(cfg.output_dir) / "validation_strategy_benchmark")
    out.to_csv(out_dir / "validation_strategy_benchmark.csv", index=False)
    print(out)


if __name__ == "__main__":
    main()
