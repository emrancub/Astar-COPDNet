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
from dg_copdnet.utils.io import ensure_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = AppConfig.from_json(args.config)

    rows = []
    for opt_name in cfg.experiments.optimizer_candidates:
        _, stats = train_standard_cv(cfg, optimizer_name=opt_name, save_prefix=f"optimizer_{opt_name}")
        row = {"optimizer": opt_name}
        for k, v in stats.items():
            if k.startswith("val_"):
                row[k.replace("val_", "")] = v["mean"]
        rows.append(row)
    out = pd.DataFrame(rows)
    out_dir = ensure_dir(Path(cfg.output_dir) / "optimizer_benchmark")
    out.to_csv(out_dir / "optimizer_benchmark.csv", index=False)
    print(out)


if __name__ == "__main__":
    main()
