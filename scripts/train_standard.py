from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from dg_copdnet.config import AppConfig
from dg_copdnet.training.trainer import train_standard_cv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--optimizer", default="adamw")
    args = parser.parse_args()
    cfg = AppConfig.from_json(args.config)
    cv_df, stats = train_standard_cv(cfg, optimizer_name=args.optimizer)
    print(cv_df)
    print(stats)


if __name__ == "__main__":
    main()
