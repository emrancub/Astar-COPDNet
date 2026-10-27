from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import pandas as pd
import torch

from dg_copdnet.config import AppConfig
from dg_copdnet.training.trainer import train_standard_cv, evaluate_checkpoint_on_df
from dg_copdnet.utils.io import ensure_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = AppConfig.from_json(args.config)
    all_df = pd.read_csv(cfg.all_csv)
    out_dir = ensure_dir(Path(cfg.output_dir) / "blind_experiments")
    rows = []

    for i, proto in enumerate(cfg.experiments.blind_protocols, start=1):
        train_sources = [s.upper() for s in proto["train_sources"]]
        test_source = proto["test_source"].upper()
        train_df = all_df[all_df["source"].str.upper().isin(train_sources)].reset_index(drop=True)
        test_df = all_df[all_df["source"].str.upper() == test_source].reset_index(drop=True)
        if train_df.empty or test_df.empty:
            continue
        prefix = f"blind_{i}_{'_'.join(train_sources)}_to_{test_source}".lower()
        _, _ = train_standard_cv(cfg, subset_df=train_df, save_prefix=prefix)
        ckpt = Path(cfg.output_dir) / prefix / "fold_1" / "best.pt"
        if not ckpt.exists():
            continue
        metrics = evaluate_checkpoint_on_df(cfg, str(ckpt), test_df, save_dir_name=f"{prefix}_eval")
        rows.append({
            "protocol": f"{'+'.join(train_sources)}-> {test_source}",
            **{k: v for k, v in metrics.items() if isinstance(v, (float, int))}
        })

    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "blind_summary.csv", index=False)
    print(out)


if __name__ == "__main__":
    main()
