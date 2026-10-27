from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import numpy as np
import pandas as pd

from dg_copdnet.config import AppConfig
from dg_copdnet.data.features import cached_feature_vector
from dg_copdnet.data.splits import make_grouped_folds
from dg_copdnet.training.classical_models import make_classical_model
from dg_copdnet.utils.metrics import compute_metrics
from dg_copdnet.utils.io import ensure_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = AppConfig.from_json(args.config)
    df = pd.read_csv(cfg.internal_csv)
    out_rows = []

    for feature_set in cfg.experiments.feature_sets_for_benchmark:
        X = np.vstack([cached_feature_vector(p, cfg.audio, cfg.features, [feature_set]) for p in df["file_path"]])
        y = df["label"].astype(int).to_numpy()
        for model_name in cfg.experiments.classical_models:
            fold_metrics = []
            for _, tr_idx, te_idx in make_grouped_folds(df, cfg.training.num_folds, cfg.seed):
                model = make_classical_model(model_name)
                model.fit(X[tr_idx], y[tr_idx])
                prob = model.predict_proba(X[te_idx])[:, 1]
                fold_metrics.append(compute_metrics(y[te_idx], prob))
            out_rows.append({
                "feature_set": feature_set,
                "model": model_name,
                "auc": float(np.mean([m["auc"] for m in fold_metrics])),
                "accuracy": float(np.mean([m["accuracy"] for m in fold_metrics])),
                "f1": float(np.mean([m["f1"] for m in fold_metrics])),
                "mcc": float(np.mean([m["mcc"] for m in fold_metrics]))
            })

    out = pd.DataFrame(out_rows)
    out_dir = ensure_dir(Path(cfg.output_dir) / "feature_benchmark")
    out.to_csv(out_dir / "feature_benchmark.csv", index=False)
    print(out)


if __name__ == "__main__":
    main()
