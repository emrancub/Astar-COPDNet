from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

from dg_copdnet.config import AppConfig
from dg_copdnet.data.metadata_builder import build_metadata_from_real_paths
from dg_copdnet.data.robust_splits import make_train_val_external_split, audit_split, save_split_audit
from dg_copdnet.training.astar_trainer import train_one_split
from dg_copdnet.utils.io import ensure_dir, save_json


def prepare_metadata(cfg: AppConfig):
    if cfg.dataset_build and cfg.dataset_build.auto_build:
        _, _, all_csv, summary = build_metadata_from_real_paths(cfg)
        return pd.read_csv(all_csv), summary
    return pd.read_csv(cfg.all_csv), {}


def global_multiclass_mapping(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    df = df.copy()
    names = sorted(df["multiclass_name"].fillna("UNKNOWN").astype(str).unique())
    # clinically useful stable ordering when available
    preferred = ["COPD0", "COPD1", "COPD2", "COPD3", "COPD4", "COPD", "HEALTHY", "NORMAL", "ASTHMA", "BRONCHIECTASIS", "PNEUMONIA", "URTI", "LRTI", "BRONCHIOLITIS"]
    ordered = [x for x in preferred if x in names] + [x for x in names if x not in preferred]
    mapping = {name: i for i, name in enumerate(ordered)}
    df["multiclass_global"] = df["multiclass_name"].fillna("UNKNOWN").astype(str).map(mapping).astype(int)
    return df, mapping


def run_task(cfg: AppConfig, df: pd.DataFrame, task: str, seeds: list[int]):
    out_root = ensure_dir(Path(cfg.output_dir) / "astar_protocol" / task)
    label_col = "binary_label" if task == "binary" else "multiclass_global"
    num_classes = 2 if task == "binary" else int(df[label_col].nunique())
    results = []
    protocols = cfg.experiments.blind_protocols
    for proto in protocols:
        train_sources = proto["train_sources"]
        test_source = proto["test_source"]
        for seed in seeds:
            name = f"{'-'.join(train_sources)}_to_{test_source}_seed{seed}".replace('/', '-')
            odir = ensure_dir(out_root / name)
            train_df, val_df, test_df = make_train_val_external_split(df, train_sources, test_source, label_col, seed=seed, val_size=0.2)
            audit = audit_split(train_df, val_df, test_df, label_col)
            save_split_audit(odir / "split_audit.json", audit)
            metrics = train_one_split(cfg, train_df, val_df, test_df, odir, task=task, label_col=label_col, num_classes=num_classes, seed=seed)
            row = {"task": task, "train_sources": "+".join(train_sources), "test_source": test_source, "seed": seed}
            for part in ["validation", "external_test"]:
                if part in metrics:
                    for k, v in metrics[part].items():
                        if isinstance(v, (int, float)):
                            row[f"{part}_{k}"] = v
            results.append(row)
            pd.DataFrame(results).to_csv(out_root / "summary_by_protocol_seed.csv", index=False)
    return pd.DataFrame(results)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/astar_real_paths.json")
    ap.add_argument("--task", choices=["binary", "multiclass", "both"], default="both")
    ap.add_argument("--seeds", default="42,43,44")
    args = ap.parse_args()
    cfg = AppConfig.from_json(args.config)
    df, summary = prepare_metadata(cfg)
    df, mc_map = global_multiclass_mapping(df)
    ensure_dir(cfg.output_dir)
    save_json({"dataset_summary": summary, "multiclass_mapping": mc_map}, Path(cfg.output_dir) / "astar_dataset_mapping.json")
    seeds = [int(x) for x in args.seeds.split(',') if x.strip()]
    tasks = ["binary", "multiclass"] if args.task == "both" else [args.task]
    all_results = []
    for task in tasks:
        all_results.append(run_task(cfg, df, task, seeds))
    pd.concat(all_results, ignore_index=True).to_csv(Path(cfg.output_dir) / "astar_protocol" / "all_task_summary.csv", index=False)

if __name__ == "__main__":
    main()
