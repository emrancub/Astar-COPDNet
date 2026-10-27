from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold


def _strat_key(df: pd.DataFrame, label_col: str) -> np.ndarray:
    # Source-aware stratification reduces accidental source/class imbalance.
    return (df[label_col].astype(str) + "__" + df["source"].astype(str)).to_numpy()


def make_grouped_internal_folds(df: pd.DataFrame, label_col: str, n_splits: int, seed: int):
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    y = _strat_key(df, label_col)
    groups = df["patient_id"].astype(str).to_numpy()
    for fold, (tr, va) in enumerate(sgkf.split(df, y, groups)):
        yield {"fold": fold, "train_idx": tr.tolist(), "val_idx": va.tolist()}


def make_train_val_external_split(df: pd.DataFrame, train_sources: list[str], test_source: str, label_col: str, seed: int, val_size: float = 0.2):
    train_pool = df[df["source"].isin(train_sources)].reset_index(drop=True)
    test_df = df[df["source"].eq(test_source)].reset_index(drop=True)
    if train_pool.empty or test_df.empty:
        raise ValueError(f"Empty train/test split for {train_sources} -> {test_source}")
    splitter = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    # GroupShuffleSplit has no stratification; report balance after split and never tune on external test.
    idx = np.arange(len(train_pool))
    tr_idx, va_idx = next(splitter.split(idx, groups=train_pool["patient_id"].astype(str)))
    return train_pool.iloc[tr_idx].reset_index(drop=True), train_pool.iloc[va_idx].reset_index(drop=True), test_df


def audit_split(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame | None, label_col: str) -> dict:
    def one(x: pd.DataFrame):
        return {
            "records": int(len(x)),
            "patients": int(x["patient_id"].nunique()),
            "sources": {str(k): int(v) for k, v in x["source"].value_counts().to_dict().items()},
            "labels": {str(k): int(v) for k, v in x[label_col].value_counts().to_dict().items()},
        }
    out = {"train": one(train_df), "validation": one(val_df)}
    if test_df is not None:
        out["external_test"] = one(test_df)
    leaks = set(train_df.patient_id) & set(val_df.patient_id)
    if test_df is not None:
        leaks |= set(train_df.patient_id) & set(test_df.patient_id)
        leaks |= set(val_df.patient_id) & set(test_df.patient_id)
    out["patient_leak_count"] = len(leaks)
    out["patient_leak_examples"] = sorted(list(leaks))[:10]
    return out


def save_split_audit(path: str | Path, audit: dict):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(audit, indent=2), encoding="utf-8")
