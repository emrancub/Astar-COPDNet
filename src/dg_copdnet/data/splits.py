from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GroupShuffleSplit


def make_grouped_folds(df: pd.DataFrame, n_splits: int, seed: int):
    groups = df["patient_id"].astype(str)
    group_df = df.groupby("patient_id").agg(label=("label", "max")).reset_index()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (tr_g, va_g) in enumerate(skf.split(group_df["patient_id"], group_df["label"]), start=1):
        tr_patients = set(group_df.iloc[tr_g]["patient_id"].astype(str))
        va_patients = set(group_df.iloc[va_g]["patient_id"].astype(str))
        tr_idx = df.index[df["patient_id"].astype(str).isin(tr_patients)].to_numpy()
        va_idx = df.index[df["patient_id"].astype(str).isin(va_patients)].to_numpy()
        yield fold, tr_idx, va_idx


def make_grouped_holdout(df: pd.DataFrame, test_size: float, seed: int):
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    groups = df["patient_id"].astype(str)
    y = df["label"].astype(int)
    tr_idx, te_idx = next(splitter.split(df, y, groups=groups))
    return tr_idx, te_idx
