from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def make_classical_model(name: str):
    name = name.lower()
    if name == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
    if name == "svm_rbf":
        return make_pipeline(StandardScaler(), SVC(probability=True, class_weight="balanced"))
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=42)
    raise ValueError(f"Unsupported classical model: {name}")
