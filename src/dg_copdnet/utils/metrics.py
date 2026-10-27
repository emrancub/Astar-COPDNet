
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score, roc_auc_score, average_precision_score, f1_score, matthews_corrcoef,
    precision_score, recall_score, confusion_matrix, brier_score_loss,
    balanced_accuracy_score
)
from sklearn.preprocessing import label_binarize


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def expected_calibration_error(y_true, y_prob, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    if y_prob.ndim == 2:
        y_prob = y_prob.max(axis=1)
    y_prob = np.nan_to_num(y_prob, nan=0.5, posinf=1.0, neginf=0.0)
    y_prob = np.clip(y_prob, 0.0, 1.0)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        idx = (y_prob >= bins[i]) & (y_prob < bins[i + 1] if i < n_bins - 1 else y_prob <= bins[i + 1])
        if not np.any(idx):
            continue
        conf = float(y_prob[idx].mean())
        acc = float(y_true[idx].mean())
        ece += abs(acc - conf) * (idx.sum() / len(y_true))
    return float(ece)


def compute_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_prob = np.nan_to_num(y_prob, nan=0.5, posinf=1.0, neginf=0.0)
    y_prob = np.clip(y_prob, 0.0, 1.0)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out = {
        'auc': float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float('nan'),
        'auprc': float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float('nan'),
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
        'sensitivity': float(recall_score(y_true, y_pred, zero_division=0)),
        'specificity': _safe_div(tn, tn + fp),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
        'mcc': float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_pred)) > 1 else 0.0,
        'brier': float(brier_score_loss(y_true, y_prob)),
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
        'ece': expected_calibration_error(y_true, y_prob, n_bins=10),
        'confusion_matrix': [[int(tn), int(fp)], [int(fn), int(tp)]],
    }
    return out


def compute_multiclass_metrics(y_true, y_prob) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_prob = np.nan_to_num(y_prob, nan=0.0, posinf=1.0, neginf=0.0)
    y_pred = y_prob.argmax(axis=1)
    labels = sorted(np.unique(y_true).tolist())
    out = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
        'macro_f1': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'weighted_f1': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
        'macro_precision': float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
        'macro_recall': float(recall_score(y_true, y_pred, average='macro', zero_division=0)),
        'mcc': float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_pred)) > 1 else 0.0,
        'ece': expected_calibration_error((y_true==y_pred).astype(int), y_prob.max(axis=1)),
    }
    if len(labels) > 2:
        Y = label_binarize(y_true, classes=labels)
        out['macro_ovr_auc'] = float(roc_auc_score(Y, y_prob, average='macro', multi_class='ovr'))
        out['macro_ovr_auprc'] = float(average_precision_score(Y, y_prob, average='macro'))
    else:
        out['macro_ovr_auc'] = float(roc_auc_score(y_true, y_prob[:,1] if y_prob.shape[1]>1 else y_prob[:,0]))
        out['macro_ovr_auprc'] = float(average_precision_score(y_true, y_prob[:,1] if y_prob.shape[1]>1 else y_prob[:,0]))
    out['confusion_matrix'] = confusion_matrix(y_true, y_pred, labels=labels).astype(int).tolist()
    return out
