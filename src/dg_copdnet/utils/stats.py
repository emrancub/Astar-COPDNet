from __future__ import annotations

import numpy as np
from scipy import stats


def paired_stats(a, b) -> dict:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) != len(b):
        raise ValueError("paired_stats expects arrays of same length")
    diff = a - b
    out = {
        "mean_diff": float(np.mean(diff)),
        "std_diff": float(np.std(diff, ddof=1)) if len(diff) > 1 else 0.0,
    }
    try:
        w = stats.wilcoxon(a, b, alternative="two-sided", zero_method="wilcox")
        out["wilcoxon_stat"] = float(w.statistic)
        out["wilcoxon_p"] = float(w.pvalue)
    except Exception:
        out["wilcoxon_stat"] = None
        out["wilcoxon_p"] = None
    try:
        t = stats.ttest_rel(a, b, alternative="two-sided")
        out["ttest_stat"] = float(t.statistic)
        out["ttest_p"] = float(t.pvalue)
    except Exception:
        out["ttest_stat"] = None
        out["ttest_p"] = None
    ci_low, ci_high = bootstrap_ci(diff)
    out["bootstrap_ci95"] = [float(ci_low), float(ci_high)]
    return out


def bootstrap_ci(values, n_boot: int = 2000, alpha: float = 0.05, seed: int = 42):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        boots.append(np.mean(sample))
    boots = np.sort(np.asarray(boots))
    low = np.quantile(boots, alpha / 2)
    high = np.quantile(boots, 1 - alpha / 2)
    return low, high
