from __future__ import annotations
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay, ConfusionMatrixDisplay, confusion_matrix, calibration_curve


def _save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=600, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def protocol_bar(summary_csv: Path, out: Path):
    df = pd.read_csv(summary_csv)
    if df.empty: return
    metric_cols = [c for c in df.columns if c.startswith("external_test_") and c in ["external_test_auc","external_test_macro_ovr_auc","external_test_balanced_accuracy","external_test_macro_f1","external_test_mcc"]]
    for metric in metric_cols:
        agg = df.groupby(["task","train_sources","test_source"])[metric].agg(["mean","std"]).reset_index()
        agg["protocol"] = agg["task"] + ": " + agg["train_sources"] + "→" + agg["test_source"]
        fig, ax = plt.subplots(figsize=(max(7, len(agg)*0.55), 4.2))
        ax.bar(np.arange(len(agg)), agg["mean"], yerr=agg["std"].fillna(0), capsize=3)
        ax.set_xticks(np.arange(len(agg))); ax.set_xticklabels(agg["protocol"], rotation=45, ha="right")
        ax.set_ylim(0, 1); ax.set_ylabel(metric.replace("external_test_", "External ").replace("_", " "))
        ax.axhline(0.5, linestyle="--", linewidth=1)
        _save(fig, out / f"{metric}_bar")


def dataset_audit_figure(mapping_json: Path, out: Path):
    if not mapping_json.exists(): return
    obj = json.loads(mapping_json.read_text(encoding="utf-8"))
    summary = obj.get("dataset_summary", {})
    rows=[]
    for src, rec in summary.items():
        if src in ["internal", "all"] or not isinstance(rec, dict): continue
        for lab, n in rec.get("binary_labels", {}).items(): rows.append({"source": src, "label": str(lab), "n": n})
    if not rows: return
    df = pd.DataFrame(rows)
    piv = df.pivot_table(index="source", columns="label", values="n", fill_value=0)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    bottom = np.zeros(len(piv))
    for col in piv.columns:
        vals = piv[col].to_numpy()
        ax.bar(np.arange(len(piv)), vals, bottom=bottom, label=f"class {col}")
        bottom += vals
    ax.set_xticks(np.arange(len(piv))); ax.set_xticklabels(piv.index)
    ax.set_ylabel("Number of recordings"); ax.legend(frameon=False)
    _save(fig, out / "dataset_source_label_audit")


def prediction_figures(root: Path, out: Path):
    for npz in root.rglob("external_test_predictions.npz"):
        try:
            dat=np.load(npz, allow_pickle=True); y=dat["y"]; prob=dat["prob"]
        except Exception: continue
        name = npz.parent.name
        if prob.ndim == 1 or (prob.ndim == 2 and prob.shape[1] == 1):
            p = prob.reshape(-1)
            fig, ax = plt.subplots(figsize=(4.5,4.2)); RocCurveDisplay.from_predictions(y, p, ax=ax); _save(fig, out / f"roc_{name}")
            fig, ax = plt.subplots(figsize=(4.5,4.2)); PrecisionRecallDisplay.from_predictions(y, p, ax=ax); _save(fig, out / f"pr_{name}")
            pred = (p >= 0.5).astype(int)
        else:
            pred = prob.argmax(axis=1)
        cm = confusion_matrix(y, pred)
        fig, ax = plt.subplots(figsize=(4.8,4.2)); ConfusionMatrixDisplay(cm).plot(ax=ax, colorbar=False); ax.set_title(name); _save(fig, out / f"cm_{name}")
        if "branch_weights" in dat:
            bw = dat["branch_weights"]
            fig, ax = plt.subplots(figsize=(5,3.2)); ax.boxplot([bw[:,0], bw[:,1], bw[:,2]], tick_labels=["spectral", "temporal", "acoustic"]); ax.set_ylabel("Gate weight"); _save(fig, out / f"branch_weights_{name}")


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--output_dir", default="outputs"); args=ap.parse_args()
    root=Path(args.output_dir); out=root/"publication_figures_astar"
    protocol_bar(root/"astar_protocol"/"all_task_summary.csv", out)
    dataset_audit_figure(root/"astar_dataset_mapping.json", out)
    prediction_figures(root/"astar_protocol", out)
    print(f"Saved figures to {out}")

if __name__ == "__main__": main()
