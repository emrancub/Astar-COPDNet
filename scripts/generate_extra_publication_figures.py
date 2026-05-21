# File: scripts/generate_publication_figures_with_confusion.py
# Run: python scripts/generate_publication_figures_with_confusion.py --config configs/windows_real_paths.json

from __future__ import annotations
import argparse
import json
import warnings
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Utilities
def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path

def save_figure(fig: plt.Figure, outdir: Path, stem: str, dpi: int = 400):
    ensure_dir(outdir)
    fig.savefig(outdir / f"{stem}.png", dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(outdir / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)

def heatmap(ax, data, row_labels, col_labels, title, cmap="Blues", fmt=".3f", cbar_label="Value"):
    im = ax.imshow(data, cmap=cmap, aspect="auto")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_title(title, fontsize=12, weight="bold")
    vmax = np.nanmax(data) if np.size(data) else 1.0
    threshold = vmax / 2.0 if vmax > 0 else 0.5
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            txt = format(val, fmt) if isinstance(val, (int, float)) else str(val)
            ax.text(j, i, txt, ha="center", va="center", color="white" if val > threshold else "black", fontsize=9, weight="bold")
    return im

def plot_confusion_heatmap(ax, cm: np.ndarray, title: str, normalize: bool = False):
    if normalize:
        denom = cm.sum(axis=1, keepdims=True)
        denom[denom == 0] = 1
        data = cm / denom
        fmt = ".2f"
        cbar_label = "Proportion"
    else:
        data = cm
        fmt = ".0f"
        cbar_label = "Count"
    heatmap(ax, data, row_labels=["True Non-COPD","True COPD"], col_labels=["Pred Non-COPD","Pred COPD"], title=title, cmap="Blues", fmt=fmt, cbar_label=cbar_label)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

# Dataset composition
def generate_dataset_composition_figure(metadata_df: pd.DataFrame, outdir: Path):
    df = metadata_df.copy()
    df["source"] = df["source"].astype(str).str.upper()
    df["label_int"] = df["label"].astype(int)
    counts = df.groupby(["source","label_int"]).size().unstack(fill_value=0).reindex(columns=[0,1], fill_value=0)
    fig, ax = plt.subplots(figsize=(8,5))
    x = np.arange(len(counts.index))
    non_copd = counts[0].values
    copd = counts[1].values
    ax.bar(x, non_copd, label="Non-COPD")
    ax.bar(x, copd, bottom=non_copd, label="COPD")
    for i, (a,b) in enumerate(zip(non_copd,copd)):
        ax.text(i,a/2 if a>0 else 0.5,str(int(a)), ha="center", va="center", fontsize=10, weight="bold")
        ax.text(i,a+b/2 if b>0 else a+0.5,str(int(b)), ha="center", va="center", fontsize=10, weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(counts.index.tolist())
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Number of recordings")
    ax.set_title("Dataset composition by binary class", weight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, outdir, "dataset_composition")

# Silhouette evaluation
def silhouette_evaluation_and_plots(feature_df: pd.DataFrame, outdir: Path):
    sources = sorted(feature_df["source"].dropna().unique())
    numeric_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in ["label"]]
    fig_scores, axes = plt.subplots(1,len(sources), figsize=(7*len(sources),5), squeeze=False)
    axes = axes.ravel()
    for ax, source in zip(axes,sources):
        sdf = feature_df[feature_df["source"]==source].copy()
        sdf[numeric_cols] = sdf[numeric_cols].fillna(sdf[numeric_cols].mean())
        sdf = sdf.dropna(subset=numeric_cols)
        if sdf.shape[0] < 2:
            continue
        X = StandardScaler().fit_transform(sdf[numeric_cols].values)
        max_k = min(8,len(sdf)-1)
        if max_k < 2:
            continue
        k_values = list(range(2,max_k+1))
        sil_scores = []
        for k in k_values:
            km = KMeans(n_clusters=k, random_state=42, n_init=20)
            labels = km.fit_predict(X)
            sil_scores.append(silhouette_score(X,labels))
        best_k = k_values[np.argmax(sil_scores)]
        best_score = max(sil_scores)
        ax.plot(k_values, sil_scores, marker='o', linewidth=2)
        ax.scatter([best_k],[best_score], s=80, zorder=5)
        ax.set_title(f"Silhouette score by cluster number: {source}", weight="bold")
        ax.set_xlabel("Number of clusters (k)")
        ax.set_ylabel("Average silhouette score")
        ax.grid(alpha=0.30)
    save_figure(fig_scores,outdir,"silhouette_scores_all_datasets")

# Confusion matrices
def generate_confusion_matrices(output_root: Path, outdir: Path):
    csv_path = output_root / "independent_analysis" / "independent_summary.csv"
    df = pd.read_csv(csv_path)
    df.fillna(0, inplace=True)  # <- NaN safety
    datasets = df["dataset"].tolist()
    fig, axes = plt.subplots(2,len(datasets), figsize=(6*len(datasets),10))
    axes = np.array(axes).reshape(2,len(datasets))
    for j, (_,row) in enumerate(df.iterrows()):
        dataset = row["dataset"]
        tp, tn, fp, fn = int(round(row.get("val_tp",0))), int(round(row.get("val_tn",0))), int(round(row.get("val_fp",0))), int(round(row.get("val_fn",0)))
        cm = np.array([[tn, fp],[fn, tp]], dtype=float)
        plot_confusion_heatmap(axes[0,j], cm, title=f"{dataset} (counts)", normalize=False)
        plot_confusion_heatmap(axes[1,j], cm, title=f"{dataset} (row-normalized)", normalize=True)
    save_figure(fig,outdir,"confusion_matrices_all_datasets")

# Main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--outdir", default=None)
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    config_path = Path(args.config)
    with open(config_path,"r",encoding="utf-8") as f:
        cfg = json.load(f)
    output_root = project_root / cfg["output_dir"]
    outdir = Path(args.outdir) if args.outdir else output_root / "publication_figures_with_confusion"
    ensure_dir(outdir)
    metadata_df = pd.read_csv(project_root / cfg["all_csv"])
    metadata_df["source"] = metadata_df["source"].astype(str).str.upper()

    # Figures
    generate_dataset_composition_figure(metadata_df,outdir)
    silhouette_evaluation_and_plots(metadata_df,outdir)
    generate_confusion_matrices(output_root,outdir)

    print(f"[INFO] Figures saved to {outdir}")

if __name__=="__main__":
    warnings.filterwarnings("ignore")
    main()