# File: scripts/generate_all_publication_figures.py
# Run: python scripts/generate_all_publication_figures.py --config configs/windows_real_paths.json

from __future__ import annotations
import sys
from pathlib import Path
import argparse
import json
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.impute import SimpleImputer
from sklearn import __version__ as sklearn_version
from packaging import version

# -------------------
# Utility functions
# -------------------
def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path

def save_figure(fig: plt.Figure, outdir: Path, stem: str, dpi: int = 400):
    ensure_dir(outdir)
    fig.savefig(outdir / f"{stem}.png", dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(outdir / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)

def standardize_features(X: np.ndarray) -> np.ndarray:
    return StandardScaler().fit_transform(X)

def reduce_for_tsne(X: np.ndarray, max_pca_components: int = 30) -> np.ndarray:
    n_samples, n_features = X.shape
    n_comp = min(max_pca_components, n_features, max(2, n_samples - 1))
    if n_features <= n_comp:
        return X
    return PCA(n_components=n_comp, random_state=42).fit_transform(X)

def choose_perplexity(n_samples: int) -> int:
    if n_samples <= 10:
        return 3
    return int(min(30, max(5, (n_samples - 1) // 3)))

def create_tsne(n_components=2, perplexity=30, random_state=42):
    if version.parse(sklearn_version) >= version.parse("0.24"):
        return TSNE(
            n_components=n_components,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            random_state=random_state,
            max_iter=1500,
        )
    else:
        return TSNE(
            n_components=n_components,
            perplexity=perplexity,
            init="pca",
            learning_rate=200.0,
            random_state=random_state,
            n_iter=1500,
        )

def get_numeric_cols(df: pd.DataFrame, exclude: list[str] | None = None) -> list[str]:
    exclude = exclude or []
    cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in cols if c not in exclude]

def impute_features(X: pd.DataFrame) -> np.ndarray:
    imp = SimpleImputer(strategy="mean")
    return imp.fit_transform(X)

# -------------------
# t-SNE figures
# -------------------
def generate_tsne_figures(feature_df: pd.DataFrame, outdir: Path):
    sources = sorted(feature_df["source"].dropna().unique())
    numeric_cols = get_numeric_cols(feature_df, exclude=["label"])
    for source in sources:
        sdf = feature_df[feature_df["source"] == source].copy()
        X = impute_features(sdf[numeric_cols])
        X = standardize_features(X)
        X = reduce_for_tsne(X)
        tsne = create_tsne(perplexity=choose_perplexity(len(sdf)))
        Z = tsne.fit_transform(X)

        fig, ax = plt.subplots(figsize=(7,6))
        for label_value, label_name, color in zip([0,1], ["Non-COPD","COPD"], ["#1f77b4","#d62728"]):
            mask = sdf["label"] == label_value
            ax.scatter(Z[mask.values,0], Z[mask.values,1], c=color, label=label_name,
                       alpha=0.8, edgecolors='white', linewidths=0.3, s=28)
        ax.set_title(f"t-SNE projection: {source}", weight="bold")
        ax.set_xlabel("t-SNE dimension 1")
        ax.set_ylabel("t-SNE dimension 2")
        ax.legend(frameon=True)
        ax.grid(alpha=0.25)
        save_figure(fig, outdir, f"tsne_{source.lower()}")

# -------------------
# Silhouette plots
# -------------------
def silhouette_evaluation_and_plots(feature_df: pd.DataFrame, outdir: Path):
    sources = sorted(feature_df["source"].dropna().unique())
    numeric_cols = get_numeric_cols(feature_df, exclude=["label"])
    for source in sources:
        sdf = feature_df[feature_df["source"] == source].copy()
        X = impute_features(sdf[numeric_cols])
        X = standardize_features(X)
        X = reduce_for_tsne(X)
        max_k = min(8, len(sdf)-1)
        k_values = list(range(2,max_k+1))
        sil_scores = []
        for k in k_values:
            km = KMeans(n_clusters=k,n_init=20,random_state=42)
            labels = km.fit_predict(X)
            sil_scores.append(silhouette_score(X,labels))
        best_k = k_values[np.argmax(sil_scores)]
        best_score = max(sil_scores)

        fig, ax = plt.subplots(figsize=(7,5))
        ax.plot(k_values, sil_scores, marker='o', linewidth=2)
        ax.scatter([best_k],[best_score], s=80, zorder=5)
        ax.set_title(f"Silhouette score vs clusters: {source}", weight="bold")
        ax.set_xlabel("Number of clusters (k)")
        ax.set_ylabel("Average silhouette score")
        ax.grid(alpha=0.3)
        save_figure(fig, outdir, f"silhouette_scores_{source.lower()}")

# -------------------
# Main
# -------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--outdir", default=None)
    args = parser.parse_args()

    config_path = Path(args.config)
    with open(config_path,"r",encoding="utf-8") as f:
        cfg = json.load(f)

    metadata_df = pd.read_csv(Path(cfg["all_csv"]))
    metadata_df["source"] = metadata_df["source"].astype(str).str.upper()
    metadata_df["label"] = metadata_df["label"].astype(int)

    outdir = Path(args.outdir) if args.outdir else Path(cfg["output_dir"]) / "publication_figures_q1_extra"
    ensure_dir(outdir)

    # Generate all extra publication figures
    generate_tsne_figures(metadata_df, outdir)
    silhouette_evaluation_and_plots(metadata_df, outdir)

    print(f"[INFO] All figures saved to {outdir}")

if __name__=="__main__":
    warnings.filterwarnings("ignore")
    main()