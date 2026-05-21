# File: scripts/ablation_study.py
# Run with: python scripts/ablation_study.py --config configs/windows_real_paths.json

from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def run_ablation_study_feature(output_root: Path, outdir: Path):
    """
    Generates feature ablation plots and summary CSV
    """
    feature_csv = output_root / "feature_benchmark/feature_benchmark.csv"
    df = pd.read_csv(feature_csv)

    metrics = ["auc","accuracy","f1","mcc"]
    for metric in metrics:
        plt.figure(figsize=(10,5))
        sns.barplot(x="feature_set", y=metric, hue="model", data=df)
        plt.title(f"Feature Ablation Study ({metric.upper()})")
        plt.ylabel(metric.upper())
        plt.xlabel("Feature Set Removed / Used")
        plt.xticks(rotation=30)
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(outdir / f"ablation_feature_{metric}.png", dpi=400)
        plt.savefig(outdir / f"ablation_feature_{metric}.pdf")
        plt.close()

    df.to_csv(outdir / "ablation_feature_summary.csv", index=False)
    print(f"[INFO] Feature ablation figures saved to {outdir}")

def run_ablation_study_optimizer(output_root: Path, outdir: Path):
    """
    Generates optimizer ablation plots and summary CSV
    """
    opt_csv = output_root / "optimizer_benchmark/optimizer_benchmark.csv"
    df = pd.read_csv(opt_csv)

    metrics = ["auc","accuracy","f1","mcc"]
    for metric in metrics:
        plt.figure(figsize=(10,5))
        sns.barplot(x="optimizer", y=metric, data=df)
        plt.title(f"Optimizer Ablation Study ({metric.upper()})")
        plt.ylabel(metric.upper())
        plt.xlabel("Optimizer")
        plt.xticks(rotation=30)
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(outdir / f"ablation_optimizer_{metric}.png", dpi=400)
        plt.savefig(outdir / f"ablation_optimizer_{metric}.pdf")
        plt.close()

    df.to_csv(outdir / "ablation_optimizer_summary.csv", index=False)
    print(f"[INFO] Optimizer ablation figures saved to {outdir}")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--outdir", default=None)
    args = parser.parse_args()

    config_path = Path(args.config)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    project_root = Path(__file__).resolve().parents[1]
    output_root = project_root / cfg["output_dir"]
    outdir = Path(args.outdir) if args.outdir else output_root / "ablation_study_figures"
    outdir.mkdir(parents=True, exist_ok=True)

    # Run feature and optimizer ablations
    run_ablation_study_feature(output_root, outdir)
    run_ablation_study_optimizer(output_root, outdir)

if __name__ == "__main__":
    main()