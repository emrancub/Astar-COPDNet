# File: scripts/subgroup_analysis.py
# Run: python scripts/subgroup_analysis.py --config configs/windows_real_paths.json

from __future__ import annotations
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import ttest_ind
from math import sqrt
import argparse, json

sns.set(style="whitegrid", palette="Set2")
plt.rcParams.update({'figure.autolayout': True})


# ------------------------ UTILITIES ------------------------

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_figure(fig, outdir: Path, name: str):
    ensure_dir(outdir)
    fig.savefig(outdir / f"{name}.png", dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(outdir / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def compute_cohens_d(x, y):
    if len(x) == 0 or len(y) == 0:
        return np.nan
    nx, ny = len(x), len(y)
    pooled_std = sqrt(((nx - 1) * np.std(x, ddof=1) ** 2 + (ny - 1) * np.std(y, ddof=1) ** 2) / (
                nx + ny - 2)) if nx + ny - 2 > 0 else np.nan
    return (np.mean(x) - np.mean(y)) / pooled_std if pooled_std > 0 else np.nan


def confidence_interval(x, y, alpha=0.05):
    if len(x) == 0 or len(y) == 0:
        return np.nan, np.nan
    nx, ny = len(x), len(y)
    mx, my = np.mean(x), np.mean(y)
    sx, sy = np.std(x, ddof=1), np.std(y, ddof=1)
    se = sqrt(sx ** 2 / nx + sy ** 2 / ny) if nx > 0 and ny > 0 else np.nan
    from scipy.stats import t
    dof = min(max(nx - 1, 0), max(ny - 1, 0))
    if se == 0 or np.isnan(se) or dof <= 0:
        return np.nan, np.nan
    t_crit = t.ppf(1 - alpha / 2, dof)
    diff = mx - my
    return diff - t_crit * se, diff + t_crit * se


# ------------------------ DATA MAPPING ------------------------

def map_subgroup_binary(df, dataset_name):
    ds_upper = dataset_name.upper()
    if ds_upper in ['ICBHI', 'RESPIRATORYDATABASE']:
        # 0=Healthy, 1=COPD, 2-7=Other diseases => Non-COPD
        df['subgroup_binary'] = df['label'].map({
            0: 'Non-COPD', 1: 'COPD', 2: 'Non-COPD', 3: 'Non-COPD',
            4: 'Non-COPD', 5: 'Non-COPD', 6: 'Non-COPD', 7: 'Non-COPD'
        })
    elif ds_upper in ['KAUH', 'JWYY9NP4GV-3']:
        # 0=Healthy,1=COPD, others Non-COPD
        df['subgroup_binary'] = df['label'].map({
            0: 'Non-COPD', 1: 'COPD', 2: 'Non-COPD', 3: 'Non-COPD',
            4: 'Non-COPD', 5: 'Non-COPD'
        })
    elif ds_upper in ['TR', 'RESPIRATORYDATABASE@TR']:
        # COPD0=Healthy=Non-COPD, COPD1-4=COPD
        df['subgroup_binary'] = df['label'].apply(lambda x: 'Non-COPD' if x == 0 else 'COPD')
    else:
        df['subgroup_binary'] = df['label'].map({0: 'Non-COPD', 1: 'COPD'})
    return df


# ------------------------ PLOT AND STATS ------------------------

def plot_box_violin_stats(df, numeric_cols, dataset_name, outdir):
    df = df.copy()
    # Drop non-numeric features like multiclass_label if exists
    numeric_cols = [c for c in numeric_cols if c in df.columns]

    fig, axes = plt.subplots(len(numeric_cols), 2, figsize=(12, len(numeric_cols) * 4))
    if len(numeric_cols) == 1:
        axes = np.array([axes])

    for i, feat in enumerate(numeric_cols):
        ax_box, ax_violin = axes[i]
        # Skip if feature empty or only one group
        if df['subgroup_binary'].nunique() < 2 or df[feat].dropna().empty:
            print(f"[WARN] Skipping {dataset_name}:{feat} due to empty group")
            continue

        sns.boxplot(x='subgroup_binary', y=feat, data=df, palette='Set2', ax=ax_box)
        ax_box.set_title(f"{dataset_name}: {feat} (Boxplot)")
        sns.violinplot(x='subgroup_binary', y=feat, data=df, palette='Set2', ax=ax_violin)
        ax_violin.set_title(f"{dataset_name}: {feat} (Violin)")

        # Statistics
        group_copd = df[df['subgroup_binary'] == 'COPD'][feat].dropna()
        group_non = df[df['subgroup_binary'] == 'Non-COPD'][feat].dropna()
        if len(group_copd) == 0 or len(group_non) == 0:
            continue
        t_stat, p_val = ttest_ind(group_copd, group_non, equal_var=False)
        d_val = compute_cohens_d(group_copd, group_non)
        ci_low, ci_high = confidence_interval(group_copd, group_non)
        print(
            f"[{dataset_name}][{feat}] t={t_stat:.3f}, p={p_val:.3e}, Cohen's d={d_val:.3f}, 95% CI=({ci_low:.3f},{ci_high:.3f})")

    save_figure(fig, outdir, f"{dataset_name}_COPD_vs_NonCOPD")


# ------------------------ MAIN ANALYSIS ------------------------

def subgroup_binary_analysis(metadata_df: pd.DataFrame, outdir: Path):
    numeric_cols = metadata_df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in ['label']]
    datasets = metadata_df['source'].dropna().unique()
    for ds in datasets:
        sdf = metadata_df[metadata_df['source'] == ds].copy()
        sdf = map_subgroup_binary(sdf, ds)
        plot_box_violin_stats(sdf, numeric_cols, ds, outdir)


# ------------------------ MAIN ------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--outdir", default=None)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = json.load(f)

    output_root = project_root / cfg["output_dir"]
    outdir = Path(args.outdir) if args.outdir else output_root / "subgroup_binary_figures"
    ensure_dir(outdir)

    metadata_df = pd.read_csv(project_root / cfg["all_csv"])
    metadata_df['source'] = metadata_df['source'].astype(str)

    subgroup_binary_analysis(metadata_df, outdir)
    print(f"[INFO] Binary subgroup analysis figures saved to {outdir}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()

# # File: scripts/subgroup_analysis_final_v3.py
# # Usage: python scripts/subgroup_analysis_final_v3.py --config configs/windows_real_paths.json
#
# from pathlib import Path
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from scipy.stats import f_oneway
# import argparse
# import json
# import warnings
#
# def ensure_dir(path: Path):
#     path.mkdir(parents=True, exist_ok=True)
#     return path
#
# def save_fig(fig, outdir, stem):
#     ensure_dir(outdir)
#     fig.savefig(outdir / f"{stem}.png", dpi=400, bbox_inches='tight', facecolor='white')
#     fig.savefig(outdir / f"{stem}.pdf", bbox_inches='tight', facecolor='white')
#     plt.close(fig)
#
# def compute_anova_pvalues(sdf, numeric_cols):
#     subgroups = sdf['subgroup'].dropna().unique()
#     results = []
#     for feat in numeric_cols:
#         groups_data = [sdf[sdf['subgroup']==sg][feat].dropna().values for sg in subgroups if len(sdf[sdf['subgroup']==sg][feat].dropna())>=2]
#         if len(groups_data) >= 2:
#             try:
#                 p = f_oneway(*groups_data).pvalue
#             except Exception:
#                 p = np.nan
#         else:
#             p = np.nan
#         results.append({'feature': feat, 'p_value': p})
#     return pd.DataFrame(results)
#
# def subgroup_analysis(metadata_df: pd.DataFrame, outdir: Path):
#     numeric_cols = metadata_df.select_dtypes(include=np.number).columns.tolist()
#     numeric_cols = [c for c in numeric_cols if c not in ['label','binary_label']]
#
#     datasets = metadata_df['source'].unique()
#     for dataset in datasets:
#         sdf = metadata_df[metadata_df['source'] == dataset].copy()
#
#         # Define subgroups per dataset
#         ds_upper = dataset.upper()
#         if ds_upper in ['ICBHI','RESPIRATORYDATABASE']:
#             sdf['subgroup'] = sdf['label'].map({
#                 0:'Healthy',1:'COPD',2:'URTI',3:'Bronchiectasis',
#                 4:'Bronchiolitis',5:'Pneumonia',6:'LRTI',7:'Asthma'
#             })
#         elif ds_upper in ['KAUH','JWYY9NP4GV-3']:
#             sdf['subgroup'] = sdf['label'].map({
#                 0:'Healthy',1:'COPD',2:'Heart Failure',3:'BRON',
#                 4:'Pneumonia',5:'Asthma'
#             })
#         elif ds_upper in ['TR','RESPIRATORYDATABASE@TR']:
#             # Map COPD0->Healthy, COPD1-4->COPD
#             sdf['subgroup'] = sdf['label'].apply(lambda x: 'Healthy' if x==0 else 'COPD')
#         else:
#             sdf['subgroup'] = sdf['label'].astype(str)
#
#         # Drop missing subgroups
#         sdf = sdf[sdf['subgroup'].notna()]
#         # Impute missing numeric features
#         sdf[numeric_cols] = sdf[numeric_cols].fillna(sdf[numeric_cols].mean())
#
#         # Compute ANOVA p-values
#         pvals = compute_anova_pvalues(sdf, numeric_cols)
#         pvals.to_csv(outdir / f'{dataset}_subgroup_stats.csv', index=False)
#
#         # Filter subgroups with ≥2 samples
#         valid_subgroups = [sg for sg in sdf['subgroup'].unique() if len(sdf[sdf['subgroup']==sg])>=2]
#         plot_sdf = sdf[sdf['subgroup'].isin(valid_subgroups)]
#         if plot_sdf.empty:
#             continue
#
#         # Plot boxplots and violin plots
#         for feat in numeric_cols:
#             feat_sdf = plot_sdf[plot_sdf[feat].notna()]
#             subgroups_with_data = [sg for sg in valid_subgroups if len(feat_sdf[feat_sdf['subgroup']==sg])>=2]
#             if len(subgroups_with_data)<2:
#                 continue  # skip features with insufficient groups
#
#             # Boxplot
#             fig, ax = plt.subplots(figsize=(10,5))
#             sns.boxplot(x='subgroup', y=feat, data=feat_sdf, ax=ax, order=subgroups_with_data)
#             ax.set_title(f'{feat} Boxplot by Subgroup - {dataset}', weight='bold')
#             ax.set_xlabel('Subgroup')
#             ax.set_ylabel(feat)
#             ax.grid(axis='y', alpha=0.3)
#             plt.xticks(rotation=30)
#             save_fig(fig, outdir, f'{dataset}_{feat}_boxplot')
#
#             # Violin plot
#             fig, ax = plt.subplots(figsize=(10,5))
#             sns.violinplot(x='subgroup', y=feat, data=feat_sdf, ax=ax, order=subgroups_with_data,
#                            inner='quartile', palette='pastel')
#             ax.set_title(f'{feat} Violin Plot by Subgroup - {dataset}', weight='bold')
#             ax.set_xlabel('Subgroup')
#             ax.set_ylabel(feat)
#             ax.grid(axis='y', alpha=0.3)
#             plt.xticks(rotation=30)
#             save_fig(fig, outdir, f'{dataset}_{feat}_violin')
#
#         # Heatmap overview of mean features per subgroup
#         grouped = plot_sdf.groupby('subgroup')[numeric_cols].mean()
#         if not grouped.empty:
#             fig, ax = plt.subplots(figsize=(len(numeric_cols)*0.7+2,len(grouped)*0.6+2))
#             sns.heatmap(grouped, annot=True, fmt=".2f", cmap='YlGnBu', cbar_kws={'label':'Mean Value'}, ax=ax)
#             ax.set_title(f'Feature Heatmap by Subgroup - {dataset}', weight='bold')
#             save_fig(fig, outdir, f'{dataset}_features_heatmap')
#
#     print(f"[INFO] Subgroup analysis completed. Figures and CSVs saved to {outdir}")
#
# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--config", required=True)
#     parser.add_argument("--outdir", default=None)
#     args = parser.parse_args()
#
#     with open(args.config,'r',encoding='utf-8') as f:
#         cfg = json.load(f)
#
#     metadata_df = pd.read_csv(cfg['all_csv'])
#     metadata_df['source'] = metadata_df['source'].astype(str).str.upper()
#
#     outdir = Path(args.outdir) if args.outdir else Path(cfg['output_dir']) / "subgroup_analysis_final_v3"
#     ensure_dir(outdir)
#
#     subgroup_analysis(metadata_df, outdir)
#
# if __name__=="__main__":
#     warnings.filterwarnings('ignore')
#     main()