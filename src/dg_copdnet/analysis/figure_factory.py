
from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns

from dg_copdnet.config import AppConfig
from dg_copdnet.utils.io import ensure_dir
from dg_copdnet.utils.plots import savefig


def _plot_confusion(cm, labels, title, path):
    plt.figure(figsize=(5,4))
    sns.heatmap(np.array(cm), annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted'); plt.ylabel('True'); plt.title(title)
    savefig(path)


def _plot_tsne(embeddings, labels, title, path):
    if len(embeddings) < 5: return
    emb2 = TSNE(n_components=2, init='random', learning_rate='auto', perplexity=min(30, max(2, len(embeddings)//4)), random_state=42).fit_transform(embeddings)
    plt.figure(figsize=(6,5))
    for cls in sorted(set(labels)):
        idx = np.array(labels)==cls
        plt.scatter(emb2[idx,0], emb2[idx,1], s=18, alpha=0.8, label=str(cls))
    plt.legend(title='Class'); plt.title(title)
    savefig(path)


def build_advanced_figure_pack(cfg: AppConfig):
    out = ensure_dir(Path(cfg.output_dir) / 'advanced_figures')
    created=[]
    # Use summaries from known outputs
    # 1. optimizer benchmark
    p=Path(cfg.output_dir)/'optimizer_benchmark'/'optimizer_benchmark.csv'
    if p.exists():
        df=pd.read_csv(p)
        plt.figure(figsize=(7,4)); plt.bar(df['optimizer'], df['auc']); plt.ylabel('AUC'); plt.title('Optimizer benchmark'); savefig(out/'optimizer_benchmark_auc.png'); created.append('optimizer_benchmark_auc.png')
    # 2. feature benchmark
    p=Path(cfg.output_dir)/'feature_benchmark'/'feature_benchmark.csv'
    if p.exists():
        df=pd.read_csv(p)
        piv=df.pivot(index='feature_set', columns='model', values='auc')
        plt.figure(figsize=(10,5))
        for col in piv.columns: plt.plot(range(len(piv.index)), piv[col].values, marker='o', label=col)
        plt.xticks(range(len(piv.index)), piv.index, rotation=35, ha='right'); plt.ylabel('AUC'); plt.title('Feature benchmark AUC'); plt.legend(); savefig(out/'feature_benchmark_auc.png'); created.append('feature_benchmark_auc.png')
    # 3. Deep baselines summary
    p=Path(cfg.output_dir)/'deep_baseline_benchmark'
    if p.exists():
        rows=[]
        for sub in p.iterdir():
            s=sub/'summary.csv'
            if s.exists():
                df=pd.read_csv(s)
                rows.append({'model':sub.name,'auc':df['auc'].mean() if 'auc' in df else np.nan,'f1':df['f1'].mean() if 'f1' in df else np.nan,'param_count':df['param_count'].mean() if 'param_count' in df else np.nan})
        if rows:
            bdf=pd.DataFrame(rows)
            bdf.to_csv(out/'deep_baseline_summary.csv', index=False)
            plt.figure(figsize=(8,4)); plt.bar(bdf['model'], bdf['auc']); plt.xticks(rotation=30, ha='right'); plt.ylabel('Mean AUC'); plt.title('Deep baseline benchmark'); savefig(out/'deep_baseline_auc.png'); created.append('deep_baseline_auc.png')
            plt.figure(figsize=(6,5)); plt.scatter(bdf['param_count'], bdf['auc']);
            for _,r in bdf.iterrows(): plt.text(r['param_count'], r['auc'], r['model'])
            plt.xlabel('Parameter count'); plt.ylabel('Mean AUC'); plt.title('Performance vs parameter count'); savefig(out/'params_vs_auc.png'); created.append('params_vs_auc.png')
    # 4. Confusion matrices and tsne from advanced outputs
    for folder in Path(cfg.output_dir).glob('**/fold_1'):
        metrics_file = folder/'best_metrics.json'
        if metrics_file.exists():
            met=json.loads(metrics_file.read_text())
            cm=met.get('confusion_matrix')
            if cm:
                labels=[str(i) for i in range(len(cm))] if len(cm)>2 else ['non-COPD','COPD']
                _plot_confusion(cm, labels, f'Confusion matrix: {folder.parent.parent.name}/{folder.parent.name}', out/f'{folder.parent.parent.name}_{folder.parent.name}_confusion.png')
                created.append(f'{folder.parent.parent.name}_{folder.parent.name}_confusion.png')
        embf = folder/'val_embeddings.npy'; yf = folder/'val_y_true.npy'
        if embf.exists() and yf.exists():
            emb=np.load(embf); y=np.load(yf)
            _plot_tsne(emb, y, f't-SNE: {folder.parent.parent.name}/{folder.parent.name}', out/f'{folder.parent.parent.name}_{folder.parent.name}_tsne.png')
            created.append(f'{folder.parent.parent.name}_{folder.parent.name}_tsne.png')
    # 5. Signal analysis examples copy manifest count
    sig_dir = Path(cfg.output_dir)/'signal_analysis'
    if sig_dir.exists():
        for img in sig_dir.glob('*.png'):
            created.append(img.name)
    # 6. Interpretability figures
    it_dir = Path(cfg.output_dir)/'interpretability'
    if it_dir.exists():
        for img in it_dir.glob('*.png'):
            created.append(img.name)
    manifest={'created_figures':created,'num_figures':len(created),'target_minimum':30}
    (out/'figure_manifest.json').write_text(json.dumps(manifest, indent=2))
    return manifest
