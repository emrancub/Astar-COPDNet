
from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
STEPS=[
    'scripts/check_real_paths.py',
    'scripts/build_metadata.py',
    'scripts/signal_analysis_report.py',
    'scripts/train_standard.py',
    'scripts/deep_baseline_benchmark.py',
    'scripts/feature_benchmark.py',
    'scripts/feature_combo_benchmark.py',
    'scripts/optimizer_benchmark.py',
    'scripts/independent_analysis.py',
    'scripts/multiclass_dataset_analysis.py',
    'scripts/blind_experiments.py',
    'scripts/validation_strategy_benchmark.py',
    'scripts/stats_tests.py',
    'scripts/generate_interpretability_figures.py',
    'scripts/figure_superpack.py',
]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config', required=True); ap.add_argument('--python', default=sys.executable); args=ap.parse_args()
    for s in STEPS:
        cmd=[args.python, str(ROOT/s), '--config', args.config]
        print('Running:', ' '.join(cmd))
        subprocess.run(cmd, check=True)
if __name__=='__main__': main()
