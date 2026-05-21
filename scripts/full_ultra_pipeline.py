
from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

STEPS=[
    ('check_real_paths.py','Path checking'),
    ('build_metadata.py','Metadata build'),
    ('signal_analysis_report.py','Signal analysis report'),
    ('train_standard.py','Standard proposed model binary CV'),
    ('deep_baseline_benchmark.py','Deep baseline benchmark'),
    ('feature_benchmark.py','Classical feature benchmark'),
    ('feature_combo_benchmark.py','Feature-combination benchmark'),
    ('optimizer_benchmark.py','Optimizer benchmark'),
    ('independent_analysis.py','Independent binary dataset analysis'),
    ('multiclass_dataset_analysis.py','Independent multiclass analysis'),
    ('blind_experiments.py','Blind cross-dataset analysis'),
    ('validation_strategy_benchmark.py','Validation strategy benchmark'),
    ('stats_tests.py','Statistical tests'),
    ('figure_superpack.py','Advanced figure pack')
]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config', required=True); ap.add_argument('--python', default=sys.executable); args=ap.parse_args()
    for script, desc in STEPS:
        print(f'\n=== {desc} ===')
        cmd=[args.python, str(ROOT/'scripts'/script), '--config', args.config]
        print(' '.join(cmd))
        subprocess.run(cmd, check=True)

if __name__=='__main__':
    main()
