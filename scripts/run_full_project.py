from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import subprocess


def run(cmd):
    print(">>>", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run([sys.executable, "scripts/check_real_paths.py", "--config", args.config])
    run([sys.executable, "scripts/build_metadata.py", "--config", args.config])
    run([sys.executable, "scripts/train_standard.py", "--config", args.config])
    run([sys.executable, "scripts/independent_analysis.py", "--config", args.config])
    run([sys.executable, "scripts/blind_experiments.py", "--config", args.config])
    run([sys.executable, "scripts/feature_benchmark.py", "--config", args.config])
    run([sys.executable, "scripts/optimizer_benchmark.py", "--config", args.config])
    run([sys.executable, "scripts/validation_strategy_benchmark.py", "--config", args.config])
    run([sys.executable, "scripts/generate_figures.py", "--config", args.config])


if __name__ == "__main__":
    main()
