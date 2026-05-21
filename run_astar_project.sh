#!/usr/bin/env bash
set -e
export PYTHONPATH="$PWD/src"
python scripts/run_astar_project.py --config configs/astar_real_paths.json --task both --seeds 42,43,44
python scripts/make_publication_figures_astar.py --output_dir outputs_astar
