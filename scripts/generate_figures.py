from __future__ import annotations
import sys
from pathlib import Path
import argparse
import pandas as pd

# =======================
# Project paths
# =======================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# =======================
# Imports
# =======================
from dg_copdnet.config import AppConfig
from dg_copdnet.analysis import signal_analysis, figure_factory

# =======================
# Dynamic function selection
# =======================
def get_main_function(module, hints, description):
    """Pick the most likely function from a module based on hints. Excludes classes."""
    for name in dir(module):
        attr = getattr(module, name)
        if callable(attr) and not name.startswith("_") and not isinstance(attr, type):
            for hint in hints:
                if hint.lower() in name.lower():
                    print(f"[INFO] Using '{name}' for {description}.")
                    return attr
    # fallback: pick first callable function
    for name in dir(module):
        attr = getattr(module, name)
        if callable(attr) and not name.startswith("_") and not isinstance(attr, type):
            print(f"[INFO] Fallback: using '{name}' for {description}.")
            return attr
    raise ImportError(f"No suitable function found in {module} for {description}.")

run_signal_analysis = get_main_function(
    signal_analysis,
    hints=["build_signal_report", "run_signal_analysis", "perform_signal_analysis"],
    description="signal analysis"
)

generate_summary_figures = get_main_function(
    figure_factory,
    hints=["build_advanced_figure_pack", "generate_summary_figures", "create_summary_figures"],
    description="figure generation"
)

# =======================
# Main
# =======================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    # Load config and data
    cfg = AppConfig.from_json(args.config)
    all_df = pd.read_csv(cfg.all_csv)

    # Run signal analysis
    run_signal_analysis(cfg, all_df)  # cfg + DataFrame

    # Generate figures
    generate_summary_figures(cfg)  # <-- pass full cfg object

if __name__ == "__main__":
    main()

# from __future__ import annotations
#
# import argparse
# import sys
# from pathlib import Path
#
# PROJECT_ROOT = Path(__file__).resolve().parents[1]
# SRC_ROOT = PROJECT_ROOT / "src"
# if str(SRC_ROOT) not in sys.path:
#     sys.path.insert(0, str(SRC_ROOT))
#
# import pandas as pd
#
# from dg_copdnet.config import AppConfig
# from dg_copdnet.analysis.signal_analysis import run_signal_analysis
# from dg_copdnet.analysis.figure_factory import generate_summary_figures
#
#
# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--config", required=True)
#     args = parser.parse_args()
#     cfg = AppConfig.from_json(args.config)
#     all_df = pd.read_csv(cfg.all_csv)
#     run_signal_analysis(all_df, cfg, Path(cfg.output_dir) / "signal_analysis")
#     manifest = generate_summary_figures(cfg.output_dir)
#     print(manifest)
#
#
# if __name__ == "__main__":
#     main()
