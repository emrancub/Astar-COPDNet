from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class Step:
    number: int
    name: str
    command: List[str]
    description: str


def build_steps(config_path: str, optimizer: str) -> List[Step]:
    py = sys.executable
    return [
        Step(1, "Check dataset paths", [py, "scripts/check_real_paths.py", "--config", config_path],
             "Verify that all real Windows dataset paths and key files exist."),
        Step(2, "Build metadata", [py, "scripts/build_metadata.py", "--config", config_path],
             "Create metadata CSV files from ICBHI, KAUH, and RespiratoryDatabase@TR."),
        Step(3, "Standard cross-validation training", [py, "-c", f"import subprocess,sys; subprocess.check_call([sys.executable, 'scripts/clear_bad_feature_cache.py', '--config', '{config_path}']); subprocess.check_call([sys.executable, 'scripts/train_standard.py', '--config', '{config_path}', '--optimizer', '{optimizer}'])"],
             "Clear invalid cached features and run the main grouped cross-validation training experiment."),
        Step(4, "Independent dataset analysis", [py, "scripts/independent_analysis.py", "--config", config_path],
             "Run the same pipeline independently on ICBHI, KAUH, and RespiratoryDatabase@TR."),
        Step(5, "Blind experiments", [py, "scripts/blind_experiments.py", "--config", config_path],
             "Run leave-one-dataset-out blind experiments."),
        Step(6, "Feature benchmark", [py, "scripts/feature_benchmark.py", "--config", config_path],
             "Compare log-Mel, MFCC, LPCC, phonation, and other supported features."),
        Step(7, "Optimizer benchmark", [py, "scripts/optimizer_benchmark.py", "--config", config_path],
             "Compare Adam, AdamW, RMSprop, and SGD."),
        Step(8, "Validation strategy benchmark", [py, "scripts/validation_strategy_benchmark.py", "--config", config_path],
             "Compare grouped CV, grouped holdout, and leave-one-dataset-out strategies."),
        Step(9, "Statistical tests", [py, "scripts/stats_tests.py", "--config", config_path],
             "Compute p-values, confidence intervals, and summary statistics from experiment outputs."),
        Step(10, "Generate figures", [py, "scripts/generate_figures.py", "--config", config_path],
             "Generate manuscript-ready plots and figure panels."),
    ]


def print_header(title: str) -> None:
    line = "=" * 88
    print(f"\n{line}\n{title}\n{line}")


def run_step(step: Step, project_root: Path, log_dir: Path) -> int:
    print_header(f"STEP {step.number}: {step.name}")
    print(step.description)
    print("\nCommand:")
    print("  " + " ".join(step.command))

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_file = log_dir / f"step_{step.number:02d}_{step.name.lower().replace(' ', '_')}_{timestamp}.log"
    print(f"\nLogging to: {log_file}")

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    with log_file.open("w", encoding="utf-8") as f:
        proc = subprocess.Popen(
            step.command,
            cwd=str(project_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            f.write(line)
        proc.wait()
        return proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the full DG-COPDNet project step by step.")
    parser.add_argument("--config", default="configs/windows_real_paths.json", help="Path to config JSON.")
    parser.add_argument("--optimizer", default="adamw", help="Optimizer for the standard training step.")
    parser.add_argument("--start-step", type=int, default=1, help="Start from this step number.")
    parser.add_argument("--end-step", type=int, default=10, help="Stop after this step number.")
    parser.add_argument("--yes", action="store_true", help="Run without asking for confirmation between steps.")
    parser.add_argument("--project-root", default=".", help="Project root directory.")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    log_dir = project_root / "outputs" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    steps = build_steps(args.config, args.optimizer)
    selected_steps = [s for s in steps if args.start_step <= s.number <= args.end_step]

    if not selected_steps:
        print("No steps selected. Check --start-step and --end-step.")
        return 1

    print_header("DG-COPDNet Full Step-by-Step Runner")
    print(f"Project root : {project_root}")
    print(f"Config file  : {args.config}")
    print(f"Optimizer    : {args.optimizer}")
    print(f"Step range   : {args.start_step} to {args.end_step}")
    print(f"Logs folder  : {log_dir}")

    for step in selected_steps:
        if not args.yes:
            choice = input(f"\nPress Enter to run STEP {step.number} ({step.name}), or type 'q' to quit: ").strip().lower()
            if choice == "q":
                print("Stopped by user.")
                return 0

        code = run_step(step, project_root, log_dir)
        if code != 0:
            print_header(f"STEP {step.number} FAILED")
            print(f"Step name : {step.name}")
            print(f"Exit code : {code}")
            print("Fix the error, then resume with for example:")
            print(f"  python {Path(__file__).name} --config {args.config} --start-step {step.number} --end-step 10")
            return code

        print_header(f"STEP {step.number} COMPLETED SUCCESSFULLY")

    print_header("ALL SELECTED STEPS COMPLETED SUCCESSFULLY")
    print("You can now check the outputs/ folder and logs in outputs/logs/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
