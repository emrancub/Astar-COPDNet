# Astar-COPDNet


Astar-COPDNet is a deep learning framework for analyzing COPD (Chronic Obstructive Pulmonary Disease) data. It integrates data processing, model training, and evaluation pipelines with advanced feature and interpretability analysis.
This repository accompanies the manuscript submitted by the authors, providing code, configuration files, and scripts to replicate all experiments.

---

## Quick start

### 1. Open in PyCharm
Open this folder in PyCharm and mark `src/` as a **Sources Root**.

### 2. Create environment and install dependencies
Install a CUDA-enabled PyTorch build first, then:

```bash
pip install -r requirements.txt
```

### 3. Check the real paths
```bash
python scripts/check_real_paths.py --config configs/windows_real_paths.json
```

### 4. Build metadata CSV files
```bash
python scripts/build_metadata.py --config configs/windows_real_paths.json
```

### 5. Run the standard internal experiment
```bash
python scripts/train_standard.py --config configs/windows_real_paths.json
```

### 6. Run independent per-dataset analysis
```bash
python scripts/independent_analysis.py --config configs/windows_real_paths.json
```

### 7. Run blind leave-one-dataset-out experiments
```bash
python scripts/blind_experiments.py --config configs/windows_real_paths.json
```

### 8. Benchmark features with classical baselines
```bash
python scripts/feature_benchmark.py --config configs/windows_real_paths.json
```

### 9. Compare optimizers
```bash
python scripts/optimizer_benchmark.py --config configs/windows_real_paths.json
```

### 10. Compare validation strategies
```bash
python scripts/validation_strategy_benchmark.py --config configs/windows_real_paths.json
```

### 11. Run statistical tests
```bash
python scripts/stats_tests.py --config configs/windows_real_paths.json
```

### 12. Generate figures
```bash
python scripts/generate_figures.py --config configs/windows_real_paths.json
```

## Key folders

```text
Astar-COPDNet/
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
├── pyproject.toml
├── run_astar_project.bat
├── run_astar_project.sh
├── run_pipeline_step_by_step.bat
├── run_pipeline_step_by_step.py
├── run_ultra_pipeline.py
├── configs/
│   ├── astar_real_paths.json
│   ├── default.json
│   └── windows_real_paths.json
├── scripts/
│   ├── ablation_study.py
│   ├── advanced_research_pipeline.py
│   ├── ...
├── src/
│   └── dg_copdnet/
│       ├── __init__.py
│       ├── config.py
│       ├── analysis/
│       ├── data/
│       ├── models/
│       ├── training/
│       └── utils/
└── FIX_STEP3_LIST_STATS_ERROR.txt
```

## One-click step-by-step runner

If you want to run the whole project in order and stop after each stage, use:

```bash
python run_pipeline_step_by_step.py --config configs/windows_real_paths.json
```

Useful options:

```bash
python run_pipeline_step_by_step.py --config configs/windows_real_paths.json --yes
python run_pipeline_step_by_step.py --config configs/windows_real_paths.json --start-step 3 --end-step 10
python run_pipeline_step_by_step.py --config configs/windows_real_paths.json --optimizer adamw
```

For Windows, you can also double-click:

```text
run_pipeline_step_by_step.bat
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/emrancub/Astar-COPDNet.git
cd Astar-COPDNet
```
## Contact
**Corresponding Author:** Dong-Jun Yu (Email: njyudj@njust.edu.cn)  
**Contact:** Md Emran Hasan (Email: mdemranhasan@njust.edu.cn)