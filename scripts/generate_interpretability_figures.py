
from __future__ import annotations
import argparse, sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import pandas as pd
from dg_copdnet.config import AppConfig
from dg_copdnet.analysis.interpretability import generate_saliency_examples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--checkpoint', default='')
    ap.add_argument('--data', choices=['internal', 'external', 'all'], default='internal')
    ap.add_argument('--output-dir', default='outputs/interpretability')
    args = ap.parse_args()
    cfg = AppConfig.from_json(args.config)
    if args.data == 'internal':
        df = pd.read_csv(cfg.internal_csv)
    elif args.data == 'external':
        df = pd.read_csv(cfg.external_csv)
    else:
        df = pd.read_csv(cfg.all_csv)
    ckpt = args.checkpoint.strip()
    if not ckpt:
        candidates = list((Path(cfg.output_dir) / 'standard_cv').glob('fold_*/best.pt'))
        if not candidates:
            raise FileNotFoundError('No checkpoint provided and none found under outputs/standard_cv/fold_*/best.pt')
        ckpt = str(candidates[0])
    saved = generate_saliency_examples(cfg, df, ckpt, args.output_dir, label_col='label')
    print({'checkpoint': ckpt, 'saved_files': saved})


if __name__ == '__main__':
    main()
