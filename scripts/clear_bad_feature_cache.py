from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import numpy as np
from dg_copdnet.config import AppConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = AppConfig.from_json(args.config)
    cache_dir = Path(cfg.features.cache_dir)
    if not cache_dir.exists():
        print(f"Cache dir does not exist: {cache_dir}")
        return
    removed = 0
    checked = 0
    for fp in cache_dir.glob("*.npy"):
        checked += 1
        try:
            arr = np.load(fp)
            if not np.all(np.isfinite(arr)):
                fp.unlink()
                removed += 1
        except Exception:
            fp.unlink(missing_ok=True)
            removed += 1
    print({"checked": checked, "removed": removed, "cache_dir": str(cache_dir)})


if __name__ == "__main__":
    main()
