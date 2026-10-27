from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from dg_copdnet.config import AppConfig
from dg_copdnet.data.metadata_builder import build_metadata_from_real_paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = AppConfig.from_json(args.config)
    internal_csv, external_csv, all_csv, summary = build_metadata_from_real_paths(cfg)
    print("Metadata built successfully")
    print("internal_csv:", internal_csv)
    print("external_csv:", external_csv)
    print("all_csv:", all_csv)
    print(summary)


if __name__ == "__main__":
    main()
