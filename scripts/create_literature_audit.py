from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from dg_copdnet.analysis.literature_audit import create_literature_audit_template


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/literature_audit")
    args = parser.parse_args()
    out = create_literature_audit_template(args.output_dir)
    print(out)


if __name__ == "__main__":
    main()
