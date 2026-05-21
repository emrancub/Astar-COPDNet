from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from dg_copdnet.config import AppConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = AppConfig.from_json(args.config)
    rp = cfg.dataset_build.real_paths
    paths = {
        "icbhi_root": rp.icbhi_root,
        "icbhi_audio_dir": rp.icbhi_audio_dir,
        "icbhi_demographic_txt": rp.icbhi_demographic_txt,
        "icbhi_filename_differences_txt": rp.icbhi_filename_differences_txt,
        "icbhi_filename_format_txt": rp.icbhi_filename_format_txt,
        "icbhi_patient_diagnosis_csv": rp.icbhi_patient_diagnosis_csv,
        "kauh_root": rp.kauh_root,
        "kauh_audio_dir": rp.kauh_audio_dir,
        "kauh_stethoscope_dir": rp.kauh_stethoscope_dir,
        "kauh_annotation_xlsx": rp.kauh_annotation_xlsx,
        "tr_root": rp.tr_root,
        "tr_audio_dir": rp.tr_audio_dir,
        "tr_labels_xlsx": rp.tr_labels_xlsx,
    }
    for name, value in paths.items():
        p = Path(value)
        print(f"{name:32s} | exists={p.exists()} | {value}")


if __name__ == "__main__":
    main()
