from __future__ import annotations

from pathlib import Path
import pandas as pd

from dg_copdnet.utils.io import ensure_dir


def create_literature_audit_template(output_dir: str | Path):
    output_dir = ensure_dir(output_dir)
    rows = [
        {"method_name": "Altan_and_Kutlu_2020", "modality": "audio", "dataset_used_in_original_paper": "RespiratoryDatabase@TR", "reproducible_on_same_dataset": "yes", "notes": "Use TR-only benchmark"},
        {"method_name": "AeroCOPDNet", "modality": "audio", "dataset_used_in_original_paper": "merged_public_respiratory_datasets", "reproducible_on_same_dataset": "yes", "notes": "Reproduce on pooled ICBHI+KAUH if code is available"},
        {"method_name": "TriSpectraKAN", "modality": "audio", "dataset_used_in_original_paper": "three_public_lung_sound_datasets", "reproducible_on_same_dataset": "partial", "notes": "Need exact dataset mapping"},
        {"method_name": "MODBN", "modality": "tabular", "dataset_used_in_original_paper": "Kaggle symptom dataset", "reproducible_on_same_dataset": "no", "notes": "Move to related work unless recreated fairly"},
        {"method_name": "Chen_et_al_CT", "modality": "ct", "dataset_used_in_original_paper": "low_dose_CT", "reproducible_on_same_dataset": "no", "notes": "Not part of audio benchmark"},
    ]
    df = pd.DataFrame(rows)
    out = Path(output_dir) / "method_dataset_audit.csv"
    df.to_csv(out, index=False)
    return out
