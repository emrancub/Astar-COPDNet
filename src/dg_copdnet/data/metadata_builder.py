
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

from dg_copdnet.config import AppConfig, DatasetPathsConfig
from dg_copdnet.utils.io import ensure_dir, save_json

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma"}


def _audio_files(folder: str | Path) -> list[Path]:
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Audio folder not found: {folder}")
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_EXTS])


def _normalize_dx(x: str) -> str:
    x = str(x).strip()
    x = re.sub(r"\s+", " ", x)
    return x.upper()


def _safe_multiclass_map(series: pd.Series) -> tuple[pd.Series, dict[str, int]]:
    cats = sorted(series.fillna("UNKNOWN").astype(str).unique().tolist())
    mapping = {c: i for i, c in enumerate(cats)}
    return series.fillna("UNKNOWN").astype(str).map(mapping).astype(int), mapping


def _read_icbhi_diagnosis(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, names=["patient_id_raw", "diagnosis_raw"])
    df["patient_id_raw"] = df["patient_id_raw"].astype(str).str.strip()
    df["diagnosis_raw"] = df["diagnosis_raw"].astype(str).str.strip().map(_normalize_dx)
    return df


def _read_icbhi_demographics(path: str | Path) -> pd.DataFrame:
    rows = []
    for line in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = re.split(r"\s+", line.strip())
        if len(parts) < 6:
            continue
        rows.append({"patient_id_raw": parts[0], "age": parts[1], "sex": parts[2], "bmi": parts[3], "weight_kg": parts[4], "height_cm": parts[5]})
    df = pd.DataFrame(rows)
    for c in ["age", "bmi", "weight_kg", "height_cm"]:
        if c in df:
            df[c] = pd.to_numeric(df[c].replace({"NA": None}), errors="coerce")
    return df


def _icbhi_parse_filename(stem: str) -> dict[str, Any]:
    parts = stem.split("_")
    return {
        "recording_id": stem,
        "patient_id_raw": parts[0] if parts else None,
        "recording_index": parts[1] if len(parts) > 1 else None,
        "chest_location": parts[2] if len(parts) > 2 else None,
        "acquisition_mode": parts[3] if len(parts) > 3 else None,
        "device": "_".join(parts[4:]) if len(parts) > 4 else None,
    }


def build_icbhi_metadata(paths: DatasetPathsConfig) -> pd.DataFrame:
    diag_df = _read_icbhi_diagnosis(paths.icbhi_patient_diagnosis_csv)
    demo_df = _read_icbhi_demographics(paths.icbhi_demographic_txt)
    diag_map = diag_df.set_index("patient_id_raw")["diagnosis_raw"].to_dict()
    demo_map = demo_df.set_index("patient_id_raw").to_dict(orient="index") if not demo_df.empty else {}
    rows = []
    for wav_path in _audio_files(paths.icbhi_audio_dir):
        meta = _icbhi_parse_filename(wav_path.stem)
        pid = str(meta["patient_id_raw"])
        diagnosis = diag_map.get(pid)
        if diagnosis is None:
            continue
        diagnosis = _normalize_dx(diagnosis)
        demo = demo_map.get(pid, {})
        rows.append({
            "file_path": str(wav_path),
            "label": 1 if diagnosis == "COPD" else 0,
            "binary_label": 1 if diagnosis == "COPD" else 0,
            "multiclass_name": diagnosis,
            "patient_id": f"ICBHI_{pid}",
            "patient_id_raw": pid,
            "source": "ICBHI",
            "diagnosis_raw": diagnosis,
            "age": demo.get("age"), "sex": demo.get("sex"), "bmi": demo.get("bmi"),
            "weight_kg": demo.get("weight_kg"), "height_cm": demo.get("height_cm"),
            "recording_index": meta.get("recording_index"), "chest_location": meta.get("chest_location"),
            "acquisition_mode": meta.get("acquisition_mode"), "device": meta.get("device"),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df['multiclass_label'], mapping = _safe_multiclass_map(df['multiclass_name'])
        df.attrs['multiclass_mapping'] = mapping
    return df


def _kauh_parse_audio_filename(stem: str) -> dict[str, Any]:
    out = {"patient_id_raw": None, "diagnosis_raw": None, "sound_type": None, "chest_location": None, "age": None, "sex": None}
    if "_" not in stem:
        out["patient_id_raw"] = stem
        return out
    patient_token, payload = stem.split("_", 1)
    out["patient_id_raw"] = patient_token.strip()
    parts = [p.strip() for p in payload.split(",")]
    if len(parts) >= 1: out["diagnosis_raw"] = parts[0]
    if len(parts) >= 2: out["sound_type"] = parts[1]
    if len(parts) >= 3: out["chest_location"] = parts[2]
    if len(parts) >= 4: out["age"] = pd.to_numeric(parts[3], errors="coerce")
    if len(parts) >= 5: out["sex"] = parts[4]
    return out


def _load_kauh_annotation(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    try:
        df = pd.read_excel(p)
        df.columns = [str(c).strip() for c in df.columns]
        return df
    except Exception:
        return pd.DataFrame()


def build_kauh_metadata(paths: DatasetPathsConfig) -> pd.DataFrame:
    ann = _load_kauh_annotation(paths.kauh_annotation_xlsx)
    rows = []
    for audio_path in _audio_files(paths.kauh_audio_dir):
        meta = _kauh_parse_audio_filename(audio_path.stem)
        diagnosis = _normalize_dx(meta.get("diagnosis_raw") or "UNKNOWN")
        label = 1 if "COPD" in diagnosis else 0
        pid = str(meta.get("patient_id_raw") or audio_path.stem)
        rows.append({
            "file_path": str(audio_path), "label": label, "binary_label": label,
            "multiclass_name": diagnosis,
            "patient_id": f"KAUH_{pid}", "patient_id_raw": pid, "source": "KAUH",
            "diagnosis_raw": diagnosis, "age": meta.get("age"), "sex": meta.get("sex"),
            "bmi": None, "weight_kg": None, "height_cm": None, "recording_index": None,
            "chest_location": meta.get("chest_location"), "acquisition_mode": "single_channel",
            "device": "KAUH_stethoscope", "sound_type": meta.get("sound_type"),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df['multiclass_label'], mapping = _safe_multiclass_map(df['multiclass_name'])
        df.attrs['multiclass_mapping'] = mapping
    if not ann.empty and 'Diagnosis' in ann.columns:
        df.attrs['annotation_rows'] = int(len(ann))
    return df


def _read_tr_labels(path: str | Path) -> pd.DataFrame:
    xls = pd.ExcelFile(path)
    dfs = []
    for sheet in xls.sheet_names:
        try:
            df = pd.read_excel(path, sheet_name=sheet)
        except Exception:
            continue
        if not df.empty:
            dfs.append(df)
    if not dfs:
        raise ValueError('No usable sheets found in TR labels file.')
    df = pd.concat(dfs, ignore_index=True)
    cols = {str(c).strip().lower(): c for c in df.columns}
    pid_col = cols.get('patient id', list(df.columns)[0])
    dx_col = cols.get('diagnosis', list(df.columns)[1])
    out = df[[pid_col, dx_col]].copy()
    out.columns = ['patient_id_raw', 'diagnosis_raw']
    out['patient_id_raw'] = out['patient_id_raw'].astype(str).str.strip().str.upper()
    out['diagnosis_raw'] = out['diagnosis_raw'].astype(str).str.strip().str.upper()
    out['label'] = out['diagnosis_raw'].apply(lambda x: 0 if x == 'COPD0' else 1)
    return out.drop_duplicates()


def _extract_tr_patient_id(stem: str) -> str:
    m = re.search(r'([A-Za-z]+\d+)', stem)
    return m.group(1).upper() if m else stem.upper()


def build_tr_metadata(paths: DatasetPathsConfig, role: str = 'external') -> pd.DataFrame:
    labels_df = _read_tr_labels(paths.tr_labels_xlsx)
    label_map = labels_df.set_index('patient_id_raw').to_dict(orient='index')
    rows, unmatched = [], []
    for audio_path in _audio_files(paths.tr_audio_dir):
        pid = _extract_tr_patient_id(audio_path.stem)
        rec = label_map.get(pid)
        if rec is None:
            unmatched.append(audio_path.name)
            continue
        dx = _normalize_dx(rec['diagnosis_raw'])
        rows.append({
            'file_path': str(audio_path), 'label': int(rec['label']), 'binary_label': int(rec['label']),
            'multiclass_name': dx,
            'patient_id': f'TR_{pid}', 'patient_id_raw': pid, 'source': 'TR', 'diagnosis_raw': dx,
            'age': None, 'sex': None, 'bmi': None, 'weight_kg': None, 'height_cm': None,
            'recording_index': None, 'chest_location': None, 'acquisition_mode': None, 'device': None,
            'dataset_role': role,
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        ordered = ['COPD0','COPD1','COPD2','COPD3','COPD4']
        present = [x for x in ordered if x in df['multiclass_name'].unique().tolist()]
        mapping = {c:i for i,c in enumerate(present)}
        for x in sorted(set(df['multiclass_name'])-set(present)):
            mapping[x]=len(mapping)
        df['multiclass_label'] = df['multiclass_name'].map(mapping).astype(int)
        df.attrs['multiclass_mapping'] = mapping
    df.attrs['unmatched_files'] = unmatched
    return df


def summarize_dataset(df: pd.DataFrame, name: str) -> dict:
    return {
        'name': name,
        'num_records': int(len(df)),
        'num_patients': int(df['patient_id'].nunique()) if not df.empty else 0,
        'binary_labels': {str(k): int(v) for k, v in df['label'].value_counts(dropna=False).to_dict().items()} if 'label' in df else {},
        'multiclass_labels': {str(k): int(v) for k, v in df['multiclass_name'].value_counts(dropna=False).to_dict().items()} if 'multiclass_name' in df else {},
        'sources': {str(k): int(v) for k, v in df['source'].value_counts(dropna=False).to_dict().items()} if 'source' in df else {},
    }


def build_metadata_from_real_paths(cfg: AppConfig):
    if cfg.dataset_build is None or cfg.dataset_build.real_paths is None:
        raise ValueError('dataset_build.real_paths is required')
    paths = cfg.dataset_build.real_paths
    meta_dir = ensure_dir(cfg.dataset_build.metadata_output_dir)
    icbhi = build_icbhi_metadata(paths); icbhi['dataset_role']='internal'
    kauh = build_kauh_metadata(paths); kauh['dataset_role']='internal'
    tr = build_tr_metadata(paths, role='external')
    internal_parts=[]
    for src in cfg.dataset_build.internal_sources:
        if src.upper()=='ICBHI': internal_parts.append(icbhi)
        elif src.upper()=='KAUH': internal_parts.append(kauh)
        elif src.upper()=='TR': internal_parts.append(tr)
    internal_df = pd.concat(internal_parts, ignore_index=True) if internal_parts else pd.DataFrame()
    external_df = tr.copy()
    all_df = pd.concat([icbhi, kauh, tr], ignore_index=True)
    internal_csv = meta_dir / 'internal_metadata.csv'
    external_csv = meta_dir / 'external_metadata_tr.csv'
    all_csv = meta_dir / 'all_datasets_metadata.csv'
    internal_df.to_csv(internal_csv, index=False)
    external_df.to_csv(external_csv, index=False)
    all_df.to_csv(all_csv, index=False)
    summary = {
        'ICBHI': summarize_dataset(icbhi, 'ICBHI'),
        'KAUH': summarize_dataset(kauh, 'KAUH'),
        'TR': summarize_dataset(tr, 'TR'),
        'internal': summarize_dataset(internal_df, 'internal'),
        'all': summarize_dataset(all_df, 'all'),
        'tr_unmatched_audio_files': tr.attrs.get('unmatched_files', []),
    }
    save_json(summary, meta_dir / 'dataset_summary.json')
    return internal_csv, external_csv, all_csv, summary
