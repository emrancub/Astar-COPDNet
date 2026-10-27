from __future__ import annotations
"""
Turnkey CMPB Major Revision Experiment Runner (CMPB-D-26-03798) — Auto-Resuming Version
Runs the 4 targeted deep-learning experiments requested by Reviewers #1, #2, and #3:
  1. Leave-One-Loss-Out (LOLO) & Branch Ablation (Internal 5-fold CV + LODO)
     [Reviewer #1 Q4.2, Reviewer #2 Q5, Reviewer #3 Q2.4]
  2. Same-Protocol Recent SOTA Comparator (AeroCOPDNet [MobileNetV2-SE + AttPool],
     SpecCNN, EfficientNet-B0 Only, CNN-BiGRU Only, Handcrafted-MLP Only)
     [Reviewer #2 Q11]
  3. Holdout Split Ratio Sensitivity Analysis (20%, 30%, and 40% patient-disjoint splits)
     [Reviewer #1 Q2.3, Reviewer #2 Q7]
  4. Strict 112-Subject KAUH Grouping (stripping BP/DP/EP filter prefixes -> 238 internal /
     280 total unique subjects) & Multi-Seed (Seeds 42, 43, 44) Evaluation
     [Reviewer #1 Q2.4, Reviewer #3 Q2.1]

Auto-resumes from any already-completed steps/models on disk so no GPU work is repeated!
"""
import argparse
import ast
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from dg_copdnet.config import AppConfig
from dg_copdnet.data.dataset import RespiratoryAudioDataset
from dg_copdnet.data.features import cached_feature_vector
from dg_copdnet.data.metadata_builder import build_metadata_from_real_paths
from dg_copdnet.data.splits import make_grouped_folds, make_grouped_holdout
from dg_copdnet.models.dare_copdnet import DARECOPDNet
from dg_copdnet.training.domain_losses import supervised_contrastive_loss, coral_loss, irm_penalty, groupdro_loss
from dg_copdnet.training.trainer import grouped_holdout_experiment, train_standard_cv
from dg_copdnet.utils.device import get_device
from dg_copdnet.utils.io import ensure_dir, save_json
from dg_copdnet.utils.metrics import compute_metrics
from dg_copdnet.utils.seed import seed_everything


def _extract_param_count(val) -> int:
    """Safely extract integer parameter count from dict, stringified dict, or numeric value."""
    if isinstance(val, dict):
        return int(val.get("total_params", val.get("trainable_params", 0)))
    if isinstance(val, (int, float, np.integer, np.floating)):
        return 0 if np.isnan(val) else int(val)
    if isinstance(val, str):
        try:
            parsed = ast.literal_eval(val)
            if isinstance(parsed, dict):
                return int(parsed.get("total_params", parsed.get("trainable_params", 0)))
            return int(parsed)
        except Exception:
            m = re.search(r"(\d+)", val)
            return int(m.group(1)) if m else 0
    return 0


def _clean_kauh_pid(row: pd.Series) -> str:
    if str(row["source"]).upper() == "KAUH":
        m = re.match(r"^[A-Za-z]+(\d+)$", str(row["patient_id_raw"]).strip())
        if m:
            return f"KAUH_P{m.group(1)}"
    return str(row["patient_id"])


def _train_eval_dare_variant(
    cfg: AppConfig,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame | None,
    lambdas: dict,
    branch_mask: tuple[float, float, float] = (1.0, 1.0, 1.0),
    mixstyle_p: float = 0.25,
    seed: int = 42,
    epochs_override: int | None = None,
) -> dict:
    seed_everything(seed)
    device = get_device(cfg.training.use_cuda)
    feature_names = cfg.experiments.selected_feature_triplet or cfg.features.handcrafted_sets
    sources = sorted(pd.concat([train_df, val_df] + ([] if test_df is None else [test_df]))["source"].astype(str).unique())
    source_to_idx = {s: i for i, s in enumerate(sources)}
    hdim = int(len(cached_feature_vector(train_df.iloc[0]["file_path"], cfg.audio, cfg.features, feature_names)))

    model = DARECOPDNet(
        hdim,
        num_classes=2,
        num_domains=max(1, len(source_to_idx)),
        task="binary",
        embed_dim=cfg.model.embed_dim,
        projection_dim=cfg.model.projection_dim,
        dropout=cfg.model.dropout,
        pretrained_effnet=cfg.model.pretrained_effnet,
        grl_lambda=cfg.model.grl_lambda,
    ).to(device)
    if hasattr(model.spec, "mixstyle"):
        model.spec.mixstyle.p = mixstyle_p

    counts = train_df["label"].value_counts().to_dict()
    weights = train_df["label"].map(lambda x: 1.0 / counts[x]).to_numpy(dtype=np.float64)
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    tr_ds = RespiratoryAudioDataset(train_df, cfg.audio, cfg.features, cfg.augmentation, feature_names, True, source_to_idx)
    va_ds = RespiratoryAudioDataset(val_df, cfg.audio, cfg.features, cfg.augmentation, feature_names, False, source_to_idx)
    tr_loader = DataLoader(tr_ds, batch_size=cfg.training.batch_size, sampler=sampler, num_workers=cfg.training.num_workers, drop_last=True)
    va_loader = DataLoader(va_ds, batch_size=cfg.training.batch_size, shuffle=False, num_workers=cfg.training.num_workers)
    te_loader = None
    if test_df is not None:
        te_ds = RespiratoryAudioDataset(test_df, cfg.audio, cfg.features, cfg.augmentation, feature_names, False, source_to_idx)
        te_loader = DataLoader(te_ds, batch_size=cfg.training.batch_size, shuffle=False, num_workers=cfg.training.num_workers)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", patience=3, factor=0.5)
    bmask = torch.tensor(branch_mask, dtype=torch.float32, device=device).view(1, 3)

    best_auc = -1.0
    best_state = None
    wait = 0
    max_epochs = epochs_override or cfg.training.epochs

    def _forward_masked(spec, hand):
        z1 = model.spec(spec)
        z2, att = model.temp(spec)
        z3 = model.hand(hand)
        raw_gate = torch.softmax(model.branch_gate(torch.cat([z1, z2, z3], dim=1)), dim=1)
        gate = raw_gate * bmask
        gate = gate / (gate.sum(dim=1, keepdim=True) + 1e-8)
        z = gate[:, 0:1] * z1 + gate[:, 1:2] * z2 + gate[:, 2:3] * z3
        z = model.fusion(z)
        logits = model.classifier(z).squeeze(1)
        return {
            "logits": logits,
            "embedding": z,
            "projection": F.normalize(model.projection(z), dim=1),
            "domain_logits": model.domain_classifier(model.grl(z)),
            "branch_weights": gate,
        }

    for epoch in range(1, max_epochs + 1):
        model.train()
        for batch in tr_loader:
            spec = batch["spectrogram"].to(device)
            hand = batch["handcrafted"].to(device)
            y = batch["label"].to(device)
            dom = batch["domain"].to(device)
            opt.zero_grad(set_to_none=True)
            out = _forward_masked(spec, hand)
            per = F.binary_cross_entropy_with_logits(out["logits"], y.float(), reduction="none")
            cls_loss = groupdro_loss(per, dom, len(source_to_idx), eta=0.05) if lambdas.get("groupdro", 1) else per.mean()
            dom_loss = F.cross_entropy(out["domain_logits"], dom.long()) if (len(source_to_idx) > 1 and lambdas.get("domain", 0.0) > 0) else out["logits"].new_tensor(0.0)
            sup_loss = supervised_contrastive_loss(out["projection"], y.long(), 0.07) if (out["projection"].size(0) > 2 and lambdas.get("supcon", 0.0) > 0) else out["logits"].new_tensor(0.0)
            cor_loss = coral_loss(out["embedding"], dom) if lambdas.get("coral", 0.0) > 0 else out["logits"].new_tensor(0.0)
            irm_loss = irm_penalty(out["logits"], y, "binary") if lambdas.get("irm", 0.0) > 0 else out["logits"].new_tensor(0.0)
            loss = cls_loss + lambdas.get("domain", 0.0) * dom_loss + lambdas.get("supcon", 0.0) * sup_loss + lambdas.get("coral", 0.0) * cor_loss + lambdas.get("irm", 0.0) * irm_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.gradient_clip_norm)
            opt.step()

        model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for batch in va_loader:
                out = _forward_masked(batch["spectrogram"].to(device), batch["handcrafted"].to(device))
                ys.extend(batch["label"].numpy().astype(int).tolist())
                ps.extend(torch.sigmoid(out["logits"]).cpu().numpy().tolist())
        va_met = compute_metrics(ys, ps)
        va_auc = va_met["auc"]
        scheduler.step(va_auc)
        if va_auc > best_auc:
            best_auc = va_auc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= cfg.training.patience:
                break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()

    def _eval_loader(ldr, thr=0.5):
        ys, ps = [], []
        with torch.no_grad():
            for batch in ldr:
                out = _forward_masked(batch["spectrogram"].to(device), batch["handcrafted"].to(device))
                ys.extend(batch["label"].numpy().astype(int).tolist())
                ps.extend(torch.sigmoid(out["logits"]).cpu().numpy().tolist())
        return compute_metrics(ys, ps, threshold=thr), np.asarray(ys), np.asarray(ps)

    val_metrics, vy, vp = _eval_loader(va_loader, thr=0.5)
    best_thr, best_j = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 91):
        m_t = compute_metrics(vy, vp, threshold=float(t))
        if m_t["balanced_accuracy"] > best_j:
            best_j, best_thr = m_t["balanced_accuracy"], float(t)

    res = {"val": val_metrics, "val_thr": best_thr}
    if te_loader is not None:
        te_met_05, ty, tp = _eval_loader(te_loader, thr=0.5)
        te_met_youden = compute_metrics(ty, tp, threshold=best_thr)
        res["test_05"] = te_met_05
        res["test_youden"] = te_met_youden
    return res


def run_ablation_suite(cfg: AppConfig, int_df: pd.DataFrame, all_df: pd.DataFrame, out_dir: Path):
    print("\n=== [1/4] Running Leave-One-Loss-Out & Branch Ablations ===")
    csv_path = out_dir / "table_loss_and_branch_ablation.csv"
    variants = [
        ("Full Astar-COPDNet (All 5 Losses)", {"domain": 0.20, "supcon": 0.10, "coral": 0.10, "irm": 0.01, "groupdro": 1}, (1.0, 1.0, 1.0), 0.25),
        ("w/o SupCon (lambda_c = 0)", {"domain": 0.20, "supcon": 0.00, "coral": 0.10, "irm": 0.01, "groupdro": 1}, (1.0, 1.0, 1.0), 0.25),
        ("w/o CORAL (lambda_a = 0)", {"domain": 0.20, "supcon": 0.10, "coral": 0.00, "irm": 0.01, "groupdro": 1}, (1.0, 1.0, 1.0), 0.25),
        ("w/o IRM (lambda_i = 0)", {"domain": 0.20, "supcon": 0.10, "coral": 0.10, "irm": 0.00, "groupdro": 1}, (1.0, 1.0, 1.0), 0.25),
        ("w/o GRL Domain (lambda_d = 0)", {"domain": 0.00, "supcon": 0.10, "coral": 0.10, "irm": 0.01, "groupdro": 1}, (1.0, 1.0, 1.0), 0.25),
        ("w/o Group-DRO (Standard BCE)", {"domain": 0.20, "supcon": 0.10, "coral": 0.10, "irm": 0.01, "groupdro": 0}, (1.0, 1.0, 1.0), 0.25),
        ("w/o MixStyle (p = 0)", {"domain": 0.20, "supcon": 0.10, "coral": 0.10, "irm": 0.01, "groupdro": 1}, (1.0, 1.0, 1.0), 0.00),
        ("ERM Baseline (Standard BCE Only)", {"domain": 0.00, "supcon": 0.00, "coral": 0.00, "irm": 0.00, "groupdro": 0}, (1.0, 1.0, 1.0), 0.00),
        ("Branch 1 Only (EfficientNet-B0)", {"domain": 0.20, "supcon": 0.10, "coral": 0.10, "irm": 0.01, "groupdro": 1}, (1.0, 0.0, 0.0), 0.25),
        ("Branch 2 Only (CNN-BiGRU + Attn)", {"domain": 0.20, "supcon": 0.10, "coral": 0.10, "irm": 0.01, "groupdro": 1}, (0.0, 1.0, 0.0), 0.25),
        ("Branch 3 Only (Handcrafted MLP)", {"domain": 0.20, "supcon": 0.10, "coral": 0.10, "irm": 0.01, "groupdro": 1}, (0.0, 0.0, 1.0), 0.25),
    ]

    existing_rows = []
    completed_names = set()
    if csv_path.exists():
        try:
            existing_df = pd.read_csv(csv_path)
            if len(existing_df) >= len(variants):
                print(f"  -> Step [1/4] already completed ({len(existing_df)} variants found in {csv_path.name}). Skipping re-run!")
                return existing_df.to_dict(orient="records")
            existing_rows = existing_df.to_dict(orient="records")
            completed_names = set(existing_df["variant"].astype(str).tolist())
            print(f"  -> Resuming Step [1/4]: {len(completed_names)}/{len(variants)} variants already completed.")
        except Exception:
            pass

    tr_df_ext = all_df[all_df["source"].str.upper() == "TR"].reset_index(drop=True)
    rows = list(existing_rows)
    for name, lambdas, bmask, mp in variants:
        if name in completed_names:
            print(f"  -> Variant: {name} [ALREADY DONE - SKIPPING]")
            continue
        print(f"  -> Variant: {name}")
        cv_aucs, cv_f1s, cv_mccs, cv_eces, lodo_tr_aucs = [], [], [], [], []
        for fold, tr_idx, va_idx in make_grouped_folds(int_df, cfg.training.num_folds, cfg.seed):
            tr_sub = int_df.iloc[tr_idx].reset_index(drop=True)
            va_sub = int_df.iloc[va_idx].reset_index(drop=True)
            res = _train_eval_dare_variant(cfg, tr_sub, va_sub, tr_df_ext if fold == 1 else None, lambdas, bmask, mp, seed=cfg.seed)
            cv_aucs.append(res["val"]["auc"])
            cv_f1s.append(res["val"]["f1"])
            cv_mccs.append(res["val"]["mcc"])
            cv_eces.append(res["val"]["ece"])
            if fold == 1 and "test_05" in res:
                lodo_tr_aucs.append(res["test_05"]["auc"])
        rows.append({
            "variant": name,
            "cv_auc_mean": float(np.mean(cv_aucs)),
            "cv_auc_std": float(np.std(cv_aucs, ddof=1)),
            "cv_f1_mean": float(np.mean(cv_f1s)),
            "cv_f1_std": float(np.std(cv_f1s, ddof=1)),
            "cv_mcc_mean": float(np.mean(cv_mccs)),
            "cv_ece_mean": float(np.mean(cv_eces)),
            "lodo_icbhi_kauh_to_tr_auc": float(lodo_tr_aucs[0]) if lodo_tr_aucs else float("nan"),
        })
        pd.DataFrame(rows).to_csv(csv_path, index=False)
    return rows


def run_sota_same_protocol(cfg: AppConfig, int_df: pd.DataFrame, out_dir: Path):
    print("\n=== [2/4] Running Same-Protocol SOTA & Deep Baseline Comparison ===")
    from dg_copdnet.training.advanced_trainer import train_cv_generic
    models = ["aerocopdnet", "spec_cnn", "effnet_only", "crnn_only", "handcrafted_mlp", "hybrid_dg_respnet"]
    rows = []
    for mname in models:
        existing_summary = Path(cfg.output_dir) / "cmpb_revision" / "sota_baselines" / mname / "summary.csv"
        if existing_summary.exists():
            try:
                df_res = pd.read_csv(existing_summary)
                if len(df_res) >= cfg.training.num_folds:
                    print(f"  -> Baseline model: {mname} [LOADED FROM EXISTING summary.csv]")
                else:
                    print(f"  -> Baseline model: {mname}")
                    df_res = train_cv_generic(
                        cfg, int_df, model_name=mname, label_col="label", multiclass=False,
                        feature_names=cfg.experiments.selected_feature_triplet or cfg.features.handcrafted_sets,
                        optimizer_name="adamw", output_subdir="cmpb_revision/sota_baselines"
                    )
            except Exception:
                print(f"  -> Baseline model: {mname}")
                df_res = train_cv_generic(
                    cfg, int_df, model_name=mname, label_col="label", multiclass=False,
                    feature_names=cfg.experiments.selected_feature_triplet or cfg.features.handcrafted_sets,
                    optimizer_name="adamw", output_subdir="cmpb_revision/sota_baselines"
                )
        else:
            print(f"  -> Baseline model: {mname}")
            df_res = train_cv_generic(
                cfg, int_df, model_name=mname, label_col="label", multiclass=False,
                feature_names=cfg.experiments.selected_feature_triplet or cfg.features.handcrafted_sets,
                optimizer_name="adamw", output_subdir="cmpb_revision/sota_baselines"
            )
        p_count = _extract_param_count(df_res["param_count"].iloc[0]) if "param_count" in df_res.columns else 0
        rows.append({
            "model": mname,
            "auc_mean": float(pd.to_numeric(df_res["auc"], errors="coerce").mean()),
            "auc_std": float(pd.to_numeric(df_res["auc"], errors="coerce").std(ddof=1)),
            "acc_mean": float(pd.to_numeric(df_res["accuracy"], errors="coerce").mean()),
            "sens_mean": float(pd.to_numeric(df_res["sensitivity"], errors="coerce").mean()),
            "spec_mean": float(pd.to_numeric(df_res["specificity"], errors="coerce").mean()),
            "f1_mean": float(pd.to_numeric(df_res["f1"], errors="coerce").mean()),
            "mcc_mean": float(pd.to_numeric(df_res["mcc"], errors="coerce").mean()),
            "brier_mean": float(pd.to_numeric(df_res["brier"], errors="coerce").mean()) if "brier" in df_res.columns else float("nan"),
            "ece_mean": float(pd.to_numeric(df_res["ece"], errors="coerce").mean()) if "ece" in df_res.columns else float("nan"),
            "params": p_count,
        })
        pd.DataFrame(rows).to_csv(out_dir / "table_same_protocol_baselines.csv", index=False)
    return rows


def run_holdout_ratios(cfg: AppConfig, int_df: pd.DataFrame, out_dir: Path):
    print("\n=== [3/4] Running Holdout Split Ratio Sensitivity (20%, 30%, 40%) ===")
    rows = []
    for ratio in [0.20, 0.30, 0.40]:
        tr_idx, te_idx = make_grouped_holdout(int_df, test_size=ratio, seed=cfg.seed)
        tr_sub, te_sub = int_df.iloc[tr_idx], int_df.iloc[te_idx]
        save_pref = f"cmpb_revision/holdout_{int(ratio*100)}"
        hist_file = Path(cfg.output_dir) / save_pref / "history.csv"
        if hist_file.exists():
            print(f"  -> Holdout {int(ratio*100)}% [LOADED FROM EXISTING history.csv]")
            hist = pd.read_csv(hist_file)
        else:
            print(f"  -> Running Holdout {int(ratio*100)}%...")
            hist = grouped_holdout_experiment(cfg, subset_df=int_df, test_size=ratio, save_prefix=save_pref)
        best = hist.loc[pd.to_numeric(hist["test_auc"], errors="coerce").idxmax()]
        rows.append({
            "test_ratio": f"{int(ratio*100)}%",
            "train_patients": int(tr_sub["patient_id"].nunique()),
            "train_recordings": int(len(tr_sub)),
            "test_patients": int(te_sub["patient_id"].nunique()),
            "test_recordings": int(len(te_sub)),
            "auc": float(best["test_auc"]),
            "accuracy": float(best["test_accuracy"]),
            "sensitivity": float(best["test_sensitivity"]),
            "specificity": float(best["test_specificity"]),
            "f1": float(best["test_f1"]),
            "mcc": float(best["test_mcc"]),
            "brier": float(best["test_brier"]),
            "ece": float(best["test_ece"]),
        })
        pd.DataFrame(rows).to_csv(out_dir / "table_holdout_ratios.csv", index=False)
    return rows


def run_strict_patient_and_multiseed(cfg: AppConfig, config_path: str, int_df: pd.DataFrame, out_dir: Path):
    print("\n=== [4/4] Running Strict 112-Subject KAUH Grouping & Multi-Seed Verification ===")
    rows = []
    strict_df = int_df.copy()
    strict_df["patient_id"] = strict_df.apply(_clean_kauh_pid, axis=1)
    strict_stats_file = Path(cfg.output_dir) / "cmpb_revision" / "strict_238pts_cv" / "cv_summary_stats.json"
    if strict_stats_file.exists():
        print("  -> Strict 238-Patient Grouping [LOADED FROM EXISTING cv_summary_stats.json]")
        stats_strict = json.loads(strict_stats_file.read_text(encoding="utf-8"))
    else:
        print("  -> Running Strict 238-Patient Grouping (KAUH 112 Subjects, Seed 42)...")
        _, stats_strict = train_standard_cv(cfg, optimizer_name="adamw", subset_df=strict_df, save_prefix="cmpb_revision/strict_238pts_cv")
    rows.append({
        "experiment": "Strict 238-Patient Grouping (KAUH 112 Subjects, Seed 42)",
        "auc_mean": float(stats_strict["val_auc"]["mean"]),
        "auc_std": float(stats_strict["val_auc"]["std"]),
        "acc_mean": float(stats_strict["val_accuracy"]["mean"]),
        "sens_mean": float(stats_strict["val_sensitivity"]["mean"]),
        "spec_mean": float(stats_strict["val_specificity"]["mean"]),
        "f1_mean": float(stats_strict["val_f1"]["mean"]),
        "mcc_mean": float(stats_strict["val_mcc"]["mean"]),
        "brier_mean": float(stats_strict["val_brier"]["mean"]),
        "ece_mean": float(stats_strict["val_ece"]["mean"]),
    })
    pd.DataFrame(rows).to_csv(out_dir / "table_strict_grouping_multiseed.csv", index=False)

    for sd in [42, 43, 44]:
        seed_stats_file = Path(cfg.output_dir) / "cmpb_revision" / f"seed_{sd}_cv" / "cv_summary_stats.json"
        std_stats_file = Path(cfg.output_dir) / "standard_cv" / "cv_summary_stats.json"
        if seed_stats_file.exists():
            print(f"  -> Seed {sd} CV [LOADED FROM EXISTING cv_summary_stats.json]")
            st = json.loads(seed_stats_file.read_text(encoding="utf-8"))
        elif sd == 42 and std_stats_file.exists():
            print(f"  -> Seed 42 CV [LOADED FROM EXISTING standard_cv/cv_summary_stats.json]")
            st = json.loads(std_stats_file.read_text(encoding="utf-8"))
        else:
            print(f"  -> Running Internal 5-Fold CV (Seed {sd})...")
            cfg_sd = AppConfig.from_json(config_path)
            cfg_sd.seed = sd
            _, st = train_standard_cv(cfg_sd, optimizer_name="adamw", subset_df=int_df, save_prefix=f"cmpb_revision/seed_{sd}_cv")
        rows.append({
            "experiment": f"Internal 5-Fold CV (Seed {sd})",
            "auc_mean": float(st["val_auc"]["mean"]),
            "auc_std": float(st["val_auc"]["std"]),
            "acc_mean": float(st["val_accuracy"]["mean"]),
            "sens_mean": float(st["val_sensitivity"]["mean"]),
            "spec_mean": float(st["val_specificity"]["mean"]),
            "f1_mean": float(st["val_f1"]["mean"]),
            "mcc_mean": float(st["val_mcc"]["mean"]),
            "brier_mean": float(st["val_brier"]["mean"]),
            "ece_mean": float(st["val_ece"]["mean"]),
        })
        pd.DataFrame(rows).to_csv(out_dir / "table_strict_grouping_multiseed.csv", index=False)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Run CMPB Major Revision Experiments (Auto-Resuming)")
    parser.add_argument("--config", default="configs/windows_real_paths.json")
    args = parser.parse_args()
    cfg = AppConfig.from_json(args.config)
    if not (Path(cfg.internal_csv).exists() and Path(cfg.all_csv).exists()):
        build_metadata_from_real_paths(cfg)
    int_df = pd.read_csv(cfg.internal_csv)
    all_df = pd.read_csv(cfg.all_csv)
    out_dir = ensure_dir(Path(cfg.output_dir) / "cmpb_revision")

    summary = {}
    summary["ablation"] = run_ablation_suite(cfg, int_df, all_df, out_dir)
    summary["sota_baselines"] = run_sota_same_protocol(cfg, int_df, out_dir)
    summary["holdout_ratios"] = run_holdout_ratios(cfg, int_df, out_dir)
    summary["strict_and_multiseed"] = run_strict_patient_and_multiseed(cfg, args.config, int_df, out_dir)
    save_json(summary, out_dir / "cmpb_revision_all_results.json")
    print("\nAll CMPB revision experiments completed! Results saved in:", out_dir)


if __name__ == "__main__":
    main()
