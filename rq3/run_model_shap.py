#!/usr/bin/env python3
"""
RQ3 — SHAP explainability on tuned XGBoost models from RQ1.

Requires ``rq1/run_tuned_evaluation.py`` outputs (saved models + ``experiment_manifest.json``).
Does not re-run tuning, calibration, or held-out metric tables.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.utils.multiclass import type_of_target

REPLICATION_ROOT = Path(__file__).resolve().parents[1]
_LIB = REPLICATION_ROOT / "lib"
if str(_LIB) not in sys.path:
    sys.path.insert(0, str(_LIB))

from replication.data import load_task_level_data, train_test_indices_from_targets
from replication.paths import default_tuned_models_dir, results_dir_for_rq
from replication.shap_analysis import _default_shap_parallel_jobs, compute_shap_and_interactions
from replication.tuned_evaluation import (
    PRIMARY_MODEL_SLUG,
    assert_replication_dataset,
    load_repo_id_per_task,
    outcome_model_prefix,
)


def _split_indices(
    n_rows: int,
    task_outcomes: pd.DataFrame,
    *,
    test_size: float,
    random_state: int,
    group_by_repo: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    idx = np.arange(n_rows)
    y_any = task_outcomes["any_success"].values if "any_success" in task_outcomes.columns else None

    if group_by_repo:
        if y_any is None:
            raise ValueError("--group-by-repo requires any_success in outcomes")
        repo_grp = np.asarray(load_repo_id_per_task(task_outcomes.index).astype(str))
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        return next(gss.split(idx, y_any, groups=repo_grp))

    if y_any is not None and "split" in task_outcomes.columns:
        return train_test_indices_from_targets(task_outcomes, np.asarray(y_any))

    if y_any is not None:
        y_strat = np.asarray(y_any)
        if y_strat.dtype.kind == "f":
            y_strat = (y_strat >= 0.5).astype(int)
        else:
            y_strat = y_strat.astype(int)
        strat = y_strat if len(np.unique(y_strat)) >= 2 else None
        return train_test_split(
            idx,
            test_size=test_size,
            random_state=random_state,
            stratify=strat,
        )

    return train_test_split(idx, test_size=test_size, random_state=random_state)


def _load_tuned_model(tuned_root: Path, target: str):
    pfx = outcome_model_prefix(target)
    pkl_path = tuned_root / pfx / f"{pfx}_final_xgb_model.pkl"
    if not pkl_path.is_file():
        raise FileNotFoundError(f"Missing tuned model: {pkl_path}")
    pack = joblib.load(pkl_path)
    model = pack["model"]
    feature_names = list(pack["feature_names"])
    return model, feature_names, pfx


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tuned-dir",
        default=None,
        help="RQ1 tuned_models directory (default: results/analysis/rq1_predictive_accuracy/tuned_models).",
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="SHAP output root (default: results/analysis/rq3_feature_importance/model_shap).",
    )
    parser.add_argument(
        "--targets",
        default=None,
        help="Comma-separated targets (default: from tuned experiment_manifest.json).",
    )
    parser.add_argument("--shap-sample", type=int, default=None)
    parser.add_argument("--shap-parallel-n-jobs", type=int, default=None)
    parser.add_argument("--shap-top-interactions", type=int, default=50)
    parser.add_argument("--no-interactions", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    tuned_root = Path(args.tuned_dir).resolve() if args.tuned_dir else default_tuned_models_dir()
    manifest_path = tuned_root / "experiment_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing {manifest_path}. Run: python rq1/run_tuned_evaluation.py --verbose"
        )
    manifest = json.loads(manifest_path.read_text())
    cli = manifest.get("cli_args") or {}

    if cli.get("aggregate_by_repo"):
        raise SystemExit("SHAP script supports task-level runs only (not --aggregate-by-repo).")
    if not cli.get("use_task_loader", True):
        raise SystemExit("SHAP script requires use_task_loader=True tuned runs.")

    test_size = float(cli.get("test_size", 0.2))
    random_state = int(cli.get("random_state", 42))
    group_by_repo = bool(cli.get("group_by_repo"))

    if args.targets:
        targets_list = [t.strip() for t in args.targets.split(",") if t.strip()]
    else:
        targets_list = manifest.get("targets_run") or ["pass_rate", "any_success", "maj_success"]

    out_root = (
        Path(args.outdir).resolve()
        if args.outdir
        else (results_dir_for_rq("rq3") / "model_shap")
    )
    out_root.mkdir(parents=True, exist_ok=True)

    assert_replication_dataset()
    X, task_outcomes = load_task_level_data()
    idx_train, idx_test = _split_indices(
        len(X),
        task_outcomes,
        test_size=test_size,
        random_state=random_state,
        group_by_repo=group_by_repo,
    )

    shap_parallel_jobs = (
        max(1, int(args.shap_parallel_n_jobs))
        if args.shap_parallel_n_jobs is not None
        else _default_shap_parallel_jobs(None)
    )

    if args.verbose:
        print(f"Tuned models: {tuned_root}")
        print(f"SHAP output: {out_root}")
        print(f"Train/test rows: {len(idx_train)} / {len(idx_test)}")
        print(f"SHAP workers: {shap_parallel_jobs}")

    overall_summary: Dict[str, Any] = {
        "tuned_models_dir": str(tuned_root.resolve()),
        "shap_row_parallel_jobs": shap_parallel_jobs,
    }

    for target in targets_list:
        if target not in task_outcomes.columns:
            raise ValueError(f"target {target!r} not in outcomes")
        print(f"\n=== SHAP: {target} ===")
        model, feature_names, pfx = _load_tuned_model(tuned_root, target)
        missing = [c for c in feature_names if c not in X.columns]
        if missing:
            raise ValueError(f"Features missing from dataset for {target}: {missing[:5]}...")

        X_test = X.iloc[idx_test][feature_names].copy()
        y_train = task_outcomes[target].values[idx_train]
        try:
            ttype = type_of_target(y_train)
        except Exception:
            ttype = "continuous"
        is_regression = ttype not in ("binary", "multiclass")

        target_out = out_root / pfx
        target_out.mkdir(parents=True, exist_ok=True)

        shap_results = compute_shap_and_interactions(
            model,
            X_test,
            str(target_out),
            artifact_prefix=pfx,
            sample_n=args.shap_sample,
            topk_interactions=args.shap_top_interactions,
            no_interactions=args.no_interactions,
            verbose=args.verbose,
            is_regression=is_regression,
            row_parallel_jobs=shap_parallel_jobs,
        )
        (target_out / f"{pfx}_shap_explainability_summary.json").write_text(
            json.dumps(shap_results, indent=2, default=str)
        )

        imp_path = shap_results.get("feature_importance_path")
        if imp_path and Path(imp_path).exists():
            df_imp = pd.read_csv(imp_path)
            topk = min(args.shap_top_interactions, len(df_imp))
            df_imp.head(topk)[["feature"]].to_csv(
                target_out / f"{pfx}_top_features_for_followups.csv", index=False
            )

        overall_summary[target] = {"shap": shap_results}

    pfx_any = outcome_model_prefix("any_success")
    pfx_rate = outcome_model_prefix("pass_rate")
    path_any = out_root / pfx_any / f"{pfx_any}_shap_feature_importance.csv"
    path_rate = out_root / pfx_rate / f"{pfx_rate}_shap_feature_importance.csv"
    cross_path = out_root / "cross_outcome_any_success_xgb_vs_pass_rate_xgb_shap_importance_consistency.csv"
    try:
        if path_any.is_file() and path_rate.is_file():
            dfa = pd.read_csv(path_any)
            dfr = pd.read_csv(path_rate)
            dfa = dfa.rename(columns={"mean_abs_shap": "mean_abs_shap_any_success"})
            dfr = dfr.rename(columns={"mean_abs_shap": "mean_abs_shap_pass_rate"})
            merged = pd.merge(dfa, dfr, on="feature", how="outer")
            merged["rank_any_success"] = merged["mean_abs_shap_any_success"].rank(
                ascending=False, method="average"
            )
            merged["rank_pass_rate"] = merged["mean_abs_shap_pass_rate"].rank(
                ascending=False, method="average"
            )
            merged["rank_delta"] = merged["rank_any_success"] - merged["rank_pass_rate"]
            merged = merged.sort_values("rank_any_success", na_position="last")
            merged.to_csv(cross_path, index=False)
    except Exception:
        pass

    (out_root / "run_multitarget_xgb_summary.json").write_text(
        json.dumps(overall_summary, indent=2, default=str)
    )

    shap_manifest = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "script_path": str((REPLICATION_ROOT / "rq3" / "run_model_shap.py").resolve()),
        "tuned_models_dir": str(tuned_root.resolve()),
        "output_root": str(out_root.resolve()),
        "targets_run": targets_list,
        "cli_args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "split_from_tuned_manifest": {
            "test_size": test_size,
            "random_state": random_state,
            "group_by_repo": group_by_repo,
        },
    }
    (out_root / "experiment_manifest.json").write_text(json.dumps(shap_manifest, indent=2))

    print(f"\nDone. SHAP outputs under {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
