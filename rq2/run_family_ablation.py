#!/usr/bin/env python3
"""
Family ablation (patch / repo / prompt) aligned with a finished ``task_level_model_shap`` run.

Reads ``cf-task-shap-7299333``-style outputs under ``--shap-outdir``:

- ``experiment_manifest.json`` — ``test_size``, ``random_state`` (same stratified split as SHAP).
- Shared VIF column set — uses the **same** 54 post-VIF features as the SHAP job (via each
  ``<target>_xgb/<target>_final_xgb_model.pkl`` ``feature_names`` list; identical across targets).
- **Tuned XGBoost hyperparameters** per target from
  ``<target>_xgb/<target>_tuned_full_train_report.json`` key ``XGBoost.best_params``; if missing,
  falls back to the params printed in that job's Slurm log:

  ``{'subsample': 0.75, 'reg_lambda': 1.0, 'n_estimators': 400, 'min_child_weight': 2,``
  ``'max_depth': 8, 'learning_rate': 0.03, 'colsample_bytree': 0.75}``

For each target in ``pass_rate``, ``any_success``, ``maj_success`` and each family combo
(single families + unions + full), fits a **fresh** XGBoost model with that target's tuned
params on **only** the columns in that combo (intersection with the VIF feature list), evaluates
on the **same** held-out test split, and writes ``family_ablation_metrics.csv`` plus a small
``family_ablation_run.json`` manifest.

Does **not** modify any other project files.

Example::

    python scripts/task_level_family_ablation_from_shap_run.py \\
      --shap-outdir /scratch \\
      --verbose
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.multiclass import type_of_target

REPLICATION_ROOT = Path(__file__).resolve().parents[1]
_LIB = REPLICATION_ROOT / "lib"
if str(_LIB) not in sys.path:
    sys.path.insert(0, str(_LIB))

from replication.data import load_task_level_data, train_test_indices_from_targets
from replication.paths import default_dataset_path, default_tuned_models_dir, results_dir_for_rq

PRIMARY_SLUG = "xgb"

# From Slurm log cf-task-shap-7299333.out (XGBoost line; same dict for all three targets in that run).
LOG_FALLBACK_XGB_PARAMS: Dict[str, Any] = {
    "subsample": 0.75,
    "reg_lambda": 1.0,
    "n_estimators": 400,
    "min_child_weight": 2,
    "max_depth": 8,
    "learning_rate": 0.03,
    "colsample_bytree": 0.75,
}

def _load_comparison_module():
    path = REPLICATION_ROOT / "rq1" / "run_model_comparison.py"
    spec = importlib.util.spec_from_file_location("_task_level_model_comparison", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module spec from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _assert_dataset() -> None:
    path = default_dataset_path()
    if not path.is_file():
        raise FileNotFoundError(f"Task-level dataset missing: {path}")


def _split_train_test_idx(
    n_rows: int,
    y_strat: np.ndarray,
    *,
    test_size: float,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray]:
    idx = np.arange(n_rows)
    strat = y_strat if len(np.unique(y_strat)) >= 2 else None
    return train_test_split(idx, test_size=test_size, random_state=random_state, stratify=strat)


def _family_groups(columns: List[str]) -> Dict[str, List[str]]:
    cols = list(columns)
    return {
        "patch": [c for c in cols if c.startswith("patch_")],
        "repo": [c for c in cols if c.startswith("repo_")],
        "prompt": [c for c in cols if c.startswith("prompt_")],
    }


def _family_combos(groups: Dict[str, List[str]]) -> Dict[str, List[str]]:
    p, r, pr = groups["patch"], groups["repo"], groups["prompt"]
    return {
        "patch": p,
        "repo": r,
        "prompt": pr,
        "patch+repo": p + r,
        "patch+prompt": p + pr,
        "repo+prompt": r + pr,
        "patch+repo+prompt": p + r + pr,
    }


def _load_xgb_best_params(shap_outdir: Path, target: str) -> Dict[str, Any]:
    report_path = shap_outdir / f"{target}_{PRIMARY_SLUG}" / f"{target}_tuned_full_train_report.json"
    if not report_path.is_file():
        return dict(LOG_FALLBACK_XGB_PARAMS)
    data = json.loads(report_path.read_text())
    xgb = data.get("XGBoost") or {}
    bp = xgb.get("best_params")
    if isinstance(bp, dict) and bp:
        return dict(bp)
    return dict(LOG_FALLBACK_XGB_PARAMS)


def _load_feature_order_from_pkl(shap_outdir: Path, target: str) -> List[str]:
    pkl = shap_outdir / f"{target}_{PRIMARY_SLUG}" / f"{target}_{PRIMARY_SLUG}_final_xgb_model.pkl"
    if not pkl.is_file():
        raise FileNotFoundError(f"Expected saved model: {pkl}")
    pack = joblib.load(pkl)
    fn = pack.get("feature_names")
    if not fn:
        raise ValueError(f"{pkl} has no feature_names")
    return list(fn)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shap-outdir",
        type=str,
        default=None,
        help="Directory from the finished task_level_model_shap run (manifest + per-target xgb/).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Where to write CSV/JSON (default: <shap-outdir>/family_ablation_xgb_tuned/).",
    )
    parser.add_argument(
        "--targets",
        type=str,
        default="pass_rate,any_success,maj_success",
        help="Comma-separated targets (default matches cf-task-shap-7299333).",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    shap_outdir = (
        Path(args.shap_outdir).resolve()
        if args.shap_outdir
        else default_tuned_models_dir()
    )
    man_path = shap_outdir / "experiment_manifest.json"
    if not man_path.is_file():
        raise FileNotFoundError(f"Missing {man_path}")

    manifest = json.loads(man_path.read_text())
    cli = manifest.get("cli_args") or {}
    if cli.get("aggregate_by_repo") or cli.get("group_by_repo"):
        raise SystemExit("This script only supports the plain stratified task-level split (no repo modes).")
    if not cli.get("use_task_loader", True):
        raise SystemExit("Only use_task_loader=True runs are supported.")

    test_size = float(cli.get("test_size", 0.2))
    random_state = int(cli.get("random_state", 42))

    out_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else (results_dir_for_rq("rq2") / "family_ablation")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = [t.strip() for t in args.targets.split(",") if t.strip()]

    _assert_dataset()
    X, task_outcomes = load_task_level_data()

    y_any = task_outcomes["any_success"].values
    if "split" in task_outcomes.columns:
        idx_train, idx_test = train_test_indices_from_targets(task_outcomes, y_any)
    else:
        y_strat = np.asarray(y_any)
        if y_strat.dtype.kind == "f":
            y_strat = (y_strat >= 0.5).astype(int)
        else:
            y_strat = y_strat.astype(int)
        idx_train, idx_test = _split_train_test_idx(
            len(X), y_strat, test_size=test_size, random_state=random_state
        )
    X_train_full = X.iloc[idx_train]
    X_test_full = X.iloc[idx_test]

    feat_ref = _load_feature_order_from_pkl(shap_outdir, targets[0])
    for t in targets[1:]:
        other = _load_feature_order_from_pkl(shap_outdir, t)
        if other != feat_ref:
            raise ValueError(
                f"VIF feature list differs between {targets[0]!r} and {t!r}; "
                "use one reference or extend this script."
            )

    X_train = X_train_full[feat_ref].copy()
    X_test = X_test_full[feat_ref].copy()

    groups = _family_groups(feat_ref)
    combos = _family_combos(groups)
    if args.verbose:
        for k, v in groups.items():
            print(f"family {k}: {len(v)} features in VIF set")

    import xgboost as xgb  # noqa: WPS433

    rows_out: List[Dict[str, Any]] = []
    params_record: Dict[str, Any] = {}

    for target in targets:
        if target not in task_outcomes.columns:
            raise ValueError(f"Unknown target column: {target}")
        y_all = task_outcomes[target].values
        y_train = y_all[idx_train]
        y_test = y_all[idx_test]

        try:
            ttype = type_of_target(y_train)
        except Exception:
            ttype = "continuous"
        is_regression = ttype not in ("binary", "multiclass")

        best = _load_xgb_best_params(shap_outdir, target)
        params_record[target] = {"source": "tuned_full_train_report.json or log fallback", "best_params": best}

        for combo_name, cols in combos.items():
            use_cols = [c for c in cols if c in X_train.columns]
            if not use_cols:
                rows_out.append(
                    {
                        "target": target,
                        "components": combo_name,
                        "n_features": 0,
                        "skipped": True,
                        "reason": "no columns for this family in VIF set",
                    }
                )
                continue
            Xtr = X_train[use_cols]
            Xte = X_test[use_cols]

            if is_regression:
                model = xgb.XGBRegressor(
                    objective="reg:squarederror",
                    random_state=42,
                    n_jobs=min(8, (os.cpu_count() or 4)),
                    tree_method="hist",
                    verbosity=0,
                    **best,
                )
                model.fit(Xtr, y_train)
                pred = np.clip(model.predict(Xte), 0.0, 1.0)
                rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
                mae = float(mean_absolute_error(y_test, pred))
                r2 = float(r2_score(y_test, pred))
                rows_out.append(
                    {
                        "target": target,
                        "components": combo_name,
                        "n_features": len(use_cols),
                        "skipped": False,
                        "rmse": rmse,
                        "mae": mae,
                        "r2": r2,
                    }
                )
                if args.verbose:
                    print(f"  {target:12s} {combo_name:20s} n={len(use_cols):2d}  RMSE={rmse:.4f}  R2={r2:.4f}")
            else:
                model = xgb.XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="auc",
                    random_state=42,
                    n_jobs=min(8, (os.cpu_count() or 4)),
                    tree_method="hist",
                    verbosity=0,
                    **best,
                )
                model.fit(Xtr, y_train.astype(int))
                prob = model.predict_proba(Xte)[:, 1]
                auc = float(roc_auc_score(y_test.astype(int), prob))
                pr_auc = float(average_precision_score(y_test.astype(int), prob))
                brier = float(brier_score_loss(y_test.astype(int), prob))
                rows_out.append(
                    {
                        "target": target,
                        "components": combo_name,
                        "n_features": len(use_cols),
                        "skipped": False,
                        "auc": auc,
                        "roc_auc": auc,
                        "pr_auc": pr_auc,
                        "brier": brier,
                    }
                )
                if args.verbose:
                    print(f"  {target:12s} {combo_name:20s} n={len(use_cols):2d}  AUC={auc:.4f}  PR-AUC={pr_auc:.4f}")

    df = pd.DataFrame(rows_out)
    csv_path = out_dir / "family_ablation_metrics.csv"
    df.to_csv(csv_path, index=False)

    meta = {
        "shap_outdir": str(shap_outdir),
        "log_reference": "cf-task-shap-7299333.out tuned XGBoost dict (fallback if JSON missing)",
        "manifest_test_size": test_size,
        "manifest_random_state": random_state,
        "n_vif_features": len(feat_ref),
        "targets": targets,
        "xgb_best_params_by_target": params_record,
        "combos": list(combos.keys()),
    }
    json_path = out_dir / "family_ablation_run.json"
    json_path.write_text(json.dumps(meta, indent=2, default=str))

    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
