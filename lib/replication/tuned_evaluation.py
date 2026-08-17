#!/usr/bin/env python3
"""
RQ1 tuned evaluation: VIF on train, RandomizedSearchCV, held-out metrics, ROC, calibration.

Hyperparameter search and test-set evaluation for task-level outcomes. SHAP explainability
lives in ``replication.shap_analysis`` and ``rq3/run_model_shap.py``.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    KFold,
    RandomizedSearchCV,
    StratifiedGroupKFold,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.utils.multiclass import type_of_target

warnings.filterwarnings("ignore")

REPLICATION_ROOT = Path(__file__).resolve().parents[2]
_LIB = REPLICATION_ROOT / "lib"
if str(_LIB) not in sys.path:
    sys.path.insert(0, str(_LIB))

from replication.data import load_task_level_data, train_test_indices_from_targets
from replication.paths import default_dataset_path, default_tuned_models_dir, load_paper_xgb_hyperparams, results_dir_for_rq
from replication.shap_analysis import _default_shap_parallel_jobs

# This script fits one family of models; all outcome-specific artifacts use this slug in names.
PRIMARY_MODEL_SLUG = "xgb"


def outcome_model_prefix(outcome: str, model_slug: str = PRIMARY_MODEL_SLUG) -> str:
    """Stable prefix for files tied to one outcome + model, e.g. ``any_success_xgb``."""
    return f"{outcome}_{model_slug}"


def assert_replication_dataset() -> None:
    """Fail fast if the published task-level dataset is missing."""
    path = default_dataset_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"Expected task-level dataset at:\n  {path}\n"
            "See data/README.md in this replication package."
        )


def _load_comparison_module():
    """Load task_level_model_comparison without running its CLI main."""
    path = REPLICATION_ROOT / "rq1" / "run_model_comparison.py"
    spec = importlib.util.spec_from_file_location("_task_level_model_comparison", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module spec from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_table(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def select_features_from_effects(effects: pd.DataFrame, top_n: Optional[int]) -> List[str]:
    if "feature" not in effects.columns or "effect_size" not in effects.columns:
        raise ValueError("effects file must contain 'feature' and 'effect_size' columns")
    df = effects.copy()
    df["abs_effect"] = df["effect_size"].abs()
    df = df.sort_values("abs_effect", ascending=False)
    if top_n is not None:
        return df.head(top_n)["feature"].tolist()
    return df["feature"].tolist()


def prepare_Xy(df: pd.DataFrame, target: str, features: Optional[List[str]] = None) -> Tuple[pd.DataFrame, np.ndarray]:
    if target not in df.columns:
        raise ValueError(f"target column '{target}' not in features table")
    if features is None:
        features = [c for c in df.columns if c != target]
    X = df[features].select_dtypes(include=[np.number]).copy()
    if X.shape[1] == 0:
        raise ValueError("no numeric features available for modeling")
    y = df[target].values
    return X, y


def build_xgb_model(is_regression: bool):
    """Instantiate XGBoost with hyperparameters **locked** to the benchmark in ``task_level_model_comparison``.

    Source of truth: ``scripts/task_level_model_comparison.py`` — ``XGBClassifier`` /
    ``XGBRegressor`` blocks in ``run_classification`` / ``run_regression`` (same
    ``n_estimators``, ``max_depth``, ``eta``, row/column subsampling, ``min_child_weight``,
    ``tree_method="hist"``). These are **not** tuned inside this SHAP script: fixing them
    keeps this explainability run **experimentally aligned** with the main task-level
    comparison so that SHAP rankings and CV scores are directly comparable across scripts.
    """
    import xgboost as xgb  # noqa: WPS433

    if is_regression:
        return xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=300,
            max_depth=4,
            eta=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            random_state=42,
            n_jobs=4,
            tree_method="hist",
            verbosity=0,
        )
    return xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        n_estimators=300,
        max_depth=4,
        eta=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        random_state=42,
        n_jobs=4,
        tree_method="hist",
        verbosity=0,
    )


def xgb_locked_hyperparameter_dict(is_regression: bool) -> Dict[str, Any]:
    """Hyperparameters frozen to match ``task_level_model_comparison`` (for experiment_manifest)."""
    if is_regression:
        return {
            "objective": "reg:squarederror",
            "n_estimators": 300,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 3,
            "random_state": 42,
            "n_jobs": 4,
            "tree_method": "hist",
            "verbosity": 0,
        }
    return {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "random_state": 42,
        "n_jobs": 4,
        "tree_method": "hist",
        "verbosity": 0,
    }


def save_xgb_calibration_outputs(
    y_true: np.ndarray,
    prob: np.ndarray,
    table_path: Path,
    plot_path: Path,
    *,
    title: str,
) -> None:
    """
    Reliability diagram + bin table. Matches ``task_level_model_comparison.calibration_table_and_plot``:
    ``sklearn.calibration.calibration_curve`` with ``n_bins=10``, ``strategy='quantile'``.
    """
    try:
        from sklearn.calibration import calibration_curve
        import matplotlib.pyplot as plt

        prob = np.asarray(prob, dtype=float)
        y_true = np.asarray(y_true)
        if len(prob) == 0:
            return
        frac_pos, mean_pred = calibration_curve(y_true, prob, n_bins=10, strategy="quantile")
        pd.DataFrame({"mean_pred": mean_pred, "frac_pos": frac_pos}).to_csv(table_path, index=False)
        plt.figure(figsize=(6, 6))
        plt.plot(mean_pred, frac_pos, marker="o")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("Mean predicted probability")
        plt.ylabel("Observed fraction positive")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=200, bbox_inches="tight")
        plt.close()
    except Exception:
        try:
            import matplotlib.pyplot as plt

            plt.close()
        except Exception:
            pass


def save_roc_curve_outputs(
    y_true: np.ndarray,
    prob: np.ndarray,
    curve_csv: Path,
    plot_png: Path,
    *,
    title: str,
) -> None:
    """
    Persist ROC curve (FPR/TPR/thresholds) and a PNG. ``holdout_roc_auc`` in the CSV matches
    ``sklearn.metrics.roc_auc_score`` for the same ``y_true`` and ``prob``.
    """
    try:
        import matplotlib.pyplot as plt

        y_true = np.asarray(y_true).astype(int)
        prob = np.asarray(prob, dtype=float)
        if len(prob) == 0 or len(y_true) != len(prob):
            return
        if len(np.unique(y_true)) < 2:
            return
        auc_v = float(roc_auc_score(y_true, prob))
        fpr, tpr, thr = roc_curve(y_true, prob)
        thr_col = np.full(len(fpr), np.nan, dtype=float)
        if len(thr) > 0:
            thr_col[1 : 1 + len(thr)] = thr
        pd.DataFrame(
            {"fpr": fpr, "tpr": tpr, "threshold": thr_col, "holdout_roc_auc": auc_v}
        ).to_csv(curve_csv, index=False)

        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, lw=2, color="#4C72B0", label=f"ROC (AUC = {auc_v:.4f})")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray", lw=1)
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        plt.title(title)
        plt.legend(loc="lower right")
        plt.xlim(0.0, 1.0)
        plt.ylim(0.0, 1.0)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plot_png, dpi=200, bbox_inches="tight")
        plt.close()
    except Exception:
        try:
            import matplotlib.pyplot as plt

            plt.close()
        except Exception:
            pass


def _rename_shared_vif_outputs(vif_dir: Path) -> None:
    """Rename VIF CSVs to explicit shared-train names (not tied to a single outcome)."""
    mapping = {
        "vif_all_features.csv": "shared_train_split_before_xgb_vif_all_features.csv",
        "vif_selected_features.csv": "shared_train_split_before_xgb_vif_selected_features.csv",
        "vif_dropped_features.csv": "shared_train_split_before_xgb_vif_dropped_features_order.csv",
    }
    for old_name, new_name in mapping.items():
        op = vif_dir / old_name
        np_ = vif_dir / new_name
        if op.exists():
            if np_.exists():
                np_.unlink()
            op.rename(np_)


def _package_versions() -> Dict[str, str]:
    out: Dict[str, str] = {}
    try:
        import importlib.metadata as imd

        for pkg in ("numpy", "pandas", "scikit-learn", "xgboost", "shap", "statsmodels"):
            try:
                out[pkg] = imd.version(pkg)
            except Exception:
                out[pkg] = "unknown"
    except Exception:
        pass
    return out


def clf_metrics(y_true, prob, threshold: float = 0.5) -> Dict[str, float]:
    """Classification metrics from predicted probabilities.

    AUC, PR-AUC, and Brier are threshold-free. F1 and MCC use ``threshold`` for hard
    labels and are therefore threshold-dependent (default 0.5; test evaluation may pass
    Youden's J optimal threshold from the train ROC).
    """
    pred = (prob >= threshold).astype(int)
    auc_v = float(roc_auc_score(y_true, prob))
    return {
        "auc": auc_v,
        "roc_auc": auc_v,
        "pr_auc": float(average_precision_score(y_true, prob)),
        "brier": float(brier_score_loss(y_true, prob)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, pred)),
    }


def optimal_threshold_youden_j(y_true: np.ndarray, prob_positive: np.ndarray) -> float:
    """Youden's J = TPR - FPR; return the score threshold that maximizes it on train."""
    y_true = np.asarray(y_true).astype(int)
    prob_positive = np.asarray(prob_positive, dtype=float)
    if len(np.unique(y_true)) < 2:
        return 0.5
    fpr, tpr, thr = roc_curve(y_true, prob_positive)
    if len(thr) == 0:
        return 0.5
    # sklearn: (fpr[k+1], tpr[k+1]) corresponds to threshold thr[k]
    j_values = tpr[1 : 1 + len(thr)] - fpr[1 : 1 + len(thr)]
    return float(thr[int(np.argmax(j_values))])


def load_repo_id_per_task(task_index: pd.Index) -> pd.Series:
    """One repo slug per task from ``base_repo`` in the published dataset."""
    _, targets = load_task_level_data()
    if "base_repo" not in targets.columns:
        raise ValueError("Dataset has no base_repo column; cannot use --group-by-repo.")
    out = targets["base_repo"].astype(str).reindex(task_index)
    if out.isna().any():
        n_miss = int(out.isna().sum())
        raise ValueError(f"base_repo missing for {n_miss} task_ids.")
    return out


def aggregate_task_level_to_repo(
    X: pd.DataFrame,
    outcomes: pd.DataFrame,
    repo_id: pd.Series,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Collapse task-level rows to one row per repo.

    Features are mean-pooled across tasks in the repo. Targets: ``pass_rate``,
    ``any_success``, ``maj_success`` are means across tasks; ``n_runs`` is the sum of
    per-task run counts (when present).
    """
    if not X.index.equals(outcomes.index):
        raise ValueError("X and outcomes must share the same index (task_id).")
    rid = repo_id.reindex(X.index)
    if rid.isna().any():
        raise ValueError(f"repo_id missing for {int(rid.isna().sum())} tasks before repo aggregation.")
    df = outcomes.join(X, how="inner")
    df = df.copy()
    df["_repo_id"] = np.asarray(rid.astype(str))
    g = df.groupby("_repo_id", sort=False)
    feat_cols = [c for c in X.columns if c in df.columns]
    X_repo = g[feat_cols].mean()
    targ_cols = [c for c in outcomes.columns if c in df.columns]
    parts = {}
    for c in targ_cols:
        if c == "n_runs":
            parts[c] = g[c].sum()
        else:
            parts[c] = g[c].mean()
    outcomes_repo = pd.DataFrame(parts).reindex(X_repo.index)
    tasks_per_repo = g.size().rename("n_tasks_in_repo")
    return X_repo, outcomes_repo, tasks_per_repo


def benchmark_hyperparameter_manifest() -> Dict[str, Any]:
    """Provenance for all learners: same fixed literals as ``task_level_model_comparison`` (not tuned)."""
    return {
        "reference_module": "scripts/task_level_model_comparison.py",
        "optimization_on_this_dataset": (
            "None. Hyperparameters are **not** selected by grid search, Bayesian optimization, "
            "or nested cross-validation in the comparison script or here. They are fixed "
            "defaults chosen once for a simple benchmark (linear + tree ensembles + gradient boosting). "
            "Reported metrics describe these specific configurations, not a tuned optimum."
        ),
        "LogisticRegression": {
            "penalty": "l1",
            "solver": "lbfgs",
            "max_iter": 2000,
            "random_state": 42,
            "feature_scaling": "StandardScaler fit on train, transform train/test",
        },
        "Ridge": {"alpha": 1.0, "random_state": 42, "feature_scaling": "StandardScaler"},
        "RandomForestClassifier": {
            "n_estimators": 300,
            "max_depth": None,
            "max_features": "sqrt",
            "min_samples_leaf": 3,
            "random_state": 42,
            "n_jobs": 4,
        },
        "RandomForestRegressor": {
            "n_estimators": 300,
            "max_depth": None,
            "max_features": "sqrt",
            "min_samples_leaf": 3,
            "random_state": 42,
            "n_jobs": 4,
        },
        "XGBClassifier": xgb_locked_hyperparameter_dict(is_regression=False),
        "XGBRegressor": xgb_locked_hyperparameter_dict(is_regression=True),
    }


def _cv_split_iterator(
    X: pd.DataFrame,
    y: np.ndarray,
    n_splits: int,
    random_state: int,
    *,
    groups: Optional[np.ndarray],
    group_by_repo: bool,
):
    """Same splitting policy as the former XGB-only CV (StratifiedKFold / Group*)."""
    try:
        ttype = type_of_target(y)
    except Exception:
        ttype = "continuous"
    is_classification = ttype in ("binary", "multiclass")

    if group_by_repo:
        if groups is None:
            raise ValueError("group_by_repo requires groups array aligned with X rows")
        g = np.asarray(groups)
        if is_classification:
            splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            return is_classification, splitter.split(X, y, g)
        splitter = GroupKFold(n_splits=n_splits)
        return is_classification, splitter.split(X, y, groups=g)

    if is_classification:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        return is_classification, splitter.split(X, y)
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return is_classification, splitter.split(X)


def _safe_clf_fold_metrics(y_va: np.ndarray, prob: np.ndarray) -> Dict[str, float]:
    """Match ``task_level_model_comparison.clf_metrics``: F1/MCC at threshold 0.5."""
    try:
        return clf_metrics(y_va, prob, threshold=0.5)
    except Exception:
        return {"auc": float("nan"), "pr_auc": float("nan"), "brier": float("nan"), "f1": float("nan"), "mcc": float("nan")}


def _safe_reg_fold_metrics(y_va: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    try:
        return reg_metrics(y_va, pred)
    except Exception:
        return {"rmse": float("nan"), "mae": float("nan"), "r2": float("nan")}


def cv_evaluate_benchmark_models(
    X: pd.DataFrame,
    y: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
    verbose: bool = False,
    *,
    groups: Optional[np.ndarray] = None,
    group_by_repo: bool = False,
) -> Dict[str, Any]:
    """
    K-fold CV for **Logistic, RandomForest, XGBoost** (classification) or **Ridge, RF, XGBoost**
    (regression), using the same folds as the prior XGB-only CV block.
    """
    is_classification, split_iter = _cv_split_iterator(
        X, y, n_splits, random_state, groups=groups, group_by_repo=group_by_repo
    )

    if is_classification:
        model_names = ["Logistic", "RandomForest", "XGBoost"]
    else:
        model_names = ["Ridge", "RandomForest", "XGBoost"]

    per_model_folds: Dict[str, List[Dict[str, Any]]] = {m: [] for m in model_names}

    fold_idx = 0
    for train_idx, test_idx in split_iter:
        fold_idx += 1
        Xtr, Xva = X.iloc[train_idx], X.iloc[test_idx]
        ytr, yva = y[train_idx], y[test_idx]
        if verbose:
            print(f"  CV fold {fold_idx}/{n_splits}: train={len(train_idx)} val={len(test_idx)}")

        if is_classification:
            # Logistic (scaled)
            try:
                scaler = StandardScaler()
                Xtr_s = scaler.fit_transform(Xtr)
                Xva_s = scaler.transform(Xva)
                lr = LogisticRegression(penalty="l1", solver="lbfgs", max_iter=2000, random_state=42)
                lr.fit(Xtr_s, ytr)
                prob_lr = lr.predict_proba(Xva_s)[:, 1]
                per_model_folds["Logistic"].append({"fold": fold_idx, **_safe_clf_fold_metrics(yva, prob_lr)})
            except Exception:
                per_model_folds["Logistic"].append({"fold": fold_idx, **_safe_clf_fold_metrics(yva, np.full(len(yva), 0.5))})

            # RF
            try:
                rf = RandomForestClassifier(
                    n_estimators=300,
                    max_depth=None,
                    max_features="sqrt",
                    min_samples_leaf=3,
                    random_state=42,
                    n_jobs=4,
                )
                rf.fit(Xtr, ytr)
                prob_rf = rf.predict_proba(Xva)[:, 1]
                per_model_folds["RandomForest"].append({"fold": fold_idx, **_safe_clf_fold_metrics(yva, prob_rf)})
            except Exception:
                per_model_folds["RandomForest"].append({"fold": fold_idx, **_safe_clf_fold_metrics(yva, np.full(len(yva), 0.5))})

            # XGB
            try:
                xgb_m = build_xgb_model(is_regression=False)
                xgb_m.fit(Xtr, ytr)
                prob_x = xgb_m.predict_proba(Xva)[:, 1]
                per_model_folds["XGBoost"].append({"fold": fold_idx, **_safe_clf_fold_metrics(yva, prob_x)})
            except Exception:
                per_model_folds["XGBoost"].append({"fold": fold_idx, **_safe_clf_fold_metrics(yva, np.full(len(yva), 0.5))})
        else:
            # Ridge (scaled), preds clipped to [0,1] like comparison script
            try:
                scaler = StandardScaler()
                Xtr_s = scaler.fit_transform(Xtr)
                Xva_s = scaler.transform(Xva)
                ridge = Ridge(alpha=1.0, random_state=42)
                ridge.fit(Xtr_s, ytr)
                pred_r = np.clip(ridge.predict(Xva_s), 0.0, 1.0)
                per_model_folds["Ridge"].append({"fold": fold_idx, **_safe_reg_fold_metrics(yva, pred_r)})
            except Exception:
                z = np.clip(np.full(len(yva), float(np.mean(ytr))), 0.0, 1.0)
                per_model_folds["Ridge"].append({"fold": fold_idx, **_safe_reg_fold_metrics(yva, z)})

            try:
                rf = RandomForestRegressor(
                    n_estimators=300,
                    max_depth=None,
                    max_features="sqrt",
                    min_samples_leaf=3,
                    random_state=42,
                    n_jobs=4,
                )
                rf.fit(Xtr, ytr)
                pred_rf = np.clip(rf.predict(Xva), 0.0, 1.0)
                per_model_folds["RandomForest"].append({"fold": fold_idx, **_safe_reg_fold_metrics(yva, pred_rf)})
            except Exception:
                z = np.clip(np.full(len(yva), float(np.mean(ytr))), 0.0, 1.0)
                per_model_folds["RandomForest"].append({"fold": fold_idx, **_safe_reg_fold_metrics(yva, z)})

            try:
                xgb_m = build_xgb_model(is_regression=True)
                xgb_m.fit(Xtr, ytr)
                pred_x = np.clip(xgb_m.predict(Xva), 0.0, 1.0)
                per_model_folds["XGBoost"].append({"fold": fold_idx, **_safe_reg_fold_metrics(yva, pred_x)})
            except Exception:
                z = np.clip(np.full(len(yva), float(np.mean(ytr))), 0.0, 1.0)
                per_model_folds["XGBoost"].append({"fold": fold_idx, **_safe_reg_fold_metrics(yva, z)})

    if is_classification:
        keys = ["auc", "roc_auc", "pr_auc", "brier", "f1", "mcc"]
    else:
        keys = ["rmse", "mae", "r2"]

    summary: Dict[str, Any] = {"models": {}, "group_by_repo": group_by_repo, "is_classification": is_classification}
    for mname, rows in per_model_folds.items():
        per_metric: Dict[str, Dict[str, float]] = {}
        for k in keys:
            vals = [float(r[k]) for r in rows if k in r and not np.isnan(float(r[k]))]
            if vals:
                per_metric[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
            else:
                per_metric[k] = {"mean": float("nan"), "std": float("nan")}
        summary["models"][mname] = {"folds": rows, "per_metric": per_metric}

    return summary


def benchmark_test_metrics_classification(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    *,
    xgb_fitted: Optional[Any] = None,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, np.ndarray]]:
    """Test metrics at **0.5** threshold for F1/MCC — matches ``run_classification`` / ``clf_metrics``.

    If ``xgb_fitted`` is provided (already fit on ``X_train``), it is used for the XGBoost row
    so the pipeline does not train two identical XGB models on the full training set.

    Returns ``(metrics_per_model, test_positive_class_prob_per_model)`` for ROC/AUC artifacts.
    """
    out: Dict[str, Dict[str, float]] = {}
    test_probs: Dict[str, np.ndarray] = {}
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(X_train)
    Xte_s = scaler.transform(X_test)
    lr = LogisticRegression(penalty="l1", solver="lbfgs", max_iter=2000, random_state=42)
    lr.fit(Xtr_s, y_train)
    pr_lr = lr.predict_proba(Xte_s)[:, 1]
    test_probs["Logistic"] = np.asarray(pr_lr, dtype=float)
    out["Logistic"] = clf_metrics(y_test, pr_lr, threshold=0.5)

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        max_features="sqrt",
        min_samples_leaf=3,
        random_state=42,
        n_jobs=4,
    )
    rf.fit(X_train, y_train)
    pr_rf = rf.predict_proba(X_test)[:, 1]
    test_probs["RandomForest"] = np.asarray(pr_rf, dtype=float)
    out["RandomForest"] = clf_metrics(y_test, pr_rf, threshold=0.5)

    if xgb_fitted is not None:
        pr_x = xgb_fitted.predict_proba(X_test)[:, 1]
    else:
        xgb_m = build_xgb_model(is_regression=False)
        xgb_m.fit(X_train, y_train)
        pr_x = xgb_m.predict_proba(X_test)[:, 1]
    test_probs["XGBoost"] = np.asarray(pr_x, dtype=float)
    out["XGBoost"] = clf_metrics(y_test, pr_x, threshold=0.5)
    return out, test_probs


def benchmark_test_metrics_regression(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    *,
    xgb_fitted: Optional[Any] = None,
) -> Dict[str, Dict[str, float]]:
    """Matches ``run_regression`` (Ridge/RF/XGB, clip to [0,1])."""
    out: Dict[str, Dict[str, float]] = {}
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(X_train)
    Xte_s = scaler.transform(X_test)
    ridge = Ridge(alpha=1.0, random_state=42)
    ridge.fit(Xtr_s, y_train)
    out["Ridge"] = reg_metrics(y_test, np.clip(ridge.predict(Xte_s), 0.0, 1.0))

    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        max_features="sqrt",
        min_samples_leaf=3,
        random_state=42,
        n_jobs=4,
    )
    rf.fit(X_train, y_train)
    out["RandomForest"] = reg_metrics(y_test, np.clip(rf.predict(X_test), 0.0, 1.0))

    if xgb_fitted is not None:
        out["XGBoost"] = reg_metrics(y_test, np.clip(xgb_fitted.predict(X_test), 0.0, 1.0))
    else:
        xgb_m = build_xgb_model(is_regression=True)
        xgb_m.fit(X_train, y_train)
        out["XGBoost"] = reg_metrics(y_test, np.clip(xgb_m.predict(X_test), 0.0, 1.0))
    return out


def reg_metrics(y_true, y_pred) -> Dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _min_class_count(y: np.ndarray) -> int:
    _, c = np.unique(np.asarray(y), return_counts=True)
    return int(np.min(c)) if len(c) else 0


def make_inner_cv_splitter(
    n_inner: int,
    random_state: int,
    y: np.ndarray,
    *,
    groups: Optional[np.ndarray],
    group_by_repo: bool,
    is_classification: bool,
) -> Any:
    """Inner CV for RandomizedSearchCV (grouped when ``--group-by-repo``)."""
    if group_by_repo and groups is not None:
        g = np.asarray(groups)
        ug = len(np.unique(g))
        nk = max(2, min(n_inner, max(2, ug - 1)))
        if is_classification:
            return StratifiedGroupKFold(n_splits=nk, shuffle=True, random_state=random_state)
        return GroupKFold(n_splits=nk)

    if is_classification:
        mcs = _min_class_count(y)
        nk = max(2, min(n_inner, mcs))
        return StratifiedKFold(n_splits=nk, shuffle=True, random_state=random_state)
    return KFold(n_splits=max(2, min(n_inner, max(2, len(y) // 4))), shuffle=True, random_state=random_state)


def _rs_fit_kwargs(groups: Optional[np.ndarray], group_by_repo: bool) -> Dict[str, Any]:
    if group_by_repo and groups is not None:
        return {"groups": np.asarray(groups)}
    return {}


def clf_metrics_youden_train(
    y_test: np.ndarray,
    prob_test: np.ndarray,
    y_train: np.ndarray,
    prob_train: np.ndarray,
) -> Dict[str, float]:
    thr = optimal_threshold_youden_j(y_train, prob_train)
    m = clf_metrics(y_test, prob_test, threshold=thr)
    m["optimal_threshold_youden_j_train"] = thr
    return m


def tune_classifiers_full_train(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    *,
    groups: Optional[np.ndarray],
    group_by_repo: bool,
    tuning_n_iter: int,
    tuning_inner_cv: int,
    random_state: int,
    verbose: bool,
    xgb_fixed_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """RandomizedSearchCV per classifier on train; test metrics + Youden threshold from train."""
    import xgboost as xgb

    inner_cv = make_inner_cv_splitter(
        tuning_inner_cv, random_state, y_train, groups=groups, group_by_repo=group_by_repo, is_classification=True
    )
    fit_kw = _rs_fit_kwargs(groups, group_by_repo)
    out: Dict[str, Any] = {}

    # --- Logistic (scaled) ---
    pipe_lr = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "lr",
                LogisticRegression(penalty="l1", solver="liblinear", max_iter=6000, random_state=random_state),
            ),
        ]
    )
    try:
        try:
            from scipy.stats import loguniform

            lr_dist = {"lr__C": loguniform(1e-4, 1e3)}
        except Exception:
            lr_dist = {"lr__C": np.logspace(-4, 3, 35).tolist()}
        search_lr = RandomizedSearchCV(
            pipe_lr,
            lr_dist,
            n_iter=max(5, min(tuning_n_iter, 40)),
            scoring="roc_auc",
            cv=inner_cv,
            random_state=random_state,
            refit=True,
            n_jobs=-1,
            error_score=np.nan,
        )
        search_lr.fit(X_train, y_train, **fit_kw)
        est_lr = search_lr.best_estimator_
        pr_tr = est_lr.predict_proba(X_train)[:, 1]
        pr_te = est_lr.predict_proba(X_test)[:, 1]
        out["Logistic"] = {
            "best_params": search_lr.best_params_,
            "best_cv_score_roc_auc": float(search_lr.best_score_) if np.isfinite(search_lr.best_score_) else None,
            "test_metrics": clf_metrics_youden_train(y_test, pr_te, y_train, pr_tr),
            "estimator": est_lr,
        }
        if verbose:
            print("  Tuned Logistic:", search_lr.best_params_, "CV AUC", search_lr.best_score_)
    except Exception as e:
        out["Logistic"] = {"error": str(e)}

    # --- Random forest ---
    try:
        rf_base = RandomForestClassifier(random_state=random_state, n_jobs=4)
        rf_dist = {
            "n_estimators": [150, 250, 350, 500],
            "max_depth": [None, 8, 12, 16, 24, 32],
            "min_samples_leaf": [1, 2, 4, 8],
            "max_features": ["sqrt", 0.25, 0.4, 0.6],
        }
        search_rf = RandomizedSearchCV(
            rf_base,
            rf_dist,
            n_iter=max(8, tuning_n_iter),
            scoring="roc_auc",
            cv=inner_cv,
            random_state=random_state + 1,
            refit=True,
            n_jobs=-1,
            error_score=np.nan,
        )
        search_rf.fit(X_train, y_train, **fit_kw)
        est_rf = search_rf.best_estimator_
        pr_tr = est_rf.predict_proba(X_train)[:, 1]
        pr_te = est_rf.predict_proba(X_test)[:, 1]
        out["RandomForest"] = {
            "best_params": search_rf.best_params_,
            "best_cv_score_roc_auc": float(search_rf.best_score_) if np.isfinite(search_rf.best_score_) else None,
            "test_metrics": clf_metrics_youden_train(y_test, pr_te, y_train, pr_tr),
            "estimator": est_rf,
        }
        if verbose:
            print("  Tuned RandomForest:", search_rf.best_params_, "CV AUC", search_rf.best_score_)
    except Exception as e:
        out["RandomForest"] = {"error": str(e)}

    # --- XGBoost ---
    try:
        if xgb_fixed_params:
            if verbose:
                print("  Fitting XGBoost from paper hyperparameters (search skipped):", xgb_fixed_params)
            est_x = xgb.XGBClassifier(
                objective="binary:logistic",
                eval_metric="auc",
                random_state=random_state,
                n_jobs=1,
                tree_method="hist",
                verbosity=0,
                **xgb_fixed_params,
            )
            est_x.fit(X_train, y_train)
            pr_tr = est_x.predict_proba(X_train)[:, 1]
            pr_te = est_x.predict_proba(X_test)[:, 1]
            out["XGBoost"] = {
                "best_params": dict(xgb_fixed_params),
                "best_cv_score_roc_auc": None,
                "hyperparameter_source": "paper_run",
                "test_metrics": clf_metrics_youden_train(y_test, pr_te, y_train, pr_tr),
                "estimator": est_x,
            }
            if verbose:
                print("  Refit XGBoost:", xgb_fixed_params)
        else:
            xgb_base = xgb.XGBClassifier(
                objective="binary:logistic",
                eval_metric="auc",
                random_state=random_state,
                n_jobs=2,
                tree_method="hist",
                verbosity=0,
            )
            xgb_dist = {
                "n_estimators": [100, 200, 300, 400, 600],
                "max_depth": [3, 4, 5, 6, 8],
                "learning_rate": [0.01, 0.03, 0.05, 0.08, 0.12],
                "subsample": [0.6, 0.75, 0.85, 1.0],
                "colsample_bytree": [0.6, 0.75, 0.85, 1.0],
                "min_child_weight": [1, 2, 3, 5, 8],
                "reg_lambda": [1.0, 2.0, 5.0],
            }
            search_x = RandomizedSearchCV(
                xgb_base,
                xgb_dist,
                n_iter=max(10, tuning_n_iter),
                scoring="roc_auc",
                cv=inner_cv,
                random_state=random_state + 2,
                refit=True,
                n_jobs=1,
                error_score=np.nan,
            )
            search_x.fit(X_train, y_train, **fit_kw)
            est_x = search_x.best_estimator_
            pr_tr = est_x.predict_proba(X_train)[:, 1]
            pr_te = est_x.predict_proba(X_test)[:, 1]
            out["XGBoost"] = {
                "best_params": search_x.best_params_,
                "best_cv_score_roc_auc": float(search_x.best_score_) if np.isfinite(search_x.best_score_) else None,
                "test_metrics": clf_metrics_youden_train(y_test, pr_te, y_train, pr_tr),
                "estimator": est_x,
            }
            if verbose:
                print("  Tuned XGBoost:", search_x.best_params_, "CV AUC", search_x.best_score_)
    except Exception as e:
        out["XGBoost"] = {"error": str(e), "estimator": None}

    return out


def tune_regressors_full_train(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    *,
    groups: Optional[np.ndarray],
    group_by_repo: bool,
    tuning_n_iter: int,
    tuning_inner_cv: int,
    random_state: int,
    verbose: bool,
    xgb_fixed_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    import xgboost as xgb

    inner_cv = make_inner_cv_splitter(
        tuning_inner_cv, random_state, y_train, groups=groups, group_by_repo=group_by_repo, is_classification=False
    )
    fit_kw = _rs_fit_kwargs(groups, group_by_repo)
    out: Dict[str, Any] = {}

    pipe_ridge = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("ridge", Ridge(random_state=random_state)),
        ]
    )
    try:
        try:
            from scipy.stats import loguniform

            rd_dist = {"ridge__alpha": loguniform(1e-4, 1e3)}
        except Exception:
            rd_dist = {"ridge__alpha": np.logspace(-4, 3, 35).tolist()}
        search_rd = RandomizedSearchCV(
            pipe_ridge,
            rd_dist,
            n_iter=max(5, min(tuning_n_iter, 40)),
            scoring="neg_mean_squared_error",
            cv=inner_cv,
            random_state=random_state,
            refit=True,
            n_jobs=-1,
            error_score=np.nan,
        )
        search_rd.fit(X_train, y_train, **fit_kw)
        est = search_rd.best_estimator_
        pred_te = np.clip(est.predict(X_test), 0.0, 1.0)
        out["Ridge"] = {
            "best_params": search_rd.best_params_,
            "best_cv_score_neg_mse": float(search_rd.best_score_) if np.isfinite(search_rd.best_score_) else None,
            "test_metrics": reg_metrics(y_test, pred_te),
        }
        if verbose:
            print("  Tuned Ridge:", search_rd.best_params_, "CV neg-MSE", search_rd.best_score_)
    except Exception as e:
        out["Ridge"] = {"error": str(e)}

    try:
        rf_base = RandomForestRegressor(random_state=random_state, n_jobs=4)
        rf_dist = {
            "n_estimators": [150, 250, 350, 500],
            "max_depth": [None, 8, 12, 16, 24, 32],
            "min_samples_leaf": [1, 2, 4, 8],
            "max_features": ["sqrt", 0.25, 0.4, 0.6],
        }
        search_rf = RandomizedSearchCV(
            rf_base,
            rf_dist,
            n_iter=max(8, tuning_n_iter),
            scoring="neg_mean_squared_error",
            cv=inner_cv,
            random_state=random_state + 1,
            refit=True,
            n_jobs=-1,
            error_score=np.nan,
        )
        search_rf.fit(X_train, y_train, **fit_kw)
        est = search_rf.best_estimator_
        pred_te = np.clip(est.predict(X_test), 0.0, 1.0)
        out["RandomForest"] = {
            "best_params": search_rf.best_params_,
            "best_cv_score_neg_mse": float(search_rf.best_score_) if np.isfinite(search_rf.best_score_) else None,
            "test_metrics": reg_metrics(y_test, pred_te),
        }
        if verbose:
            print("  Tuned RandomForest:", search_rf.best_params_)
    except Exception as e:
        out["RandomForest"] = {"error": str(e)}

    try:
        if xgb_fixed_params:
            if verbose:
                print("  Fitting XGBoost from paper hyperparameters (search skipped):", xgb_fixed_params)
            est_x = xgb.XGBRegressor(
                objective="reg:squarederror",
                random_state=random_state,
                n_jobs=1,
                tree_method="hist",
                verbosity=0,
                **xgb_fixed_params,
            )
            est_x.fit(X_train, y_train)
            pred_te = np.clip(est_x.predict(X_test), 0.0, 1.0)
            out["XGBoost"] = {
                "best_params": dict(xgb_fixed_params),
                "best_cv_score_neg_mse": None,
                "hyperparameter_source": "paper_run",
                "test_metrics": reg_metrics(y_test, pred_te),
                "estimator": est_x,
            }
            if verbose:
                print("  Refit XGBoost:", xgb_fixed_params)
        else:
            xgb_base = xgb.XGBRegressor(
                objective="reg:squarederror",
                random_state=random_state,
                n_jobs=2,
                tree_method="hist",
                verbosity=0,
            )
            xgb_dist = {
                "n_estimators": [100, 200, 300, 400, 600],
                "max_depth": [3, 4, 5, 6, 8],
                "learning_rate": [0.01, 0.03, 0.05, 0.08, 0.12],
                "subsample": [0.6, 0.75, 0.85, 1.0],
                "colsample_bytree": [0.6, 0.75, 0.85, 1.0],
                "min_child_weight": [1, 2, 3, 5, 8],
                "reg_lambda": [1.0, 2.0, 5.0],
            }
            search_x = RandomizedSearchCV(
                xgb_base,
                xgb_dist,
                n_iter=max(10, tuning_n_iter),
                scoring="neg_mean_squared_error",
                cv=inner_cv,
                random_state=random_state + 2,
                refit=True,
                n_jobs=1,
                error_score=np.nan,
            )
            search_x.fit(X_train, y_train, **fit_kw)
            est_x = search_x.best_estimator_
            pred_te = np.clip(est_x.predict(X_test), 0.0, 1.0)
            out["XGBoost"] = {
                "best_params": search_x.best_params_,
                "best_cv_score_neg_mse": float(search_x.best_score_) if np.isfinite(search_x.best_score_) else None,
                "test_metrics": reg_metrics(y_test, pred_te),
                "estimator": est_x,
            }
            if verbose:
                print("  Tuned XGBoost:", search_x.best_params_)
    except Exception as e:
        out["XGBoost"] = {"error": str(e), "estimator": None}

    return out


def cv_evaluate_nested_tuned(
    X: pd.DataFrame,
    y: np.ndarray,
    n_splits: int,
    random_state: int,
    verbose: bool,
    *,
    groups: Optional[np.ndarray],
    group_by_repo: bool,
    is_regression: bool,
    tuning_n_iter: int,
    tuning_inner_cv: int,
) -> Dict[str, Any]:
    """
    Outer CV folds; **inside each outer train fold** run RandomizedSearchCV (inner CV).
    Classification: validation metrics use Youden threshold from **outer train** predictions.
    """
    is_classification, split_iter = _cv_split_iterator(
        X, y, n_splits, random_state, groups=groups, group_by_repo=group_by_repo
    )
    names = ["Ridge", "RandomForest", "XGBoost"] if is_regression else ["Logistic", "RandomForest", "XGBoost"]
    per_model_folds: Dict[str, List[Dict[str, Any]]] = {m: [] for m in names}

    fold_idx = 0
    n_iter_nested = max(6, tuning_n_iter // 2)
    inner_splits_nested = max(2, min(tuning_inner_cv, 4))

    for train_idx, val_idx in split_iter:
        fold_idx += 1
        Xtr, Xva = X.iloc[train_idx], X.iloc[val_idx]
        ytr, yva = y[train_idx], y[val_idx]
        gtr = groups[train_idx] if groups is not None else None
        if verbose:
            print(f"  Nested CV outer fold {fold_idx}/{n_splits}: train={len(train_idx)} val={len(val_idx)}")

        if is_regression:
            pack = tune_regressors_full_train(
                Xtr,
                Xva,
                ytr,
                yva,
                groups=gtr,
                group_by_repo=group_by_repo,
                tuning_n_iter=n_iter_nested,
                tuning_inner_cv=inner_splits_nested,
                random_state=random_state + fold_idx,
                verbose=False,
            )
            for m in names:
                row: Dict[str, Any] = {"fold": fold_idx}
                if "test_metrics" in pack.get(m, {}):
                    row.update(pack[m]["test_metrics"])
                else:
                    row["error"] = pack.get(m, {}).get("error", "unknown")
                per_model_folds[m].append(row)
        else:
            pack = tune_classifiers_full_train(
                Xtr,
                Xva,
                ytr,
                yva,
                groups=gtr,
                group_by_repo=group_by_repo,
                tuning_n_iter=n_iter_nested,
                tuning_inner_cv=inner_splits_nested,
                random_state=random_state + fold_idx,
                verbose=False,
            )
            for m in names:
                row = {"fold": fold_idx}
                if "test_metrics" in pack.get(m, {}):
                    row.update(pack[m]["test_metrics"])
                else:
                    row["error"] = pack.get(m, {}).get("error", "unknown")
                per_model_folds[m].append(row)

    if is_regression:
        keys = ["rmse", "mae", "r2"]
    else:
        keys = ["auc", "roc_auc", "pr_auc", "brier", "f1", "mcc"]

    summary: Dict[str, Any] = {
        "models": {},
        "group_by_repo": group_by_repo,
        "is_classification": is_classification,
        "protocol": "nested_randomized_search_per_outer_train_fold",
    }
    for mname, rows in per_model_folds.items():
        per_metric: Dict[str, Dict[str, float]] = {}
        for k in keys:
            vals = []
            for r in rows:
                if k in r and isinstance(r[k], (int, float)) and not (isinstance(r[k], float) and np.isnan(r[k])):
                    vals.append(float(r[k]))
            if vals:
                per_metric[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
            else:
                per_metric[k] = {"mean": float("nan"), "std": float("nan")}
        summary["models"][mname] = {"folds": rows, "per_metric": per_metric}

    return summary



def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Task-level VIF + data-driven RandomizedSearchCV (default) + tuned XGBoost SHAP, "
            "or --benchmark-fixed-params for legacy comparison literals."
        )
    )
    parser.add_argument(
        "--features-path",
        default=None,
        help="Parquet/CSV feature table (with target column(s)). Ignored when --use-task-loader.",
    )
    parser.add_argument("--target", default=None, help="Single target column (backward compatible).")
    parser.add_argument(
        "--targets",
        default=None,
        help="Comma-separated targets, e.g. pass_rate,any_success,maj_success",
    )
    parser.add_argument(
        "--effect-sizes",
        default=None,
        help="Optional CSV with 'feature' and 'effect_size' to restrict to top-N columns before VIF.",
    )
    parser.add_argument("--top-n", type=int, default=100, help="Top features from --effect-sizes when used.")
    parser.add_argument(
        "--use-task-loader",
        dest="use_task_loader",
        action="store_true",
        default=True,
        help="Load canonical task-level parquet merge (default: on).",
    )
    parser.add_argument(
        "--no-use-task-loader",
        dest="use_task_loader",
        action="store_false",
        help="Rare: load --features-path.",
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="Output directory. Default: CODERFORGE_ARTIFACTS_DIR/reports/task_level_model_shap",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Holdout fraction for SHAP/explainability.")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--vif-threshold", type=float, default=10.0, help="VIF cutoff (comparison script uses 10).")
    parser.add_argument("--n-splits", type=int, default=10, help="CV folds on the training set (post-VIF columns).")
    parser.add_argument(
        "--benchmark-fixed-params",
        action="store_true",
        help="Legacy: fixed hyperparameters from task_level_model_comparison; outer CV + F1/MCC at 0.5.",
    )
    parser.add_argument(
        "--tuning-n-iter",
        type=int,
        default=28,
        help="RandomizedSearchCV draws per model (default data-driven mode).",
    )
    parser.add_argument(
        "--tuning-inner-cv",
        type=int,
        default=4,
        help="Inner CV folds inside each RandomizedSearchCV.",
    )
    parser.add_argument(
        "--nested-cv-tune",
        action="store_true",
        help="Nested tuning: RandomizedSearchCV on each outer CV train fold (slow, rigorous CV).",
    )
    parser.add_argument(
        "--group-by-repo",
        action="store_true",
        help="Task-level only: GroupShuffleSplit train/test (no repo overlap) and "
        "StratifiedGroupKFold / GroupKFold for CV. Ignored when --aggregate-by-repo is set.",
    )
    parser.add_argument(
        "--aggregate-by-repo",
        action="store_true",
        help="Collapse task-level rows to one row per repo (mean-pooled features; mean/sum targets) "
        "before VIF/CV/SHAP. Use standard train/test and CV on repo rows (not --group-by-repo).",
    )
    parser.add_argument(
        "--shap-sample",
        type=int,
        default=None,
        help=(
            "Max **held-out test** rows passed to TreeExplainer only (SHAP + optional interactions). "
            "Does not change train/test split, tuning, or any reported metrics / ROC / calibration "
            "(those always use the full test matrix)."
        ),
    )
    parser.add_argument(
        "--shap-parallel-n-jobs",
        type=int,
        default=None,
        help=(
            "Parallel row chunks for TreeExplainer shap_values + shap_interaction_values (joblib loky). "
            "Default: SLURM_CPUS_PER_TASK if set, else min(16, CPU count). Use 1 to force single-process."
        ),
    )
    parser.add_argument("--shap-top-interactions", type=int, default=50)
    parser.add_argument("--no-interactions", action="store_true", help="Skip shap_interaction_values.")
    parser.add_argument(
        "--search-xgboost",
        action="store_true",
        help=(
            "Run RandomizedSearchCV for XGBoost. Default is to refit the published "
            "paper_run hyperparameters so re-runs keep the camera-ready 400-tree model class."
        ),
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.outdir is None:
        out_root = default_tuned_models_dir()
    else:
        out_root = Path(args.outdir)
    out_root.mkdir(parents=True, exist_ok=True)
    if args.verbose:
        print(f"Output directory: {out_root.resolve()}")

    tlmc = _load_comparison_module()

    # --- Load X, outcomes ---
    if args.use_task_loader:
        assert_replication_dataset()
        if args.verbose:
            print(f"Task loader: {default_dataset_path()}")
        X, task_outcomes = load_task_level_data()
    else:
        if not args.features_path:
            raise ValueError("Provide --features-path or enable --use-task-loader")
        feats = read_table(args.features_path)
        if args.targets:
            tlist = [t.strip() for t in args.targets.split(",") if t.strip()]
        elif args.target:
            tlist = [args.target]
        else:
            raise ValueError("With --no-use-task-loader, set --target or --targets")
        if args.aggregate_by_repo and "repo_id" not in feats.columns:
            raise ValueError("--aggregate-by-repo requires a repo_id column in the features table")
        task_outcomes = feats[tlist].copy()
        X = feats.drop(columns=[c for c in tlist if c in feats.columns], errors="ignore")
        X = X.select_dtypes(include=[np.number])
        X = X.loc[:, X.nunique(dropna=False) > 1]
        X = X.fillna(X.median(numeric_only=True))

    if args.effect_sizes:
        effects = read_table(args.effect_sizes)
        selected = select_features_from_effects(effects, top_n=args.top_n)
        selected = [c for c in selected if c in X.columns]
        if not selected:
            raise ValueError("No effect-size features found in X after filtering")
        X = X[selected]

    if args.targets:
        targets_list = [t.strip() for t in args.targets.split(",") if t.strip()]
    elif args.target:
        targets_list = [args.target]
    elif args.use_task_loader:
        targets_list = ["pass_rate", "any_success", "maj_success"]
    else:
        raise ValueError("Specify --targets or --target")

    if args.aggregate_by_repo:
        if args.group_by_repo and args.verbose:
            print(
                "Note: --aggregate-by-repo disables task-level --group-by-repo "
                "(each modeling row is already one repo)."
            )
        if args.use_task_loader:
            rs = load_repo_id_per_task(X.index)
        else:
            rs = feats["repo_id"].reindex(X.index)
        X, task_outcomes, tasks_per_repo = aggregate_task_level_to_repo(X, task_outcomes, rs)
        counts_df = tasks_per_repo.reset_index(name="n_tasks_in_repo")
        c0 = counts_df.columns[0]
        counts_df = counts_df.rename(columns={c0: "repo_id"})
        counts_df.to_csv(out_root / "shared_aggregate_repo_level_task_counts.csv", index=False)
        if args.verbose:
            print(f"Aggregated to {len(X)} repos (shared_aggregate_repo_level_task_counts.csv).")
        repo_grp = None
        group_tasks_by_repo = False
    elif args.group_by_repo:
        if args.use_task_loader:
            repo_series = load_repo_id_per_task(X.index)
        else:
            if "repo_id" not in feats.columns:
                raise ValueError("--group-by-repo requires a repo_id column in the features table")
            repo_series = feats["repo_id"].reindex(X.index)
        if repo_series.isna().any():
            raise ValueError(f"repo_id is missing for {int(repo_series.isna().sum())} rows after alignment with X")
        repo_grp = np.asarray(repo_series.astype(str))
        group_tasks_by_repo = True
    else:
        repo_grp = None
        group_tasks_by_repo = False

    y_any = task_outcomes["any_success"].values if "any_success" in task_outcomes.columns else None
    if y_any is None and args.use_task_loader:
        raise RuntimeError("expected any_success in outcomes")

    idx = np.arange(len(X))
    if group_tasks_by_repo:
        assert repo_grp is not None
        y_split = y_any if y_any is not None else np.zeros(len(X), dtype=int)
        gss = GroupShuffleSplit(
            n_splits=1,
            test_size=args.test_size,
            random_state=args.random_state,
        )
        idx_train, idx_test = next(gss.split(idx, y_split, groups=repo_grp))
    elif y_any is not None and "split" in task_outcomes.columns:
        idx_train, idx_test = train_test_indices_from_targets(task_outcomes, np.asarray(y_any))
    elif y_any is not None:
        y_strat = np.asarray(y_any)
        if y_strat.dtype.kind == "f":
            y_strat = (y_strat >= 0.5).astype(int)
        else:
            y_strat = y_strat.astype(int)
        strat = y_strat if len(np.unique(y_strat)) >= 2 else None
        idx_train, idx_test = train_test_split(
            idx,
            test_size=args.test_size,
            random_state=args.random_state,
            stratify=strat,
        )
    else:
        idx_train, idx_test = train_test_split(idx, test_size=args.test_size, random_state=args.random_state)

    X_train_full = X.iloc[idx_train]
    X_test_full = X.iloc[idx_test]

    if group_tasks_by_repo and repo_grp is not None:
        train_repos = set(repo_grp[idx_train].tolist())
        test_repos = set(repo_grp[idx_test].tolist())
        overlap = train_repos & test_repos
        if overlap:
            raise RuntimeError(f"GroupShuffleSplit leaked repos across train/test: {overlap!r}")
        all_repos = sorted(train_repos | test_repos)
        assign = pd.DataFrame(
            {
                "repo_id": all_repos,
                "split": ["train" if r in train_repos else "test" for r in all_repos],
            }
        )
        assign.to_csv(out_root / "shared_train_test_split_repo_assignment.csv", index=False)
        if args.verbose:
            print(
                f"Repo-grouped split: {len(train_repos)} train repos, {len(test_repos)} test repos, "
                f"{len(idx_train)} train tasks, {len(idx_test)} test tasks"
            )

    # VIF on train only (same helper as comparison script). Outputs are **shared** across
    # all outcome targets (not outcome-specific); see ``shared_train_split_before_xgb_*`` files.
    vif_dir = out_root / "shared_train_split_vif_before_xgb"
    vif_dir.mkdir(parents=True, exist_ok=True)
    X_train_vif, vif_final, vif_dropped = tlmc.select_features_by_vif(
        X_train_full,
        threshold=args.vif_threshold,
        output_dir=vif_dir,
    )
    _rename_shared_vif_outputs(vif_dir)
    X_test_vif = X_test_full[X_train_vif.columns]

    overall_summary: Dict[str, Any] = {
        "n_features_raw": int(X.shape[1]),
        "n_modeling_rows": int(len(X)),
        "n_after_vif_train": int(X_train_vif.shape[1]),
        "vif_dropped_count": len(vif_dropped),
        "group_by_repo": bool(group_tasks_by_repo),
        "aggregate_by_repo": bool(args.aggregate_by_repo),
    }

    cv_groups = repo_grp[idx_train] if group_tasks_by_repo and repo_grp is not None else None

    shap_parallel_jobs = (
        max(1, int(args.shap_parallel_n_jobs))
        if args.shap_parallel_n_jobs is not None
        else _default_shap_parallel_jobs(None)
    )
    overall_summary["shap_row_parallel_jobs"] = shap_parallel_jobs
    if args.verbose:
        print(f"SHAP row-parallel workers: {shap_parallel_jobs} (set SLURM_CPUS_PER_TASK or --shap-parallel-n-jobs)")

    for target in targets_list:
        if target not in task_outcomes.columns:
            raise ValueError(f"target '{target}' not in outcomes / table")
        print(f"\n=== Target: {target} ===")
        y = task_outcomes[target].values
        y_train, y_test = y[idx_train], y[idx_test]

        X_train = X_train_vif.copy()
        X_test = X_test_vif.copy()

        try:
            ttype = type_of_target(y_train)
        except Exception:
            ttype = "continuous"
        is_regression = ttype not in ("binary", "multiclass")

        pfx = outcome_model_prefix(target)
        target_out = out_root / pfx
        target_out.mkdir(parents=True, exist_ok=True)

        vif_final.to_csv(target_out / f"{pfx}_vif_selected_feature_table.csv", index=False)
        pd.DataFrame({"dropped_feature": vif_dropped}).to_csv(
            target_out / f"{pfx}_vif_dropped_features_order.csv", index=False
        )

        try:
            feat_path = target_out / f"{pfx}_X_train_final.parquet"
            X_train.to_parquet(feat_path)
        except Exception:
            pass

        tuned_pack: Dict[str, Any] = {}
        clf_test_probs: Dict[str, np.ndarray] = {}

        if args.benchmark_fixed_params:
            print("Cross-validated benchmark models (fixed literals, train set)...")
            cv_stats = cv_evaluate_benchmark_models(
                X_train,
                y_train,
                n_splits=args.n_splits,
                verbose=args.verbose,
                groups=cv_groups,
                group_by_repo=group_tasks_by_repo,
            )
            (target_out / f"{target}_cv_benchmark_models.json").write_text(json.dumps(cv_stats, indent=2))

            print("Fitting final XGBoost (fixed hyperparameters)...")
            final_model = build_xgb_model(is_regression=is_regression)
            final_model.fit(X_train, y_train)

            if not is_regression:
                all_models_test, clf_test_probs = benchmark_test_metrics_classification(
                    X_train, X_test, y_train, y_test, xgb_fitted=final_model
                )
            else:
                all_models_test = benchmark_test_metrics_regression(
                    X_train, X_test, y_train, y_test, xgb_fitted=final_model
                )
                clf_test_probs = {}
            combined_test = {
                "target": target,
                "reference": "task_level_model_comparison.run_classification / run_regression (fixed literals)",
                "models": all_models_test,
            }
            if not is_regression:
                combined_test["classification_f1_mcc_threshold"] = 0.5
            (target_out / f"{target}_all_models_test_metrics.json").write_text(json.dumps(combined_test, indent=2))
            for mname, mdict in all_models_test.items():
                safe = mname.replace(" ", "_")
                (target_out / f"{target}_{safe}_test_metrics.json").write_text(json.dumps(mdict, indent=2))
            if not is_regression and clf_test_probs:
                for mname, pr in clf_test_probs.items():
                    safe = mname.replace(" ", "_")
                    save_roc_curve_outputs(
                        y_test,
                        pr,
                        target_out / f"{target}_{safe}_roc_curve_holdout_test.csv",
                        target_out / f"{target}_{safe}_roc_curve_holdout_test.png",
                        title=f"ROC — {target} {mname} (held-out test, fixed params)",
                    )
                save_roc_curve_outputs(
                    y_test,
                    clf_test_probs["XGBoost"],
                    target_out / f"{pfx}_roc_curve_holdout_test.csv",
                    target_out / f"{pfx}_roc_curve_holdout_test.png",
                    title=f"ROC — {pfx} (held-out test, XGBoost)",
                )
        else:
            if args.nested_cv_tune:
                print("Nested RandomizedSearchCV (each outer train fold; slow)...")
                cv_stats = cv_evaluate_nested_tuned(
                    X_train,
                    y_train,
                    n_splits=args.n_splits,
                    random_state=args.random_state,
                    verbose=args.verbose,
                    groups=cv_groups,
                    group_by_repo=group_tasks_by_repo,
                    is_regression=is_regression,
                    tuning_n_iter=args.tuning_n_iter,
                    tuning_inner_cv=args.tuning_inner_cv,
                )
            else:
                cv_stats = {
                    "skipped_outer_cv": True,
                    "note": (
                        "RandomizedSearchCV runs on the full VIF-filtered training set only. "
                        "Use --nested-cv-tune for nested search on each outer CV fold (unbiased CV curve)."
                    ),
                }
            (target_out / f"{target}_cv_benchmark_models.json").write_text(
                json.dumps(cv_stats, indent=2, default=str)
            )

            xgb_fixed_params = (
                None if args.search_xgboost else load_paper_xgb_hyperparams(target)
            )
            if xgb_fixed_params is None:
                print(
                    f"RandomizedSearchCV per model (n_iter≈{args.tuning_n_iter}, inner_cv={args.tuning_inner_cv})..."
                )
            else:
                print(
                    "RandomizedSearchCV for non-XGB models; XGBoost refit from paper_run hyperparameters "
                    f"{xgb_fixed_params}"
                )
            if not is_regression:
                tuned_pack = tune_classifiers_full_train(
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    groups=cv_groups,
                    group_by_repo=group_tasks_by_repo,
                    tuning_n_iter=args.tuning_n_iter,
                    tuning_inner_cv=args.tuning_inner_cv,
                    random_state=args.random_state,
                    verbose=args.verbose,
                    xgb_fixed_params=xgb_fixed_params,
                )
            else:
                tuned_pack = tune_regressors_full_train(
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    groups=cv_groups,
                    group_by_repo=group_tasks_by_repo,
                    tuning_n_iter=args.tuning_n_iter,
                    tuning_inner_cv=args.tuning_inner_cv,
                    random_state=args.random_state,
                    verbose=args.verbose,
                    xgb_fixed_params=xgb_fixed_params,
                )

            def _strip_estimators(p: Dict[str, Any]) -> Dict[str, Any]:
                out: Dict[str, Any] = {}
                for k, v in p.items():
                    if isinstance(v, dict):
                        out[k] = {kk: vv for kk, vv in v.items() if kk != "estimator"}
                    else:
                        out[k] = v
                return out

            (target_out / f"{target}_tuned_full_train_report.json").write_text(
                json.dumps(_strip_estimators(tuned_pack), indent=2, default=str)
            )

            xgb_entry = tuned_pack.get("XGBoost", {})
            final_model = xgb_entry.get("estimator")
            if final_model is None:
                raise RuntimeError(
                    "Tuned XGBoost is missing or failed during RandomizedSearchCV. "
                    "Check logs or use --benchmark-fixed-params as a fallback."
                )

            all_models_test = {
                k: v.get("test_metrics", {"error": v.get("error", "missing")}) for k, v in tuned_pack.items()
            }
            combined_test = {
                "target": target,
                "classification_threshold_policy": (
                    "Youden J on train ROC per model for F1/MCC (threshold reported in each model dict)"
                    if not is_regression
                    else "N/A (regression)"
                ),
                "hyperparameter_selection": (
                    f"RandomizedSearchCV on train only; inner_cv={args.tuning_inner_cv}, "
                    f"n_iter≈{args.tuning_n_iter} per family"
                ),
                "models": all_models_test,
            }
            (target_out / f"{target}_all_models_test_metrics.json").write_text(
                json.dumps(combined_test, indent=2, default=str)
            )
            for mname, mdict in all_models_test.items():
                safe = mname.replace(" ", "_")
                (target_out / f"{target}_{safe}_test_metrics.json").write_text(
                    json.dumps(mdict, indent=2, default=str)
                )

        joblib.dump(
            {"model": final_model, "feature_names": list(X_train.columns)},
            target_out / f"{pfx}_final_xgb_model.pkl",
        )

        prob_te: Optional[np.ndarray] = None
        pred_te: Optional[np.ndarray] = None
        if is_regression:
            pred_te = np.clip(final_model.predict(X_test), 0.0, 1.0)
            test_metrics = reg_metrics(y_test, pred_te)
        else:
            prob_te = final_model.predict_proba(X_test)[:, 1]
            prob_tr = final_model.predict_proba(X_train)[:, 1]
            optimal_threshold = optimal_threshold_youden_j(y_train, prob_tr)
            test_metrics = clf_metrics(y_test, prob_te, threshold=optimal_threshold)
            test_metrics["optimal_threshold_youden_j_train"] = optimal_threshold
        (target_out / f"{pfx}_test_metrics.json").write_text(json.dumps(test_metrics, indent=2))
        print("  XGBoost primary test metrics (held-out):", test_metrics)

        if not is_regression and prob_te is not None:
            if args.benchmark_fixed_params:
                try:
                    save_xgb_calibration_outputs(
                        y_test,
                        prob_te,
                        target_out / f"{pfx}_calibration_quantile10_bins.csv",
                        target_out / f"{pfx}_calibration_reliability_diagram.png",
                        title=f"Calibration — {pfx} (held-out test, 10 quantile bins)",
                    )
                    save_roc_curve_outputs(
                        y_test,
                        prob_te,
                        target_out / f"{pfx}_roc_curve_holdout_test.csv",
                        target_out / f"{pfx}_roc_curve_holdout_test.png",
                        title=f"ROC — {pfx} (held-out test, fixed XGBoost)",
                    )
                except Exception:
                    pass
            else:
                for mname, info in tuned_pack.items():
                    est = info.get("estimator")
                    if est is None or not hasattr(est, "predict_proba"):
                        continue
                    try:
                        prob_m = est.predict_proba(X_test)[:, 1]
                        safe = mname.replace(" ", "_")
                        save_xgb_calibration_outputs(
                            y_test,
                            prob_m,
                            target_out / f"{target}_{safe}_calibration_quantile10_bins.csv",
                            target_out / f"{target}_{safe}_calibration_reliability_diagram.png",
                            title=f"Calibration — {target} {mname} (held-out test, tuned)",
                        )
                        save_roc_curve_outputs(
                            y_test,
                            prob_m,
                            target_out / f"{target}_{safe}_roc_curve_holdout_test.csv",
                            target_out / f"{target}_{safe}_roc_curve_holdout_test.png",
                            title=f"ROC — {target} {mname} (held-out test, tuned)",
                        )
                    except Exception:
                        pass
                try:
                    save_xgb_calibration_outputs(
                        y_test,
                        prob_te,
                        target_out / f"{pfx}_calibration_quantile10_bins.csv",
                        target_out / f"{pfx}_calibration_reliability_diagram.png",
                        title=f"Calibration — {pfx} (XGBoost held-out test, tuned)",
                    )
                    save_roc_curve_outputs(
                        y_test,
                        prob_te,
                        target_out / f"{pfx}_roc_curve_holdout_test.csv",
                        target_out / f"{pfx}_roc_curve_holdout_test.png",
                        title=f"ROC — {pfx} (held-out test, tuned XGBoost)",
                    )
                except Exception:
                    pass

        else:
            tuning_record = {
                k: {kk: vv for kk, vv in v.items() if kk != "estimator"} for k, v in tuned_pack.items()
            }
            overall_summary[target] = {
                "cv_benchmark_models": cv_stats,
                "tuning_full_train": tuning_record,
                "test_metrics_all_models_youden_j": all_models_test,
                "test_metrics_xgb": test_metrics,
            }


    (out_root / "run_multitarget_xgb_summary.json").write_text(json.dumps(overall_summary, indent=2, default=str))

    if args.benchmark_fixed_params:
        hyperparameter_policy: Dict[str, Any] = {
            "mode": "benchmark_fixed_literals_from_task_level_model_comparison",
            "summary": (
                "Hyperparameters copied from ``task_level_model_comparison.py``. "
                "F1/MCC in pooled model JSON use threshold 0.5; XGB primary JSON uses Youden J."
            ),
            "all_learners": benchmark_hyperparameter_manifest(),
        }
        calibration_protocol: Dict[str, Any] = {
            "where": "``<outcome>_xgb/`` contains ``<pfx>_calibration_*`` for XGBoost only.",
            "method": "sklearn.calibration.calibration_curve, n_bins=10, strategy=quantile",
        }
        roc_protocol: Dict[str, Any] = {
            "where": (
                "Per classifier: ``<target>_<Model>_roc_curve_holdout_test.csv`` and ``.png``. "
                "XGBoost also: ``<pfx>_roc_curve_holdout_test.*``."
            ),
            "metrics_json": "``auc`` and ``roc_auc`` in each ``*_test_metrics.json`` are the same ROC-AUC (test).",
            "method": "sklearn.metrics.roc_curve + roc_auc_score on held-out test scores.",
        }
        evaluation_notes: Dict[str, Any] = {
            "train_test_split": "Stratified on any_success (or GroupShuffleSplit by repo when enabled).",
            "cv": "See ``<target>_cv_benchmark_models.json`` (fixed-param outer folds).",
            "classification_threshold": (
                "``<target>_all_models_test_metrics.json`` uses 0.5 for F1/MCC; "
                "``<outcome>_xgb_test_metrics.json`` uses Youden J for XGBoost."
            ),
        }
        summary_note = (
            "Fixed-param run: keys ``test_metrics_all_models_threshold_0p5`` and ``test_metrics_xgb`` per target."
        )
    else:
        hyperparameter_policy = {
            "mode": "randomized_search_on_training_set",
            "implementation": "sklearn.model_selection.RandomizedSearchCV",
            "inner_cv_folds": args.tuning_inner_cv,
            "tuning_n_iter_per_model_family": args.tuning_n_iter,
            "classification_scoring": "roc_auc",
            "regression_scoring": "neg_mean_squared_error",
            "nested_cv": (
                "Pass ``--nested-cv-tune`` to repeat RandomizedSearchCV inside each outer CV train fold "
                "(unbiased CV estimate; slow). Default tunes only on the full VIF-filtered training set "
                "and evaluates once on the held-out test set."
            ),
            "caveat": (
                "Test metrics use a single held-out split; for rigorous generalization reporting combine "
                "nested CV with an external validation set or repeated splits."
            ),
        }
        calibration_protocol = {
            "where": "Binary: ``<target>_<Model>_calibration_*`` for each tuned probabilistic model.",
            "method": "calibration_curve, n_bins=10, strategy=quantile",
        }
        roc_protocol = {
            "where": (
                "``<target>_<Model>_roc_curve_holdout_test.{csv,png}`` for each tuned classifier; "
                "``<pfx>_roc_curve_holdout_test.*`` for tuned XGBoost."
            ),
            "metrics_json": "``auc`` / ``roc_auc`` in test metric JSON files (identical ROC-AUC).",
            "method": "roc_curve + roc_auc_score on held-out test probabilities.",
        }
        evaluation_notes = {
            "train_test_split": "Stratified on any_success (or GroupShuffleSplit by repo when enabled).",
            "cv": "See ``<target>_cv_benchmark_models.json`` (nested folds or skipped-outer note).",
            "classification_threshold": (
                "Youden J threshold fit on **train** probabilities, applied to **test** for F1/MCC "
                "(per model). Documented in each model's test metrics dict."
            ),
        }
        summary_note = (
            "Default tuned run: ``tuning_full_train``, ``test_metrics_all_models_youden_j``, ``test_metrics_xgb``."
        )

    manifest: Dict[str, Any] = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "script_path": str((REPLICATION_ROOT / "rq1" / "run_tuned_evaluation.py").resolve()),
        "primary_model": "xgboost",
        "model_slug_for_filenames": PRIMARY_MODEL_SLUG,
        "benchmark_fixed_params": bool(args.benchmark_fixed_params),
        "directory_layout": {
            "per_outcome": f"<outcome>_{PRIMARY_MODEL_SLUG}/ holds all artifacts for that outcome and model.",
            "shared_vif": "shared_train_split_vif_before_xgb/ holds VIF tables (same feature set for all outcomes).",
            "shared_splits": "shared_train_test_split_repo_assignment.csv when --group-by-repo.",
        },
        "hyperparameter_policy": hyperparameter_policy,
        "note": "SHAP is computed separately via rq3/run_model_shap.py using models saved here.",
        "vif_protocol": {
            "reference": "task_level_model_comparison.select_features_by_vif (statsmodels VIF, iterative drop)",
            "threshold": args.vif_threshold,
            "output_subdirectory": "shared_train_split_vif_before_xgb",
        },
        "calibration_protocol": calibration_protocol,
        "roc_protocol": roc_protocol,
        "evaluation_notes": evaluation_notes,
        "package_versions": _package_versions(),
        "cli_args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "targets_run": targets_list,
        "output_root": str(out_root.resolve()),
        "summary_json_file": "run_multitarget_xgb_summary.json",
        "summary_note": summary_note,
    }
    (out_root / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(f"\nDone. Outputs under {out_root}")



if __name__ == "__main__":
    raise SystemExit(main())
