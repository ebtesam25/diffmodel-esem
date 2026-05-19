#!/usr/bin/env python3
"""
Task-level model comparison.

Each task has multiple runs sharing identical prompt/repo/patch features.
We deduplicate to one row per task and predict three targets:
  - pass_rate   : fraction of runs that succeeded (continuous, 0–1) → regression
  - any_success : did at least one run succeed? (binary pass@k)      → classification
  - maj_success : did the majority of runs succeed? (≥50% pass rate) → classification
"""

from pathlib import Path
import sys
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

REPLICATION_ROOT = Path(__file__).resolve().parents[1]
_LIB = REPLICATION_ROOT / "lib"
if str(_LIB) not in sys.path:
    sys.path.insert(0, str(_LIB))

from replication.data import load_task_level_data as _load_task_level_data
from replication.data import train_test_indices_from_targets
from replication.paths import results_dir_for_rq


def load_task_level_data():
    """Load published task-level features and outcomes from ``data/task_level_dataset.parquet``."""
    X, targets = _load_task_level_data()
    outcome_cols = ["pass_rate", "any_success", "maj_success", "n_runs"]
    return X, targets[outcome_cols]


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def clf_metrics(y_true, prob, threshold=0.5):
    pred = (prob >= threshold).astype(int)
    return {
        "auc":    roc_auc_score(y_true, prob),
        "pr_auc": average_precision_score(y_true, prob),
        "brier":  brier_score_loss(y_true, prob),
        "f1":     f1_score(y_true, pred, zero_division=0),
        "mcc":    matthews_corrcoef(y_true, pred),
    }


def reg_metrics(y_true, y_pred):
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae":  float(mean_absolute_error(y_true, y_pred)),
        "r2":   float(r2_score(y_true, y_pred)),
    }


# ---------------------------------------------------------------------------
# VIF-based feature selection
# ---------------------------------------------------------------------------

def compute_vif(X):
    """Compute VIF for all features. Returns DataFrame sorted by VIF descending."""
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    Xs = pd.DataFrame(
        StandardScaler().fit_transform(X),
        columns=X.columns,
    )
    return pd.DataFrame({
        "feature": Xs.columns,
        "VIF": [variance_inflation_factor(Xs.values, i) for i in range(Xs.shape[1])],
    }).sort_values("VIF", ascending=False).reset_index(drop=True)


def select_features_by_vif(X, threshold=10.0, output_dir=None):
    """
    Iteratively drop the highest-VIF feature until all VIF < threshold.

    Standard iterative procedure:
      1. Compute VIF for all remaining features
      2. If max VIF > threshold, drop that feature
      3. Repeat until all VIF <= threshold

    Parameters
    ----------
    X          : pd.DataFrame  — task-level feature matrix
    threshold  : float         — VIF cutoff (10 = standard, 5 = conservative)
    output_dir : Path or None  — if set, saves CSV tables

    Returns
    -------
    X_reduced : pd.DataFrame  — features with multicollinearity removed
    vif_final : pd.DataFrame  — VIF table for retained features
    dropped   : list[str]     — names of dropped features
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    print(f"\n{'='*60}")
    print(f"VIF-BASED FEATURE SELECTION (threshold={threshold})")
    print(f"{'='*60}")
    print(f"Starting features: {X.shape[1]}")

    # Full VIF table before selection
    vif_all = compute_vif(X)
    print(f"\nTop 15 VIF scores (before selection):")
    print(vif_all.head(15).to_string(index=False))
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        vif_all.to_csv(output_dir / "vif_all_features.csv", index=False)

    # Iterative elimination
    remaining = list(X.columns)
    dropped = []
    iteration = 0

    while True:
        iteration += 1
        Xs = pd.DataFrame(
            StandardScaler().fit_transform(X[remaining]),
            columns=remaining,
        )
        vifs = np.array([
            variance_inflation_factor(Xs.values, i)
            for i in range(len(remaining))
        ])
        max_vif = vifs.max()
        max_feat = remaining[int(np.argmax(vifs))]

        if max_vif <= threshold:
            print(f"\nConverged after {iteration-1} iterations. "
                  f"Max VIF={max_vif:.2f} <= {threshold}")
            break

        print(f"  Iter {iteration:3d}: dropping '{max_feat}' (VIF={max_vif:.2f})")
        dropped.append(max_feat)
        remaining.remove(max_feat)

    X_reduced = X[remaining].copy()
    vif_final = compute_vif(X_reduced)

    print(f"\nRetained: {len(remaining)} features  |  Dropped: {len(dropped)}")
    print(f"\nFinal VIF table:")
    print(vif_final.to_string(index=False))
    if dropped:
        print(f"\nDropped: {dropped}")

    if output_dir is not None:
        vif_final.to_csv(output_dir / "vif_selected_features.csv", index=False)
        pd.DataFrame({"dropped_feature": dropped}).to_csv(
            output_dir / "vif_dropped_features.csv", index=False
        )
        print(f"VIF tables saved to {output_dir}")

    return X_reduced, vif_final, dropped


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def run_classification(X_train, X_test, y_train, y_test, target_name):
    """Run LR, RF, and XGBoost classifiers. Returns dict of test metrics."""
    results = {}

    # Logistic Regression
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(X_train)
    Xte_s = scaler.transform(X_test)
    lr = LogisticRegression(penalty="l2", solver="lbfgs", max_iter=2000, random_state=42)
    lr.fit(Xtr_s, y_train)
    prob = lr.predict_proba(Xte_s)[:, 1]
    results["Logistic"] = clf_metrics(y_test, prob)

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=None, max_features="sqrt",
        min_samples_leaf=3, random_state=42, n_jobs=4,
    )
    rf.fit(X_train, y_train)
    prob = rf.predict_proba(X_test)[:, 1]
    results["RandomForest"] = clf_metrics(y_test, prob)
    rf_model = rf  # save for feature importance

    # XGBoost
    try:
        import xgboost as xgb
        clf = xgb.XGBClassifier(
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
        clf.fit(X_train, y_train)
        prob = clf.predict_proba(X_test)[:, 1]
        results["XGBoost"] = clf_metrics(y_test, prob)
    except ImportError:
        print("  XGBoost not installed; skipping.")

    print(f"\n[{target_name}] Classification results (test):")
    for name, m in results.items():
        print(f"  {name:15s}  AUC={m['auc']:.4f}  PR-AUC={m['pr_auc']:.4f}  "
              f"Brier={m['brier']:.4f}  F1={m['f1']:.4f}")

    return results, rf_model


def run_regression(X_train, X_test, y_train, y_test, target_name):
    """Run Ridge and RF regression for pass_rate target."""
    results = {}

    # Ridge
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(X_train)
    Xte_s = scaler.transform(X_test)
    ridge = Ridge(alpha=1.0, random_state=42)
    ridge.fit(Xtr_s, y_train)
    pred = np.clip(ridge.predict(Xte_s), 0, 1)
    results["Ridge"] = reg_metrics(y_test, pred)

    # Random Forest
    rf = RandomForestRegressor(
        n_estimators=300, max_depth=None, max_features="sqrt",
        min_samples_leaf=3, random_state=42, n_jobs=4,
    )
    rf.fit(X_train, y_train)
    pred = np.clip(rf.predict(X_test), 0, 1)
    results["RandomForest"] = reg_metrics(y_test, pred)

    # XGBoost
    try:
        import xgboost as xgb
        reg = xgb.XGBRegressor(
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
        reg.fit(X_train, y_train)
        pred = np.clip(reg.predict(X_test), 0, 1)
        results["XGBoost"] = reg_metrics(y_test, pred)
    except ImportError:
        pass

    print(f"\n[{target_name}] Regression results (test):")
    for name, m in results.items():
        print(f"  {name:15s}  RMSE={m['rmse']:.4f}  MAE={m['mae']:.4f}  R²={m['r2']:.4f}")

    return results


# ---------------------------------------------------------------------------
# Cross-validation (task-level, no leakage)
# ---------------------------------------------------------------------------

def cross_validate_task_level(X, y_clf, y_reg, n_splits=10):
    """
    GroupKFold CV across tasks. Since each row IS a task here,
    this is just standard KFold — but we keep GroupKFold for consistency
    in case task_id grouping is ever needed upstream.
    """
    from sklearn.model_selection import KFold
    import xgboost as xgb

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    clf_aucs, reg_r2s = [], []
    # store per-model per-fold scores for statistical comparison
    clf_scores = {"Logistic": [], "RandomForest": [], "XGBoost": []}

    for fold, (tr, va) in enumerate(kf.split(X)):
        Xtr, Xva = X.iloc[tr], X.iloc[va]
        ytr_c, yva_c = y_clf[tr], y_clf[va]
        ytr_r, yva_r = y_reg[tr], y_reg[va]

        clf = xgb.XGBClassifier(
            objective="binary:logistic", n_estimators=300, max_depth=4,
            eta=0.05, subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=4, tree_method="hist", verbosity=0,
        )
        clf.fit(Xtr, ytr_c)
        prob = clf.predict_proba(Xva)[:, 1]
        auc_xgb = roc_auc_score(yva_c, prob)
        clf_aucs.append(auc_xgb)
        clf_scores["XGBoost"].append(auc_xgb)

        # Logistic (simple baseline for CV comparisons)
        from sklearn.linear_model import LogisticRegression
        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(Xtr)
        Xva_s = scaler.transform(Xva)
        lr = LogisticRegression(penalty="l2", solver="lbfgs", max_iter=2000, random_state=42)
        lr.fit(Xtr_s, ytr_c)
        prob_lr = lr.predict_proba(Xva_s)[:, 1]
        clf_scores["Logistic"].append(roc_auc_score(yva_c, prob_lr))

        # Random Forest baseline
        from sklearn.ensemble import RandomForestClassifier
        rf = RandomForestClassifier(n_estimators=300, max_depth=None, max_features="sqrt", min_samples_leaf=3, random_state=42)
        rf.fit(Xtr, ytr_c)
        prob_rf = rf.predict_proba(Xva)[:, 1]
        clf_scores["RandomForest"].append(roc_auc_score(yva_c, prob_rf))

        reg = xgb.XGBRegressor(
            objective="reg:squarederror", n_estimators=300, max_depth=4,
            eta=0.05, subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=4, tree_method="hist", verbosity=0,
        )
        reg.fit(Xtr, ytr_r)
        pred = np.clip(reg.predict(Xva), 0, 1)
        reg_r2s.append(r2_score(yva_r, pred))

    print(f"\n[{n_splits}-fold CV] XGBoost any_success AUC:")
    print(f"  Fold-level AUCs: {[round(x, 4) for x in clf_aucs]}")
    print(f"  Mean ± SD: {np.mean(clf_aucs):.4f} ± {np.std(clf_aucs):.4f}")
    print(f"[{n_splits}-fold CV] XGBoost pass_rate R²:")
    print(f"  Fold-level R²: {[round(x, 4) for x in reg_r2s]}")
    print(f"  Mean ± SD: {np.mean(reg_r2s):.4f} ± {np.std(reg_r2s):.4f}")

    return {
            "cv_any_success_auc_mean": float(np.mean(clf_aucs)),
            "cv_any_success_auc_std":  float(np.std(clf_aucs)),
            "cv_pass_rate_r2_mean":    float(np.mean(reg_r2s)),
            "cv_pass_rate_r2_std":     float(np.std(reg_r2s)),
            "cv_clf_scores_per_model": clf_scores,
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def generate_plots(X_test, y_any, y_maj, y_rate,
                   rf_any_model, clf_results, reg_results, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. AUC comparison across targets and models
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (target, results) in zip(axes, [
        ("any_success", clf_results["any_success"]),
        ("maj_success", clf_results["maj_success"]),
    ]):
        names = list(results.keys())
        aucs  = [results[n]["auc"] for n in names]
        colors = ["#4C72B0", "#DD8452", "#55A868"]
        ax.bar(names, aucs, color=colors[:len(names)])
        ax.set_ylim(0.5, 1.0)
        ax.set_ylabel("Test AUC")
        ax.set_title(f"Target: {target}")
        for i, v in enumerate(aucs):
            ax.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=9)

    fig.suptitle("Task-Level Model Comparison (AUC)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "task_clf_auc_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 2. pass_rate regression: RMSE + R²
    fig, ax = plt.subplots(figsize=(8, 5))
    names  = list(reg_results.keys())
    rmses  = [reg_results[n]["rmse"] for n in names]
    r2s    = [reg_results[n]["r2"]   for n in names]
    x = np.arange(len(names))
    ax.bar(x - 0.2, rmses, width=0.35, label="RMSE", color="#4C72B0")
    ax2 = ax.twinx()
    ax2.bar(x + 0.2, r2s, width=0.35, label="R²", color="#DD8452", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("RMSE")
    ax2.set_ylabel("R²")
    ax.set_title("Task-Level pass_rate Regression")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    plt.tight_layout()
    plt.savefig(output_dir / "task_reg_pass_rate.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 3. RF feature importances (any_success)
    if rf_any_model is not None:
        importances = pd.Series(rf_any_model.feature_importances_, index=X_test.columns)
        top20 = importances.nlargest(20).iloc[::-1]
        fig, ax = plt.subplots(figsize=(10, 8))
        top20.plot(kind="barh", ax=ax, color="#4C72B0")
        ax.set_xlabel("Importance")
        ax.set_title("RF Feature Importance (any_success target)")
        plt.tight_layout()
        plt.savefig(output_dir / "task_rf_feature_importance.png", dpi=300, bbox_inches="tight")
        plt.close()

    # 4. pass_rate distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(y_rate, bins=20, color="#4C72B0", edgecolor="white")
    ax.set_xlabel("pass_rate (fraction of runs succeeded)")
    ax.set_ylabel("Number of tasks")
    ax.set_title("Distribution of Task pass_rate")
    plt.tight_layout()
    plt.savefig(output_dir / "task_pass_rate_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Plots saved to {output_dir}")


# ---------------------------------------------------------------------------
# Component ablation
# ---------------------------------------------------------------------------

def run_component_ablation(X_train, X_test, y_any_train, y_any_test,
                           y_rate_train, y_rate_test, output_dir):
    """
    Train XGBoost separately on each feature group and all combinations.
    Reports AUC (any_success) and R2 (pass_rate) for each subset.
    This shows how much each component contributes independently.
    """
    import xgboost as xgb

    groups = {
        "patch": [c for c in X_train.columns if c.startswith("patch_")],
        "repo":  [c for c in X_train.columns if c.startswith("repo_")],
        "prompt":[c for c in X_train.columns if c.startswith("prompt_")],
    }
    combos = {
        "patch":             groups["patch"],
        "repo":              groups["repo"],
        "prompt":            groups["prompt"],
        "patch+repo":        groups["patch"] + groups["repo"],
        "patch+prompt":      groups["patch"] + groups["prompt"],
        "repo+prompt":       groups["repo"]  + groups["prompt"],
        "patch+repo+prompt": groups["patch"] + groups["repo"] + groups["prompt"],
    }

    print("\n" + "=" * 60)
    print("COMPONENT ABLATION (XGBoost)")
    print("=" * 60)
    print(f"  patch  features: {len(groups['patch'])}")
    print(f"  repo   features: {len(groups['repo'])}")
    print(f"  prompt features: {len(groups['prompt'])}")

    results = []
    for name, cols in combos.items():
        if not cols:
            continue
        Xtr = X_train[cols]
        Xte = X_test[cols]

        # Classification
        clf = xgb.XGBClassifier(
            objective="binary:logistic", n_estimators=300, max_depth=4,
            eta=0.05, subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=4, tree_method="hist", verbosity=0,
        )
        clf.fit(Xtr, y_any_train)
        prob = clf.predict_proba(Xte)[:, 1]
        auc = roc_auc_score(y_any_test, prob)

        # Regression
        reg = xgb.XGBRegressor(
            objective="reg:squarederror", n_estimators=300, max_depth=4,
            eta=0.05, subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=4, tree_method="hist", verbosity=0,
        )
        reg.fit(Xtr, y_rate_train)
        pred = np.clip(reg.predict(Xte), 0, 1)
        r2 = r2_score(y_rate_test, pred)

        results.append({
            "components": name,
            "n_features": len(cols),
            "any_success_auc": round(auc, 4),
            "pass_rate_r2":    round(r2, 4),
        })
        print(f"  {name:25s}  n={len(cols):2d}  AUC={auc:.4f}  R²={r2:.4f}")

    df = pd.DataFrame(results)
    df.to_csv(output_dir / "component_ablation.csv", index=False)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ["#4C72B0" if "+" not in r else "#DD8452" if r != "patch+repo+prompt" else "#55A868"
              for r in df["components"]]

    axes[0].barh(df["components"], df["any_success_auc"], color=colors)
    axes[0].set_xlabel("AUC (any_success)")
    axes[0].set_title("Component Ablation — Classification")
    axes[0].set_xlim(0.5, 1.0)
    for i, v in enumerate(df["any_success_auc"]):
        axes[0].text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=8)

    axes[1].barh(df["components"], df["pass_rate_r2"], color=colors)
    axes[1].set_xlabel("R² (pass_rate)")
    axes[1].set_title("Component Ablation — Regression")
    for i, v in enumerate(df["pass_rate_r2"]):
        axes[1].text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=8)

    fig.suptitle("Feature Component Ablation", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "component_ablation.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved component_ablation.csv and .png")

    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    output_dir = results_dir_for_rq("rq1") / "model_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Loading and aggregating to task level...")
    print("=" * 60)
    X, full_targets = _load_task_level_data()
    targets = full_targets[["pass_rate", "any_success", "maj_success", "n_runs"]]

    y_rate = targets["pass_rate"].to_numpy(dtype=float)
    y_any  = targets["any_success"].to_numpy(dtype=int)
    y_maj  = targets["maj_success"].to_numpy(dtype=int)

    # ----------------------------------------------------------------
    # Step 1: Train/test split FIRST — before any feature selection.
    # All selection decisions are made on train only to avoid leakage.
    # ----------------------------------------------------------------
    idx_train, idx_test = train_test_indices_from_targets(full_targets, y_any)
    X_train, X_test = X.iloc[idx_train], X.iloc[idx_test]
    y_rate_train, y_rate_test = y_rate[idx_train], y_rate[idx_test]
    y_any_train,  y_any_test  = y_any[idx_train],  y_any[idx_test]
    y_maj_train,  y_maj_test  = y_maj[idx_train],  y_maj[idx_test]

    print(f"\nTrain: {len(X_train)} tasks, Test: {len(X_test)} tasks")

    # ----------------------------------------------------------------
    # Step 2: VIF on TRAIN only. Apply same column selection to test.
    # ----------------------------------------------------------------
    X_train_vif, vif_final, vif_dropped = select_features_by_vif(
        X_train, threshold=10.0, output_dir=output_dir
    )
    X_test_vif = X_test[X_train_vif.columns]
    print(f"\nFeatures after VIF selection: {X_train_vif.shape[1]}")

    # ----------------------------------------------------------------
    # Step 3: Fit XGBoost on train, compute SHAP on TRAIN,
    #         identify zero-importance features, drop them.
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 3: SHAP-based feature screening (train only)")
    print("=" * 60)
    try:
        import xgboost as xgb
        import shap as shap_lib

        screen_clf = xgb.XGBClassifier(
            objective="binary:logistic", n_estimators=300, max_depth=4,
            eta=0.05, subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=4, tree_method="hist", verbosity=0,
        )
        screen_clf.fit(X_train_vif, y_any_train)

        explainer = shap_lib.TreeExplainer(screen_clf)
        shap_train = explainer.shap_values(X_train_vif)
        if isinstance(shap_train, list):
            shap_train = shap_train[1] if len(shap_train) > 1 else shap_train[0]

        mean_abs_shap = np.abs(shap_train).mean(axis=0)
        shap_screen_df = pd.DataFrame({
            "feature": X_train_vif.columns,
            "mean_abs_shap": mean_abs_shap,
        }).sort_values("mean_abs_shap", ascending=False)
        shap_screen_df.to_csv(output_dir / "shap_screening_train.csv", index=False)

        # Drop features with zero SHAP on train
        zero_shap = shap_screen_df[shap_screen_df["mean_abs_shap"] == 0]["feature"].tolist()
        if zero_shap:
            print(f"  Dropping {len(zero_shap)} zero-SHAP features (identified on train): {zero_shap}")
        else:
            print("  No zero-SHAP features found on train.")

        X_train_final = X_train_vif.drop(columns=zero_shap, errors="ignore")
        X_test_final  = X_test_vif.drop(columns=zero_shap, errors="ignore")
        model_features = list(X_train_final.columns)
        X_train_final = X_train_final[model_features]
        X_test_final = X_test_final[model_features]
        print(f"  Final feature count: {X_train_final.shape[1]}")

    except ImportError:
        print("  XGBoost/SHAP not available; skipping SHAP screening.")
        X_train_final = X_train_vif.copy()
        X_test_final  = X_test_vif.copy()
        model_features = list(X_train_final.columns)

    # Hard guard: downstream explanation/statistical analyses must only use the
    # exact feature set that survived VIF + SHAP screening.
    X_train_final = X_train_final[model_features]
    X_test_final = X_test_final[model_features]

    # ----------------------------------------------------------------
    # Step 4: Fit all models on final feature set, evaluate on test.
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("CLASSIFICATION TARGETS")
    print("=" * 60)
    clf_results = {}
    rf_models   = {}

    any_clf_results, rf_any = run_classification(
        X_train_final, X_test_final, y_any_train, y_any_test, "any_success"
    )
    clf_results["any_success"] = any_clf_results
    rf_models["any_success"]   = rf_any

    maj_clf_results, rf_maj = run_classification(
        X_train_final, X_test_final, y_maj_train, y_maj_test, "maj_success"
    )
    clf_results["maj_success"] = maj_clf_results
    rf_models["maj_success"]   = rf_maj

    print("\n" + "=" * 60)
    print("REGRESSION TARGET: pass_rate")
    print("=" * 60)
    reg_results = run_regression(
        X_train_final, X_test_final, y_rate_train, y_rate_test, "pass_rate"
    )

    print("\n" + "=" * 60)
    print("CROSS-VALIDATION (10-fold, task level)")
    print("=" * 60)
    cv_results = cross_validate_task_level(
        X_train_final, y_any_train, y_rate_train, n_splits=10
    )

    # Majority-class baseline on test (train prevalence)
    try:
        maj_prob = np.full_like(y_any_test, fill_value=float(np.mean(y_any_train)), dtype=float)
        maj_metrics = clf_metrics(y_any_test, maj_prob)
        pd.DataFrame({"majority_baseline": maj_metrics}).to_csv(
            output_dir / "baseline_majority_metrics.csv"
        )
        print("\nSaved majority-class baseline metrics -> baseline_majority_metrics.csv")
    except Exception as e:
        print(f"Baseline computation failed: {e}")

    # ----------------------------------------------------------------
    # Step 6: Component ablation.
    # ----------------------------------------------------------------
    print("\nRunning component ablation...")
    ablation_df = run_component_ablation(
        X_train_final, X_test_final,
        y_any_train, y_any_test,
        y_rate_train, y_rate_test,
        output_dir,
    )

    # ----------------------------------------------------------------
    # Step 7: Plots + save.
    # ----------------------------------------------------------------
    print("\nGenerating plots...")
    generate_plots(
        X_test_final, y_any_test, y_maj_test, y_rate_test,
        rf_models.get("any_success"),
        clf_results, reg_results, output_dir,
    )

    all_results = {
        "n_features_after_vif":        int(X_train_vif.shape[1]),
        "n_features_final":            int(X_train_final.shape[1]),
        "vif_dropped":                 vif_dropped,
        "clf_any_success":             clf_results["any_success"],
        "clf_maj_success":             clf_results["maj_success"],
        "reg_pass_rate":               reg_results,
        "cross_validation":            cv_results,
        "component_ablation":          ablation_df.to_dict(orient="records"),
    }
    (output_dir / "task_level_results.json").write_text(json.dumps(all_results, indent=2))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Features: 60 raw → {X_train_vif.shape[1]} after VIF → {X_train_final.shape[1]} after SHAP screening")

    print("\nClassification — any_success (pass@k):")
    df_any = pd.DataFrame(clf_results["any_success"]).T
    print(df_any.to_string())

    print("\nClassification — maj_success (majority vote):")
    df_maj = pd.DataFrame(clf_results["maj_success"]).T
    print(df_maj.to_string())

    print("\nRegression — pass_rate:")
    df_reg = pd.DataFrame(reg_results).T
    print(df_reg.to_string())

    df_any.to_csv(output_dir / "task_clf_any_success.csv")
    df_maj.to_csv(output_dir / "task_clf_maj_success.csv")
    df_reg.to_csv(output_dir / "task_reg_pass_rate.csv")

    print(f"\nAll outputs saved to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())