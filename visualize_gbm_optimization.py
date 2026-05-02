from __future__ import annotations

import json
import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_DIR / ".matplotlib_cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_DIR / ".cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESULTS_DIR = PROJECT_DIR / "results"
FIGURES_DIR = PROJECT_DIR / "figures"
SUMMARY_PATH = RESULTS_DIR / "all_results_with_optimized.csv"

TARGET_DATASET_LABELS = {
    "AUC": {
        "gdsc_auc_metadata_only": "Metadata only",
        "gdsc_auc_metadata_plus_expression": "Metadata + expression",
    },
    "LN_IC50": {
        "gdsc_metadata_only": "Metadata only",
        "gdsc_metadata_plus_expression": "Metadata + expression",
    },
}

AUC_DATASET_LABELS = {
    "gdsc_auc_metadata_only": "Metadata only",
    "gdsc_auc_metadata_plus_expression": "Metadata + expression",
}

TRACE_PARAM_COLUMNS = [
    "param_model__n_estimators",
    "param_model__learning_rate",
    "param_model__max_depth",
    "param_model__subsample",
    "param_model__colsample_bytree",
    "param_model__reg_lambda",
]


def load_results() -> pd.DataFrame:
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(
            f"Missing {SUMMARY_PATH}. Run `python docs/build_results_summary.py` first."
        )
    return pd.read_csv(SUMMARY_PATH)


def load_best_params() -> pd.DataFrame:
    frames = []
    for path in sorted(RESULTS_DIR.glob("optimized_model_comparison_gdsc_auc_*_best_params.csv")):
        df = pd.read_csv(path)
        df["Source File"] = path.name
        frames.append(df)
    if not frames:
        raise FileNotFoundError("No optimized GDSC AUC best-parameter files found.")
    return pd.concat(frames, ignore_index=True)


def save_figure(fig: plt.Figure, stem: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        fig.savefig(FIGURES_DIR / f"{stem}.{suffix}", bbox_inches="tight", dpi=200)
    plt.close(fig)


def auc_gbm_results(results: pd.DataFrame) -> pd.DataFrame:
    df = results[
        (results["Dataset Mode"].isin(AUC_DATASET_LABELS))
        & (results["Target"] == "AUC")
        & (results["Split"] == "random")
        & (results["Model"] == "Gradient Boosting")
    ].copy()
    if df.empty:
        raise ValueError("No GDSC AUC Gradient Boosting rows found.")
    df["Dataset Label"] = df["Dataset Mode"].map(AUC_DATASET_LABELS)
    return df


def plot_rmse_comparison(results: pd.DataFrame) -> None:
    gbm = auc_gbm_results(results)
    labels = list(AUC_DATASET_LABELS.values())
    x = np.arange(len(labels))
    width = 0.26

    baseline_rmse = []
    optimized_val_rmse = []
    optimized_test_rmse = []
    for dataset_mode in AUC_DATASET_LABELS:
        base = gbm[
            (gbm["Dataset Mode"] == dataset_mode)
            & (gbm["Experiment Type"] == "Baseline")
        ]
        opt = gbm[
            (gbm["Dataset Mode"] == dataset_mode)
            & (gbm["Experiment Type"] == "Optimized")
        ]
        baseline_rmse.append(float(base["RMSE"].iloc[0]) if not base.empty else np.nan)
        optimized_val_rmse.append(float(opt["RMSE"].iloc[0]) if not opt.empty else np.nan)
        optimized_test_rmse.append(float(opt["Test RMSE"].iloc[0]) if not opt.empty else np.nan)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(x - width, baseline_rmse, width, label="Baseline GBM validation", color="#9aa6b2")
    ax.bar(x, optimized_val_rmse, width, label="Optimized GBM validation", color="#326273")
    ax.bar(x + width, optimized_test_rmse, width, label="Optimized GBM test", color="#e39774")

    ax.set_title("GDSC AUC Gradient Boosting Optimization: RMSE", fontsize=14, weight="bold")
    ax.set_ylabel("RMSE, lower is better")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, max(v for v in [*baseline_rmse, *optimized_val_rmse, *optimized_test_rmse] if not np.isnan(v)) * 1.18)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)

    for container in ax.containers:
        ax.bar_label(container, fmt="%.4f", fontsize=9, padding=3)

    save_figure(fig, "gbm_auc_rmse_optimization")


def plot_rmse_line_comparison(results: pd.DataFrame) -> None:
    gbm = auc_gbm_results(results)
    stages = ["Baseline\nvalidation", "Optimized\nvalidation", "Optimized\ntest"]
    x = np.arange(len(stages))

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for dataset_mode, label in AUC_DATASET_LABELS.items():
        base = gbm[
            (gbm["Dataset Mode"] == dataset_mode)
            & (gbm["Experiment Type"] == "Baseline")
        ]
        opt = gbm[
            (gbm["Dataset Mode"] == dataset_mode)
            & (gbm["Experiment Type"] == "Optimized")
        ]
        values = [
            float(base["RMSE"].iloc[0]) if not base.empty else np.nan,
            float(opt["RMSE"].iloc[0]) if not opt.empty else np.nan,
            float(opt["Test RMSE"].iloc[0]) if not opt.empty else np.nan,
        ]
        ax.plot(x, values, marker="o", linewidth=2.5, markersize=7, label=label)
        for xi, value in zip(x, values):
            ax.text(xi, value, f"{value:.4f}", ha="center", va="bottom", fontsize=9)

    ax.set_title("GDSC AUC Gradient Boosting RMSE Through Optimization", fontsize=14, weight="bold")
    ax.set_ylabel("RMSE, lower is better")
    ax.set_xticks(x)
    ax.set_xticklabels(stages)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    save_figure(fig, "gbm_auc_rmse_optimization_line")


def plot_r2_comparison(results: pd.DataFrame) -> None:
    gbm = auc_gbm_results(results)
    labels = list(AUC_DATASET_LABELS.values())
    x = np.arange(len(labels))
    width = 0.26

    baseline_r2 = []
    optimized_val_r2 = []
    optimized_test_r2 = []
    for dataset_mode in AUC_DATASET_LABELS:
        base = gbm[
            (gbm["Dataset Mode"] == dataset_mode)
            & (gbm["Experiment Type"] == "Baseline")
        ]
        opt = gbm[
            (gbm["Dataset Mode"] == dataset_mode)
            & (gbm["Experiment Type"] == "Optimized")
        ]
        baseline_r2.append(float(base["R2"].iloc[0]) if not base.empty else np.nan)
        optimized_val_r2.append(float(opt["R2"].iloc[0]) if not opt.empty else np.nan)
        optimized_test_r2.append(float(opt["Test R2"].iloc[0]) if not opt.empty else np.nan)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(x - width, baseline_r2, width, label="Baseline GBM validation", color="#9aa6b2")
    ax.bar(x, optimized_val_r2, width, label="Optimized GBM validation", color="#326273")
    ax.bar(x + width, optimized_test_r2, width, label="Optimized GBM test", color="#e39774")

    ax.set_title("GDSC AUC Gradient Boosting Optimization: R²", fontsize=14, weight="bold")
    ax.set_ylabel("R², higher is better")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, max(v for v in [*baseline_r2, *optimized_val_r2, *optimized_test_r2] if not np.isnan(v)) * 1.18)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)

    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=9, padding=3)

    save_figure(fig, "gbm_auc_r2_optimization")


def plot_r2_line_comparison(results: pd.DataFrame) -> None:
    gbm = auc_gbm_results(results)
    stages = ["Baseline\nvalidation", "Optimized\nvalidation", "Optimized\ntest"]
    x = np.arange(len(stages))

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for dataset_mode, label in AUC_DATASET_LABELS.items():
        base = gbm[
            (gbm["Dataset Mode"] == dataset_mode)
            & (gbm["Experiment Type"] == "Baseline")
        ]
        opt = gbm[
            (gbm["Dataset Mode"] == dataset_mode)
            & (gbm["Experiment Type"] == "Optimized")
        ]
        values = [
            float(base["R2"].iloc[0]) if not base.empty else np.nan,
            float(opt["R2"].iloc[0]) if not opt.empty else np.nan,
            float(opt["Test R2"].iloc[0]) if not opt.empty else np.nan,
        ]
        ax.plot(x, values, marker="o", linewidth=2.5, markersize=7, label=label)
        for xi, value in zip(x, values):
            ax.text(xi, value, f"{value:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_title("GDSC AUC Gradient Boosting R² Through Optimization", fontsize=14, weight="bold")
    ax.set_ylabel("R², higher is better")
    ax.set_xticks(x)
    ax.set_xticklabels(stages)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    save_figure(fig, "gbm_auc_r2_optimization_line")


def plot_hyperparameters(params_df: pd.DataFrame) -> None:
    gbm = params_df[
        (params_df["Model"] == "Gradient Boosting")
        & (params_df["Dataset Mode"].isin(AUC_DATASET_LABELS))
    ].copy()
    if gbm.empty:
        raise ValueError("No optimized Gradient Boosting parameter rows found.")

    records = []
    for _, row in gbm.iterrows():
        params = json.loads(row["Best Params JSON"])
        records.append({
            "Dataset": AUC_DATASET_LABELS[row["Dataset Mode"]],
            "n_estimators": params["model__n_estimators"],
            "learning_rate": params["model__learning_rate"],
            "max_depth": params["model__max_depth"],
            "subsample": params["model__subsample"],
            "colsample_bytree": params["model__colsample_bytree"],
            "reg_lambda": params["model__reg_lambda"],
        })
    param_table = pd.DataFrame(records).set_index("Dataset")

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes = axes.ravel()
    colors = ["#326273", "#e39774"]
    for ax, col in zip(axes, param_table.columns):
        values = param_table[col].astype(float)
        ax.bar(values.index, values.values, color=colors[: len(values)])
        ax.set_title(col)
        ax.grid(axis="y", alpha=0.25)
        for i, value in enumerate(values.values):
            ax.text(i, value, f"{value:g}", ha="center", va="bottom", fontsize=9)
    fig.suptitle("Selected Optimized XGBoost Hyperparameters", fontsize=14, weight="bold")
    fig.tight_layout()
    save_figure(fig, "gbm_auc_best_hyperparameters")


def dataset_mode_from_cv_path(path: Path) -> tuple[str, str, str] | None:
    for target, dataset_labels in TARGET_DATASET_LABELS.items():
        for dataset_mode, feature_label in dataset_labels.items():
            marker = f"optimized_model_comparison_{dataset_mode}_random_gradient_boosting_"
            if path.name.startswith(marker):
                return target, dataset_mode, feature_label
    return None


def load_gbm_cv_trials() -> pd.DataFrame:
    rows = []
    cv_files = sorted(RESULTS_DIR.glob("*gradient_boosting_cv_results.csv"))
    for path in cv_files:
        parsed = dataset_mode_from_cv_path(path)
        if parsed is None:
            continue

        target, dataset_mode, feature_label = parsed
        df = pd.read_csv(path).copy()
        if "mean_test_score" not in df:
            continue

        for index, row in df.reset_index(drop=True).iterrows():
            trial = index + 1
            trial_row = {
                "Target": target,
                "Dataset Mode": dataset_mode,
                "Feature Set": feature_label,
                "Trial": trial,
                "CV RMSE": -float(row["mean_test_score"]),
                "CV RMSE Std": float(row.get("std_test_score", np.nan)),
                "Rank": int(row.get("rank_test_score", trial)),
                "Source File": path.name,
            }
            for col in TRACE_PARAM_COLUMNS:
                trial_row[col.replace("param_model__", "")] = row[col]
            rows.append(trial_row)

    if not rows:
        return pd.DataFrame()

    trials = pd.DataFrame(rows)
    trials["Best So Far CV RMSE"] = trials.groupby(["Target", "Dataset Mode"])["CV RMSE"].cummin()
    trials.to_csv(RESULTS_DIR / "gbm_random_search_trials.csv", index=False)
    return trials


def plot_random_search_rmse_traces(trials: pd.DataFrame) -> None:
    for target, target_trials in trials.groupby("Target", sort=False):
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for feature_label, feature_trials in target_trials.groupby("Feature Set", sort=False):
            feature_trials = feature_trials.sort_values("Trial")
            ax.plot(
                feature_trials["Trial"],
                feature_trials["CV RMSE"],
                marker="o",
                alpha=0.35,
                linewidth=1.4,
                label=f"{feature_label}: tried candidate",
            )
            ax.plot(
                feature_trials["Trial"],
                feature_trials["Best So Far CV RMSE"],
                marker="o",
                linewidth=2.5,
                label=f"{feature_label}: best so far",
            )

        ax.set_title(f"{target} GBM Random Search Trace", fontsize=14, weight="bold")
        ax.set_xlabel("Random-search candidate tried")
        ax.set_ylabel("Cross-validation RMSE, lower is better")
        ax.set_xticks(sorted(target_trials["Trial"].unique()))
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
        target_slug = "lnic50" if target == "LN_IC50" else target.lower()
        save_figure(fig, f"gbm_{target_slug}_random_search_cv_trace")


def plot_random_search_parameter_traces(trials: pd.DataFrame) -> None:
    params = [
        "n_estimators",
        "learning_rate",
        "max_depth",
        "subsample",
        "colsample_bytree",
        "reg_lambda",
    ]
    for target, target_trials in trials.groupby("Target", sort=False):
        fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), sharex=True)
        axes = axes.ravel()
        for ax, param in zip(axes, params):
            for feature_label, feature_trials in target_trials.groupby("Feature Set", sort=False):
                feature_trials = feature_trials.sort_values("Trial")
                ax.plot(
                    feature_trials["Trial"],
                    pd.to_numeric(feature_trials[param], errors="coerce"),
                    marker="o",
                    linewidth=1.8,
                    label=feature_label,
                )
            ax.set_title(param)
            ax.grid(alpha=0.25)
        for ax in axes[-3:]:
            ax.set_xlabel("Random-search candidate tried")
        axes[0].legend(frameon=False, fontsize=8)
        fig.suptitle(f"{target} GBM Hyperparameter Values Tried", fontsize=14, weight="bold")
        fig.tight_layout()
        target_slug = "lnic50" if target == "LN_IC50" else target.lower()
        save_figure(fig, f"gbm_{target_slug}_random_search_parameter_trace")


def main() -> None:
    results = load_results()
    params = load_best_params()
    trials = load_gbm_cv_trials()
    plot_rmse_comparison(results)
    plot_rmse_line_comparison(results)
    plot_r2_comparison(results)
    plot_r2_line_comparison(results)
    plot_hyperparameters(params)
    if not trials.empty:
        plot_random_search_rmse_traces(trials)
        plot_random_search_parameter_traces(trials)
    print(f"Wrote GBM optimization figures to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
