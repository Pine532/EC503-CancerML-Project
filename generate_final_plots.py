from __future__ import annotations

import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_DIR / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESULTS_PATH = PROJECT_DIR / "results" / "all_results_combined.csv"
FIGURES_DIR = PROJECT_DIR / "figures"

CORE_MODELS = ["Dummy Mean", "Linear", "Ridge", "Gradient Boosting", "Neural Network"]


def load_results() -> pd.DataFrame:
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(f"Missing combined results file: {RESULTS_PATH}")

    df = pd.read_csv(RESULTS_PATH)
    for col in ["RMSE", "MAE", "R2"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def compact_label(row: pd.Series) -> str:
    mode = str(row["Dataset Mode"]).replace("secondary_screen_", "sec_").replace("gdsc_", "gdsc_")
    split = str(row["Split"])
    model = str(row["Model"])
    return f"{mode}\n{split}\n{model}"


def plot_metric_bars(df: pd.DataFrame, metric: str, output_path: Path) -> None:
    plot_df = df[df["Model"].isin(CORE_MODELS)].dropna(subset=[metric]).copy()
    plot_df = plot_df.sort_values(["Dataset Mode", "Split", "Model"])
    labels = [compact_label(row) for _, row in plot_df.iterrows()]

    width = max(12, len(plot_df) * 0.45)
    fig, ax = plt.subplots(figsize=(width, 7))
    ax.bar(np.arange(len(plot_df)), plot_df[metric].to_numpy())
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} by Model, Dataset, and Split")
    ax.set_xticks(np.arange(len(plot_df)))
    ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_gdsc_metadata_vs_expression(df: pd.DataFrame, output_path: Path) -> None:
    plot_df = df[
        (df["Dataset Mode"].isin(["gdsc_metadata_only", "gdsc_metadata_plus_expression"]))
        & (df["Split"] == "random")
        & (df["Model"].isin(["Linear", "Ridge"]))
    ].copy()
    plot_df = plot_df.dropna(subset=["R2"])

    pivot = plot_df.pivot_table(
        index="Model",
        columns="Dataset Mode",
        values="R2",
        aggfunc="max",
    )
    pivot = pivot.reindex(["Linear", "Ridge"])

    fig, ax = plt.subplots(figsize=(8, 5))
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel("R2")
    ax.set_title("GDSC Metadata Only vs Metadata + Expression")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Dataset Mode")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_secondary_auc_vs_ic50(df: pd.DataFrame, output_path: Path) -> None:
    plot_df = df[
        (df["Dataset Mode"].isin(["secondary_screen_auc", "secondary_screen_ic50"]))
        & (df["Model"].isin(["Linear", "Ridge", "Gradient Boosting"]))
    ].copy()
    plot_df = plot_df.dropna(subset=["R2"])

    labels = [
        f"{row['Dataset Mode'].replace('secondary_screen_', '')}\n{row['Split']}\n{row['Model']}"
        for _, row in plot_df.sort_values(["Dataset Mode", "Split", "Model"]).iterrows()
    ]
    values = plot_df.sort_values(["Dataset Mode", "Split", "Model"])["R2"].to_numpy()

    width = max(10, len(values) * 0.55)
    fig, ax = plt.subplots(figsize=(width, 6))
    ax.bar(np.arange(len(values)), values)
    ax.set_ylabel("R2")
    ax.set_title("Secondary Screen AUC vs Log IC50")
    ax.set_xticks(np.arange(len(values)))
    ax.set_xticklabels(labels, rotation=70, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    df = load_results()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    outputs = {
        "r2": FIGURES_DIR / "r2_by_model_dataset_split.png",
        "rmse": FIGURES_DIR / "rmse_by_model_dataset_split.png",
        "gdsc": FIGURES_DIR / "gdsc_metadata_vs_expression.png",
        "secondary": FIGURES_DIR / "secondary_auc_vs_ic50.png",
    }

    plot_metric_bars(df, "R2", outputs["r2"])
    plot_metric_bars(df, "RMSE", outputs["rmse"])
    plot_gdsc_metadata_vs_expression(df, outputs["gdsc"])
    plot_secondary_auc_vs_ic50(df, outputs["secondary"])

    print("Saved figures:")
    for path in outputs.values():
        print(path)


if __name__ == "__main__":
    main()
