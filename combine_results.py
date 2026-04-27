from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_DIR / "results"
OUTPUT_PATH = RESULTS_DIR / "all_results_combined.csv"

STANDARD_COLUMNS = [
    "Dataset",
    "Dataset Mode",
    "Target",
    "Split",
    "Feature Set",
    "Model",
    "RMSE",
    "MAE",
    "R2",
    "Train Seconds",
    "Predict Seconds",
    "Max Rows",
    "Target Clipping",
    "Clip Low Quantile",
    "Clip High Quantile",
    "Clip Low Value",
    "Clip High Value",
    "Neural Network Backend",
    "Source File",
]


def result_files() -> list[Path]:
    files = [
        *RESULTS_DIR.glob("model_comparison*.csv"),
        *RESULTS_DIR.glob("neural_network*.csv"),
    ]
    return sorted(path for path in files if path.name != OUTPUT_PATH.name)


def normalize_result_frame(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Source File"] = path.name

    if "Backend" in df.columns and "Neural Network Backend" not in df.columns:
        df["Neural Network Backend"] = df["Backend"]

    if "Rows Used" in df.columns and "Max Rows" not in df.columns:
        df["Max Rows"] = df["Rows Used"]

    if "Target Clipping" not in df.columns:
        df["Target Clipping"] = "none"

    for col in STANDARD_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    numeric_cols = [
        "RMSE",
        "MAE",
        "R2",
        "Train Seconds",
        "Predict Seconds",
        "Max Rows",
        "Clip Low Quantile",
        "Clip High Quantile",
        "Clip Low Value",
        "Clip High Value",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df[STANDARD_COLUMNS]


def main() -> None:
    if not RESULTS_DIR.exists():
        raise FileNotFoundError(f"Missing results directory: {RESULTS_DIR}")

    files = result_files()
    if not files:
        raise FileNotFoundError(f"No result CSVs found in {RESULTS_DIR}")

    combined = pd.concat([normalize_result_frame(path) for path in files], ignore_index=True)
    dedupe_cols = [
        "Dataset Mode",
        "Target",
        "Split",
        "Feature Set",
        "Model",
        "Max Rows",
        "Target Clipping",
        "Clip Low Quantile",
        "Clip High Quantile",
    ]
    combined = combined.drop_duplicates(subset=dedupe_cols, keep="last")
    combined = combined.sort_values(
        by=["Dataset Mode", "Target", "Split", "R2"],
        ascending=[True, True, True, False],
    ).reset_index(drop=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUTPUT_PATH, index=False)

    print(f"Read {len(files)} result files.")
    print(f"Saved combined results to: {OUTPUT_PATH}")
    print("\nCompact summary")
    print("---------------")
    summary_cols = ["Dataset Mode", "Target", "Split", "Model", "RMSE", "MAE", "R2"]
    print(combined[summary_cols].to_string(index=False, max_rows=80))


if __name__ == "__main__":
    main()
