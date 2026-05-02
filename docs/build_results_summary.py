from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_DIR / "results"
DOCS_DIR = PROJECT_DIR / "docs"
COMBINED_OUTPUT = RESULTS_DIR / "all_results_with_optimized.csv"
MARKDOWN_OUTPUT = DOCS_DIR / "RESULTS_ALL_IN_ONE.md"


DISPLAY_COLUMNS = [
    "Experiment Type",
    "Dataset Mode",
    "Target",
    "Split",
    "Feature Set",
    "Model",
    "RMSE",
    "MAE",
    "R2",
    "Test RMSE",
    "Test MAE",
    "Test R2",
    "Train Seconds",
    "Predict Seconds",
    "Tuning Max Rows",
    "N Iter",
    "CV Folds",
    "Best CV RMSE",
    "Best Params JSON",
    "Convergence Warnings",
    "Source File",
]


def markdown_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df.empty:
        return "_No rows._"

    display = df.head(max_rows).copy() if max_rows is not None else df.copy()
    for col in display.columns:
        display[col] = display[col].map(format_value)

    header = "| " + " | ".join(display.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    rows = [
        "| " + " | ".join(row) + " |"
        for row in display.astype(str).itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def format_value(value) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        if abs(value) >= 100:
            return f"{value:.1f}"
        return f"{value:.4f}"
    text = str(value)
    return text.replace("\n", " ").replace("|", "\\|")


def read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def normalize_baseline(path: Path) -> pd.DataFrame:
    df = read_csv(path)
    if df.empty:
        return df

    if "Backend" in df.columns and "Neural Network Backend" not in df.columns:
        df["Neural Network Backend"] = df["Backend"]
    if "Rows Used" in df.columns and "Max Rows" not in df.columns:
        df["Max Rows"] = df["Rows Used"]

    normalized = pd.DataFrame({
        "Experiment Type": "Baseline",
        "Dataset": df.get("Dataset", np.nan),
        "Dataset Mode": df.get("Dataset Mode", np.nan),
        "Target": df.get("Target", np.nan),
        "Split": df.get("Split", np.nan),
        "Feature Set": df.get("Feature Set", np.nan),
        "Model": df.get("Model", np.nan),
        "RMSE": df.get("RMSE", np.nan),
        "MAE": df.get("MAE", np.nan),
        "R2": df.get("R2", np.nan),
        "Test RMSE": np.nan,
        "Test MAE": np.nan,
        "Test R2": np.nan,
        "Train Seconds": df.get("Train Seconds", np.nan),
        "Predict Seconds": df.get("Predict Seconds", np.nan),
        "Max Rows": df.get("Max Rows", np.nan),
        "Target Clipping": df.get("Target Clipping", "none"),
        "Tuning Max Rows": np.nan,
        "Final Train Max Rows": np.nan,
        "N Iter": np.nan,
        "CV Folds": np.nan,
        "Best CV RMSE": np.nan,
        "Best Params JSON": "",
        "Param Search Space JSON": "",
        "Convergence Warnings": "",
        "Source File": path.name,
    })
    return normalized


def normalize_optimized(path: Path, params_by_key: dict[tuple[str, str, str], dict]) -> pd.DataFrame:
    df = read_csv(path)
    if df.empty:
        return df

    rows = []
    for _, row in df.iterrows():
        key = (
            row.get("Dataset Mode", ""),
            row.get("Split", ""),
            row.get("Model", ""),
        )
        params = params_by_key.get(key, {})
        rows.append({
            "Experiment Type": "Optimized",
            "Dataset": row.get("Dataset", np.nan),
            "Dataset Mode": row.get("Dataset Mode", np.nan),
            "Target": row.get("Target", np.nan),
            "Split": row.get("Split", np.nan),
            "Feature Set": row.get("Feature Set", np.nan),
            "Model": row.get("Model", np.nan),
            "RMSE": row.get("Validation RMSE", np.nan),
            "MAE": row.get("Validation MAE", np.nan),
            "R2": row.get("Validation R2", np.nan),
            "Test RMSE": row.get("Test RMSE", np.nan),
            "Test MAE": row.get("Test MAE", np.nan),
            "Test R2": row.get("Test R2", np.nan),
            "Train Seconds": row.get("Final Train Seconds", np.nan),
            "Predict Seconds": row.get("Predict Seconds", np.nan),
            "Max Rows": np.nan,
            "Target Clipping": "none",
            "Tuning Max Rows": row.get("Tuning Max Rows", np.nan),
            "Final Train Max Rows": row.get("Final Train Max Rows", np.nan),
            "N Iter": row.get("N Iter", np.nan),
            "CV Folds": row.get("CV Folds", np.nan),
            "Best CV RMSE": row.get("Best CV RMSE", np.nan),
            "Best Params JSON": params.get("Best Params JSON", ""),
            "Param Search Space JSON": params.get("Param Search Space JSON", ""),
            "Convergence Warnings": row.get("Convergence Warnings", ""),
            "Source File": path.name,
        })
    return pd.DataFrame(rows)


def load_params() -> dict[tuple[str, str, str], dict]:
    params_by_key = {}
    for path in sorted(RESULTS_DIR.glob("optimized_model_comparison*_best_params.csv")):
        df = read_csv(path)
        if df.empty:
            continue
        for _, row in df.iterrows():
            key = (
                row.get("Dataset Mode", ""),
                row.get("Split", ""),
                row.get("Model", ""),
            )
            params_by_key[key] = {
                "Best Params JSON": row.get("Best Params JSON", ""),
                "Param Search Space JSON": row.get("Param Search Space JSON", ""),
                "Best Params Source File": path.name,
            }
    return params_by_key


def compact_params(params_json: str) -> str:
    if not isinstance(params_json, str) or not params_json:
        return ""
    try:
        params = json.loads(params_json)
    except json.JSONDecodeError:
        return params_json
    return ", ".join(f"{key.replace('model__', '').replace('regressor__model__', '')}={value}" for key, value in params.items())


def combine_results() -> pd.DataFrame:
    params_by_key = load_params()
    frames = []

    baseline_files = [
        *RESULTS_DIR.glob("model_comparison*.csv"),
        *RESULTS_DIR.glob("neural_network*.csv"),
    ]
    for path in sorted(path for path in baseline_files if path.name != "all_results_combined.csv"):
        frames.append(normalize_baseline(path))

    optimized_files = [
        path
        for path in RESULTS_DIR.glob("optimized_model_comparison*.csv")
        if not path.name.endswith("_best_params.csv")
    ]
    for path in sorted(optimized_files):
        frames.append(normalize_optimized(path, params_by_key))

    combined = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True)

    numeric_cols = [
        "RMSE",
        "MAE",
        "R2",
        "Test RMSE",
        "Test MAE",
        "Test R2",
        "Train Seconds",
        "Predict Seconds",
        "Max Rows",
        "Tuning Max Rows",
        "Final Train Max Rows",
        "N Iter",
        "CV Folds",
        "Best CV RMSE",
    ]
    for col in numeric_cols:
        combined[col] = pd.to_numeric(combined[col], errors="coerce")

    metadata_only_mask = combined["Dataset Mode"].isna() & (combined["Feature Set"] == "metadata_only")
    combined.loc[metadata_only_mask, "Dataset"] = "gdsc"
    combined.loc[metadata_only_mask, "Dataset Mode"] = "gdsc_metadata_only"
    combined.loc[metadata_only_mask, "Target"] = "LN_IC50"

    metadata_expression_mask = combined["Dataset Mode"].isna() & (
        combined["Feature Set"] == "metadata_plus_expression"
    )
    combined.loc[metadata_expression_mask, "Dataset"] = "gdsc"
    combined.loc[metadata_expression_mask, "Dataset Mode"] = "gdsc_metadata_plus_expression"
    combined.loc[metadata_expression_mask, "Target"] = "LN_IC50"

    combined["Primary R2 For Dedupe"] = combined["Test R2"].fillna(combined["R2"])
    combined["Primary RMSE For Dedupe"] = combined["Test RMSE"].fillna(combined["RMSE"])
    combined = combined.sort_values(
        by=["Primary R2 For Dedupe", "Primary RMSE For Dedupe"],
        ascending=[False, True],
    )
    dedupe_cols = [
        "Experiment Type",
        "Dataset Mode",
        "Target",
        "Split",
        "Feature Set",
        "Model",
        "Max Rows",
        "Target Clipping",
        "Tuning Max Rows",
        "N Iter",
        "CV Folds",
    ]
    combined = combined.drop_duplicates(subset=dedupe_cols, keep="first")
    combined = combined.drop(columns=["Primary R2 For Dedupe", "Primary RMSE For Dedupe"])

    combined["Best Params Compact"] = combined["Best Params JSON"].map(compact_params)
    combined = combined.sort_values(
        by=["Dataset Mode", "Target", "Split", "Experiment Type", "R2"],
        ascending=[True, True, True, True, False],
    ).reset_index(drop=True)
    return combined


def best_rows(combined: pd.DataFrame) -> pd.DataFrame:
    metric_df = combined.copy()
    metric_df["Primary R2"] = metric_df["Test R2"].fillna(metric_df["R2"])
    metric_df["Primary RMSE"] = metric_df["Test RMSE"].fillna(metric_df["RMSE"])
    metric_df = metric_df.dropna(subset=["Primary R2"])
    idx = metric_df.groupby(["Dataset Mode", "Target", "Split"], dropna=False)["Primary R2"].idxmax()
    return metric_df.loc[idx].sort_values(["Dataset Mode", "Split"]).reset_index(drop=True)


def write_markdown(combined: pd.DataFrame) -> None:
    best = best_rows(combined)
    optimized = combined[combined["Experiment Type"] == "Optimized"].copy()
    baseline = combined[combined["Experiment Type"] == "Baseline"].copy()

    gdsc_auc = combined[
        combined["Dataset Mode"].astype(str).str.startswith("gdsc_auc")
    ].copy()
    gdsc_ln_ic50 = combined[
        combined["Dataset Mode"].astype(str).isin(["gdsc_metadata_only", "gdsc_metadata_plus_expression"])
    ].copy()
    secondary = combined[
        combined["Dataset Mode"].astype(str).str.startswith("secondary_screen")
    ].copy()

    display_cols = [
        "Experiment Type",
        "Dataset Mode",
        "Target",
        "Split",
        "Model",
        "RMSE",
        "MAE",
        "R2",
        "Test RMSE",
        "Test MAE",
        "Test R2",
        "Best Params Compact",
    ]
    best_cols = [
        "Dataset Mode",
        "Target",
        "Split",
        "Experiment Type",
        "Model",
        "Primary RMSE",
        "Primary R2",
        "Best Params Compact",
    ]
    opt_cols = [
        "Dataset Mode",
        "Target",
        "Split",
        "Model",
        "Tuning Max Rows",
        "N Iter",
        "CV Folds",
        "Best CV RMSE",
        "RMSE",
        "R2",
        "Test RMSE",
        "Test R2",
        "Best Params Compact",
    ]

    lines = [
        "# Results Summary",
        "",
        "This file consolidates the current baseline results, optimized results, and optimized hyperparameters.",
        "",
        "The full machine-readable table is `results/all_results_with_optimized.csv`.",
        "",
        "## Best Result Per Dataset/Split",
        "",
        markdown_table(best[best_cols]),
        "",
        "## Optimized Runs",
        "",
        markdown_table(optimized[opt_cols]),
        "",
        "## GDSC AUC Results",
        "",
        markdown_table(gdsc_auc[display_cols]),
        "",
        "## GDSC LN_IC50 Results",
        "",
        markdown_table(gdsc_ln_ic50[display_cols]),
        "",
        "## Secondary Screen Results",
        "",
        markdown_table(secondary[display_cols]),
        "",
        "## Notes",
        "",
        "- Baseline `RMSE`, `MAE`, and `R2` are validation metrics saved by `model_comparison.py` and `neural_network_baseline.py`.",
        "- Optimized rows include validation metrics plus final held-out test metrics.",
        "- Random Forest and Neural Network were abandoned for the main optimized workflow because they were slow relative to their current benefit.",
        "- Lasso is excluded from optimization because it takes too long on these datasets.",
        "- Optimized Gradient Boosting uses XGBoost in this environment.",
        "",
        f"Total baseline rows: {len(baseline)}.",
        f"Total optimized rows: {len(optimized)}.",
    ]
    MARKDOWN_OUTPUT.write_text("\n".join(lines))


def main() -> None:
    combined = combine_results()
    combined.to_csv(COMBINED_OUTPUT, index=False)
    write_markdown(combined)
    print(f"Saved combined results to: {COMBINED_OUTPUT}")
    print(f"Saved markdown summary to: {MARKDOWN_OUTPUT}")
    print(f"Rows: {len(combined)}")


if __name__ == "__main__":
    main()
