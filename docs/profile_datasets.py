from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from cancer_ml_utils import (
    SUPPORTED_DATASETS,
    load_model_dataframe,
)


OUTPUT_DIR = PROJECT_DIR / "docs" / "dataset_profile"
COUNTS_DIR = OUTPUT_DIR / "categorical_value_counts"


def pct(value: float) -> float:
    return float(value * 100)


def safe_slug(text: str) -> str:
    return (
        text.lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("-", "_")
        .replace(".", "_")
    )


def format_top_values(counts: pd.Series, total: int, n: int = 8) -> str:
    pieces = []
    for value, count in counts.head(n).items():
        pieces.append(f"{value}: {count} ({pct(count / total):.1f}%)")
    return "; ".join(pieces)


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"

    display = df.copy()
    for col in display.columns:
        display[col] = display[col].map(
            lambda value: "" if pd.isna(value) else str(value).replace("\n", " ")
        )

    header = "| " + " | ".join(display.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    rows = [
        "| " + " | ".join(row) + " |"
        for row in display.astype(str).itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def profile_dataset(dataset_mode: str) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    model_data = load_model_dataframe(dataset=dataset_mode)
    df = model_data.model_df
    target = df[model_data.target_col]

    dataset_rows = [{
        "Dataset Mode": dataset_mode,
        "Dataset Name": model_data.dataset_name,
        "Target": model_data.target_col,
        "Raw Rows": model_data.raw_shape[0],
        "Raw Columns": model_data.raw_shape[1],
        "Modeling Rows": len(df),
        "Modeling Columns": df.shape[1],
        "Predictor Features Before Encoding": len(model_data.feature_columns),
        "Categorical Features": len(model_data.categorical_cols),
        "Numerical Features": len(model_data.numerical_cols),
        "Estimated One-Hot Columns": int(
            sum(df[col].nunique(dropna=True) for col in model_data.categorical_cols)
        ),
        "Estimated Encoded Feature Columns": int(
            sum(df[col].nunique(dropna=True) for col in model_data.categorical_cols)
            + len(model_data.numerical_cols)
        ),
        "Target Mean": target.mean(),
        "Target Std": target.std(),
        "Target Min": target.min(),
        "Target 25%": target.quantile(0.25),
        "Target Median": target.quantile(0.50),
        "Target 75%": target.quantile(0.75),
        "Target Max": target.max(),
    }]

    feature_rows = []
    category_rows = []
    continuous_rows = []

    for col in model_data.categorical_cols:
        counts = df[col].fillna("Missing").value_counts(dropna=False)
        total = int(counts.sum())
        count_frame = counts.rename_axis("Value").reset_index(name="Count")
        count_frame["Percent"] = count_frame["Count"] / total * 100
        count_path = COUNTS_DIR / f"{dataset_mode}__{safe_slug(col)}.csv"
        count_frame.to_csv(count_path, index=False)

        top_value = str(counts.index[0]) if len(counts) else ""
        top_count = int(counts.iloc[0]) if len(counts) else 0
        rare_count = int(counts.iloc[-1]) if len(counts) else 0

        feature_rows.append({
            "Dataset Mode": dataset_mode,
            "Feature": col,
            "Type": "categorical",
            "Unique Values": int(df[col].nunique(dropna=True)),
            "Missing Count": int(df[col].isna().sum()),
            "Most Common Value": top_value,
            "Most Common Count": top_count,
            "Most Common Percent": pct(top_count / total) if total else np.nan,
            "Rarest Category Count": rare_count,
            "Top Values": format_top_values(counts, total),
            "Value Counts CSV": str(count_path.relative_to(PROJECT_DIR)),
        })

        for value, count in counts.items():
            category_rows.append({
                "Dataset Mode": dataset_mode,
                "Feature": col,
                "Value": value,
                "Count": int(count),
                "Percent": pct(count / total) if total else np.nan,
            })

    for col in model_data.numerical_cols:
        series = pd.to_numeric(df[col], errors="coerce")
        continuous_rows.append({
            "Dataset Mode": dataset_mode,
            "Feature": col,
            "Type": "continuous",
            "Missing Count": int(series.isna().sum()),
            "Mean": series.mean(),
            "Std": series.std(),
            "Min": series.min(),
            "25%": series.quantile(0.25),
            "Median": series.quantile(0.50),
            "75%": series.quantile(0.75),
            "Max": series.max(),
        })

    return dataset_rows, feature_rows, category_rows, continuous_rows


def write_markdown(
    dataset_summary: pd.DataFrame,
    categorical_summary: pd.DataFrame,
    continuous_summary: pd.DataFrame,
) -> None:
    lines = [
        "# Dataset Profile Summary",
        "",
        "This file summarizes the datasets exactly as the canonical modeling pipeline loads them.",
        "",
        "## Dataset-Level Summary",
        "",
        markdown_table(dataset_summary),
        "",
        "## Categorical Feature Summary",
        "",
        markdown_table(categorical_summary[
            [
                "Dataset Mode",
                "Feature",
                "Unique Values",
                "Missing Count",
                "Most Common Value",
                "Most Common Count",
                "Most Common Percent",
                "Top Values",
            ]
        ]),
        "",
        "## Continuous Feature Summary",
        "",
    ]

    if continuous_summary.empty:
        lines.append("No continuous predictor columns are used.")
    else:
        compact = continuous_summary.copy()
        compact = compact.sort_values(["Dataset Mode", "Std"], ascending=[True, False])
        lines.append(
            markdown_table(compact[
                ["Dataset Mode", "Feature", "Mean", "Std", "Min", "Median", "Max"]
            ].head(80))
        )
        lines.append("")
        lines.append("For complete continuous-feature statistics, see `continuous_feature_summary.csv`.")

    lines.extend([
        "",
        "## How Categorical Features Become Numerical",
        "",
        "Linear regression cannot directly consume strings such as tissue names or drug targets. The pipeline first imputes missing categorical values with the most frequent value, then applies one-hot encoding. One-hot encoding creates one binary indicator column per category value. For example, if `Screen Medium` has values `R` and `D/F12`, it becomes two numerical columns indicating which value each row has.",
        "",
        "After this preprocessing, Linear Regression, Ridge, Lasso, Linear SVR, and the other models receive a numerical design matrix.",
        "",
    ])

    (OUTPUT_DIR / "DATASET_PROFILE_SUMMARY.md").write_text("\n".join(lines))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    COUNTS_DIR.mkdir(parents=True, exist_ok=True)

    dataset_rows = []
    feature_rows = []
    category_rows = []
    continuous_rows = []

    for dataset_mode in SUPPORTED_DATASETS:
        ds_rows, feat_rows, cat_rows, cont_rows = profile_dataset(dataset_mode)
        dataset_rows.extend(ds_rows)
        feature_rows.extend(feat_rows)
        category_rows.extend(cat_rows)
        continuous_rows.extend(cont_rows)

    dataset_summary = pd.DataFrame(dataset_rows)
    feature_summary = pd.DataFrame(feature_rows)
    category_counts = pd.DataFrame(category_rows)
    continuous_summary = pd.DataFrame(continuous_rows)

    categorical_summary = feature_summary[feature_summary["Type"] == "categorical"].copy()

    dataset_summary.to_csv(OUTPUT_DIR / "dataset_summary.csv", index=False)
    categorical_summary.to_csv(OUTPUT_DIR / "categorical_feature_summary.csv", index=False)
    category_counts.to_csv(OUTPUT_DIR / "all_categorical_value_counts.csv", index=False)
    continuous_summary.to_csv(OUTPUT_DIR / "continuous_feature_summary.csv", index=False)

    write_markdown(dataset_summary, categorical_summary, continuous_summary)

    print(f"Wrote dataset profile to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
