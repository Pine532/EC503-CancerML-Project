# clean_secondary_screen.py

import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

INPUT_PATH = BASE_DIR / "secondary-screen-dose-response-curve-parameters.csv"

OUTPUT_DIR = BASE_DIR / "data" / "secondary_screen"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PARQUET = OUTPUT_DIR / "secondary_screen_auc_clean.parquet"
OUTPUT_SAMPLE = OUTPUT_DIR / "secondary_screen_auc_sample.csv"
OUTPUT_SUMMARY = OUTPUT_DIR / "secondary_screen_summary.csv"

print("Loading secondary screen dataset...")
df = pd.read_csv(INPUT_PATH, low_memory=False)

print("Raw shape:", df.shape)
print("Columns:", df.columns.tolist())

# --------------------------------------------------
# 1. Basic row filtering
# --------------------------------------------------
# Keep rows with usable cell-line ID and target.
df = df.dropna(subset=["depmap_id", "auc"]).copy()

# Keep only rows that passed structural profiling.
if "passed_str_profiling" in df.columns:
    df = df[df["passed_str_profiling"] == True].copy()

print("Shape after basic filtering:", df.shape)

# --------------------------------------------------
# 2. Define target
# --------------------------------------------------
df["target_auc"] = pd.to_numeric(df["auc"], errors="coerce")
df = df.dropna(subset=["target_auc"]).copy()

# Optional: remove extreme AUC values if needed.
# Keep this conservative for now.
df = df[np.isfinite(df["target_auc"])].copy()

# --------------------------------------------------
# 3. Clean categorical feature columns
# --------------------------------------------------
categorical_cols = [
    "depmap_id",
    "ccle_name",
    "screen_id",
    "name",
    "moa",
    "target",
    "disease.area",
    "indication",
    "phase",
]

for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].astype("string").fillna("Unknown")

# Derive tissue type from CCLE name if possible.
# Example: MFE296_ENDOMETRIUM -> ENDOMETRIUM
df["ccle_tissue"] = (
    df["ccle_name"]
    .astype("string")
    .str.split("_")
    .str[-1]
    .fillna("Unknown")
)

# --------------------------------------------------
# 4. Create optional cleaned continuous response fields
# --------------------------------------------------
# These should NOT be predictors if target is AUC.
df["ic50_clean"] = pd.to_numeric(df["ic50"], errors="coerce")
df["ec50_clean"] = pd.to_numeric(df["ec50"], errors="coerce")

df["log_ic50"] = np.where(
    (df["ic50_clean"] > 0) & np.isfinite(df["ic50_clean"]),
    np.log10(df["ic50_clean"]),
    np.nan
)

df["log_ec50"] = np.where(
    (df["ec50_clean"] > 0) & np.isfinite(df["ec50_clean"]),
    np.log10(df["ec50_clean"]),
    np.nan
)

# --------------------------------------------------
# 5. Keep modeling columns
# --------------------------------------------------
model_cols = [
    "target_auc",
    "broad_id",
    "depmap_id",
    "ccle_name",
    "ccle_tissue",
    "screen_id",
    "name",
    "moa",
    "target",
    "disease.area",
    "indication",
    "phase",
    "smiles",
    "log_ic50",
    "log_ec50",
]

model_cols = [col for col in model_cols if col in df.columns]

clean = df[model_cols].copy()

# Make ID/text fields consistent.
for col in clean.select_dtypes(include=["object", "string"]).columns:
    clean[col] = clean[col].astype("string").fillna("Unknown")

print("Clean modeling shape:", clean.shape)

# --------------------------------------------------
# 6. Save outputs
# --------------------------------------------------
clean.to_parquet(OUTPUT_PARQUET, index=False, compression="snappy")
clean.head(1000).to_csv(OUTPUT_SAMPLE, index=False)

# Summary table
summary_rows = []
for col in clean.columns:
    summary_rows.append({
        "column": col,
        "missing_count": clean[col].isna().sum(),
        "unique_values": clean[col].nunique(dropna=True),
        "dtype": str(clean[col].dtype)
    })

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUTPUT_SUMMARY, index=False)

print("Saved:")
print(OUTPUT_PARQUET)
print(OUTPUT_SAMPLE)
print(OUTPUT_SUMMARY)

print("\nTarget AUC summary:")
print(clean["target_auc"].describe())

print("\nTop tissues:")
print(clean["ccle_tissue"].value_counts().head(10))

print("\nTop drug names:")
print(clean["name"].value_counts().head(10))