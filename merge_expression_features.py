# merge_expression_features.py

import pandas as pd
from pathlib import Path

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent

GDSC_PATH = BASE_DIR / "GDSC_DATASET.csv"
EXPR_TOP500_PATH = BASE_DIR / "data" / "expression_cleaned" / "gene_expression_top500_cell_rows.csv"

OUTPUT_DIR = BASE_DIR / "data" / "model_ready"
OUTPUT_PATH = OUTPUT_DIR / "gdsc_metadata_expression_top500.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# 1. Load datasets
# -----------------------------
print("Loading GDSC dataset...")
gdsc = pd.read_csv(GDSC_PATH)

print("Loading cleaned top-500 expression dataset...")
if not EXPR_TOP500_PATH.exists():
    raise FileNotFoundError(
        f"Missing cleaned expression file: {EXPR_TOP500_PATH}. "
        "Run clean_expression_data.py first."
    )
expr = pd.read_csv(EXPR_TOP500_PATH)

print("\nInitial shapes:")
print("GDSC shape:", gdsc.shape)
print("Expression shape:", expr.shape)

# -----------------------------
# 2. Validate merge key
# -----------------------------
if "COSMIC_ID" not in gdsc.columns:
    raise ValueError("COSMIC_ID not found in GDSC dataset.")

if "COSMIC_ID" not in expr.columns:
    raise ValueError("COSMIC_ID not found in expression dataset.")

gdsc["COSMIC_ID"] = gdsc["COSMIC_ID"].astype(int)
expr["COSMIC_ID"] = expr["COSMIC_ID"].astype(int)

# -----------------------------
# 3. Check overlap
# -----------------------------
gdsc_ids = set(gdsc["COSMIC_ID"].unique())
expr_ids = set(expr["COSMIC_ID"].unique())

overlap_ids = gdsc_ids.intersection(expr_ids)

print("\nCOSMIC_ID coverage:")
print("Unique COSMIC_IDs in GDSC:", len(gdsc_ids))
print("Unique COSMIC_IDs in expression:", len(expr_ids))
print("Overlapping COSMIC_IDs:", len(overlap_ids))

if len(overlap_ids) == 0:
    raise ValueError("No overlapping COSMIC_IDs found. Merge would produce empty dataset.")

# -----------------------------
# 4. Merge
# -----------------------------
print("\nMerging datasets on COSMIC_ID...")

merged = gdsc.merge(
    expr,
    on="COSMIC_ID",
    how="inner"
)

print("Merged shape:", merged.shape)

# -----------------------------
# 5. Basic diagnostics
# -----------------------------
expression_cols = [col for col in expr.columns if col != "COSMIC_ID"]

print("\nExpression feature count:", len(expression_cols))
print("Rows retained after merge:", len(merged))
print("Unique COSMIC_IDs after merge:", merged["COSMIC_ID"].nunique())

print("\nTarget check:")
print(merged["LN_IC50"].describe())

print("\nMissing expression values:")
missing_expr_total = merged[expression_cols].isna().sum().sum()
print("Total missing expression values:", missing_expr_total)

# -----------------------------
# 6. Save model-ready dataset
# -----------------------------
merged.to_csv(OUTPUT_PATH, index=False)

print(f"\nSaved merged dataset to:\n{OUTPUT_PATH}")
