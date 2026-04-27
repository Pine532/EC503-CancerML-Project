import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

INPUT_PATH = BASE_DIR / "data" / "model_ready" / "gdsc_metadata_expression_top500.csv"
OUTPUT_PATH = BASE_DIR / "data" / "model_ready" / "gdsc_metadata_expression_top500.parquet"

print("Loading merged CSV...")
df = pd.read_csv(INPUT_PATH, low_memory=False)

print("Original shape:", df.shape)

# Original GDSC metadata / response columns
metadata_cols = [
    "COSMIC_ID",
    "CELL_LINE_NAME",
    "TCGA_DESC",
    "DRUG_ID",
    "DRUG_NAME",
    "LN_IC50",
    "AUC",
    "Z_SCORE",
    "GDSC Tissue descriptor 1",
    "GDSC Tissue descriptor 2",
    "Cancer Type (matching TCGA label)",
    "Microsatellite instability Status (MSI)",
    "Screen Medium",
    "Growth Properties",
    "CNA",
    "Gene Expression",
    "Methylation",
    "TARGET",
    "TARGET_PATHWAY",
]

# Columns not in original metadata are the 500 gene-expression features
gene_cols = [col for col in df.columns if col not in metadata_cols]

print("Number of gene-expression columns:", len(gene_cols))

# -----------------------------
# Clean numeric columns
# -----------------------------
df["COSMIC_ID"] = pd.to_numeric(df["COSMIC_ID"], errors="coerce").astype("Int64")
df["DRUG_ID"] = pd.to_numeric(df["DRUG_ID"], errors="coerce").astype("Int64")

for col in ["LN_IC50", "AUC", "Z_SCORE"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")

# Gene-expression columns should be continuous numeric values
for col in gene_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")

# -----------------------------
# Clean string / categorical columns
# -----------------------------
string_cols = [
    col for col in metadata_cols
    if col in df.columns and col not in ["COSMIC_ID", "DRUG_ID", "LN_IC50", "AUC", "Z_SCORE"]
]

for col in string_cols:
    df[col] = df[col].astype("string").fillna("Unknown")

print("\nCleaned dtypes:")
print(df.dtypes.head(25))

print("\nSaving Parquet...")
df.to_parquet(OUTPUT_PATH, index=False, compression="snappy")

print("Saved:", OUTPUT_PATH)
print("Final shape:", df.shape)
print("Number of gene-expression features:", len(gene_cols))