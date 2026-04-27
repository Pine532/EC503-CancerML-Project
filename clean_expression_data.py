import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

INPUT_PATH = BASE_DIR / "data" / "expression" / "Cell_line_RMA_proc_basalExp.txt"
OUTPUT_DIR = BASE_DIR / "data" / "expression_cleaned"

GENE_ROWS_CSV = OUTPUT_DIR / "gene_expression_clean_gene_rows.csv"
CELL_ROWS_TOP500_CSV = OUTPUT_DIR / "gene_expression_top500_cell_rows.csv"
SUMMARY_CSV = OUTPUT_DIR / "gene_expression_summary.csv"

N_TOP_GENES = 500

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading raw expression file...")
expr_raw = pd.read_csv(INPUT_PATH, sep="\t")

print("Raw shape:", expr_raw.shape)
print("First columns:", expr_raw.columns[:5].tolist())

if "GENE_SYMBOLS" not in expr_raw.columns:
    raise ValueError("Expected column 'GENE_SYMBOLS' not found.")

# Keep only DATA columns
expr_cols = [col for col in expr_raw.columns if col.startswith("DATA.")]
print("Number of DATA expression columns:", len(expr_cols))

if len(expr_cols) == 0:
    raise ValueError("No DATA.* expression columns found.")

# Build gene-row matrix
expr_values = expr_raw[["GENE_SYMBOLS"] + expr_cols].copy()
expr_values = expr_values.dropna(subset=["GENE_SYMBOLS"])
expr_values = expr_values.set_index("GENE_SYMBOLS")

# Convert all expression values to numeric
expr_values = expr_values.apply(pd.to_numeric, errors="coerce")

# Average duplicate gene symbols if any
expr_values = expr_values.groupby(expr_values.index).mean()

print("Gene-row matrix shape:", expr_values.shape)

# Transpose: rows = cell lines, columns = genes
expr_cell_rows = expr_values.T

print("Before COSMIC_ID extraction:")
print(expr_cell_rows.index[:5].tolist())

# Robustly extract the numeric COSMIC_ID from names like DATA.906826
expr_cell_rows["COSMIC_ID"] = (
    expr_cell_rows.index.to_series()
    .astype(str)
    .str.extract(r"(\d+)")[0]
    .to_numpy()
)

missing_ids = expr_cell_rows["COSMIC_ID"].isna().sum()
print("Rows missing extracted COSMIC_ID:", missing_ids)

expr_cell_rows = expr_cell_rows.dropna(subset=["COSMIC_ID"])
expr_cell_rows["COSMIC_ID"] = expr_cell_rows["COSMIC_ID"].astype(int)

# Average duplicate COSMIC_ID rows, e.g. DATA.1503362 and DATA.1503362.1
expr_cell_rows = expr_cell_rows.groupby("COSMIC_ID", as_index=False).mean()

print("Cell-row expression shape:", expr_cell_rows.shape)

if expr_cell_rows.shape[0] == 0:
    raise ValueError("Expression cell-row table has 0 rows after COSMIC_ID extraction.")

# Save cleaned gene-row version for readability
gene_rows_clean = expr_cell_rows.set_index("COSMIC_ID").T
gene_rows_clean.index.name = "GENE_SYMBOLS"
gene_rows_clean = gene_rows_clean.reset_index()

print("Saving readable gene-row CSV...")
gene_rows_clean.to_csv(GENE_ROWS_CSV, index=False)

# Gene summary statistics
gene_cols = [col for col in expr_cell_rows.columns if col != "COSMIC_ID"]

summary = pd.DataFrame({
    "GENE_SYMBOLS": gene_cols,
    "mean_expression": expr_cell_rows[gene_cols].mean(axis=0).values,
    "std_expression": expr_cell_rows[gene_cols].std(axis=0).values,
    "min_expression": expr_cell_rows[gene_cols].min(axis=0).values,
    "max_expression": expr_cell_rows[gene_cols].max(axis=0).values,
    "missing_count": expr_cell_rows[gene_cols].isna().sum(axis=0).values
})

summary = summary.sort_values("std_expression", ascending=False)

print("Saving gene summary CSV...")
summary.to_csv(SUMMARY_CSV, index=False)

# Select top variable genes
top_genes = summary.head(N_TOP_GENES)["GENE_SYMBOLS"].tolist()
expr_top500 = expr_cell_rows[["COSMIC_ID"] + top_genes].copy()

print(f"Top-{N_TOP_GENES} expression shape:", expr_top500.shape)

if expr_top500.shape[0] == 0:
    raise ValueError("Top gene expression table has 0 rows.")

print("Saving top-500 cell-row expression CSV...")
expr_top500.to_csv(CELL_ROWS_TOP500_CSV, index=False)

print("\nDone.")
print(f"Saved: {GENE_ROWS_CSV}")
print(f"Saved: {CELL_ROWS_TOP500_CSV}")
print(f"Saved: {SUMMARY_CSV}")
