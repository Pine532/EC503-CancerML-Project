import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

GDSC_PATH = BASE_DIR / "GDSC_DATASET.csv"
EXPR_PATH = BASE_DIR / "data" / "expression" / "Cell_line_RMA_proc_basalExp.txt"
OUT_PATH = BASE_DIR / "data" / "gdsc_metadata_expression_top500.csv"

N_TOP_GENES = 500

print("Loading GDSC dataset...")
gdsc = pd.read_csv(GDSC_PATH)

print("GDSC shape:", gdsc.shape)

print("Loading expression dataset...")
expr = pd.read_csv(EXPR_PATH, sep="\t")

print("Raw expression shape:", expr.shape)
print("First columns:", expr.columns[:5].tolist())

# Keep only expression columns plus gene symbols to avoid carrying labels like GENE_title.
expr_cols = [col for col in expr.columns if col.startswith("DATA.")]
expr = expr[["GENE_SYMBOLS"] + expr_cols].copy()
expr = expr.dropna(subset=["GENE_SYMBOLS"])
expr = expr.set_index("GENE_SYMBOLS")

# Convert expression values to numeric
expr = expr.apply(pd.to_numeric, errors="coerce")

# Duplicate gene symbols cause the "top 500" selection to explode into far more columns.
# Collapse them before transposing so each gene appears once.
expr = expr.groupby(expr.index).mean()

# Transpose so rows = cell lines, columns = genes
expr_t = expr.T

# Clean COSMIC IDs from row labels like DATA.906826 and DATA.1503362.1
expr_t["COSMIC_ID"] = (
    expr_t.index.to_series()
    .astype(str)
    .str.extract(r"(\d+)")[0]
    .to_numpy()
)

expr_t = expr_t.dropna(subset=["COSMIC_ID"])
expr_t["COSMIC_ID"] = expr_t["COSMIC_ID"].astype(int)

print("Transposed expression shape:", expr_t.shape)

# Some COSMIC IDs may appear more than once after cleaning .1 suffixes.
# Average duplicate rows if present.
expr_t = expr_t.groupby("COSMIC_ID", as_index=False).mean()

print("Expression shape after duplicate handling:", expr_t.shape)

# Select top N most variable genes
gene_cols = [col for col in expr_t.columns if col != "COSMIC_ID"]

gene_variances = expr_t[gene_cols].var(axis=0).sort_values(ascending=False)
top_genes = gene_variances.head(N_TOP_GENES).index.tolist()

expr_top = expr_t[["COSMIC_ID"] + top_genes]

print(f"Selected top {N_TOP_GENES} most variable genes.")
print("Expression top-gene shape:", expr_top.shape)

# Merge with GDSC response table
merged = gdsc.merge(expr_top, on="COSMIC_ID", how="inner")

print("Merged shape:", merged.shape)
print("Unique COSMIC_IDs in GDSC:", gdsc["COSMIC_ID"].nunique())
print("Unique COSMIC_IDs in expression:", expr_top["COSMIC_ID"].nunique())
print("Unique COSMIC_IDs after merge:", merged["COSMIC_ID"].nunique())

merged.to_csv(OUT_PATH, index=False)

print(f"Saved merged dataset to: {OUT_PATH}")
