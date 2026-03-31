import pandas as pd

# Load main dataset
df = pd.read_csv("GDSC_DATASET.csv")

# Target
target_col = "LN_IC50"

# First-pass feature set
feature_cols = [
    "TCGA_DESC",
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
    "TARGET_PATHWAY"
]

# Build clean working table
model_df = df[feature_cols + [target_col]].copy()

# Basic checks
print("Shape of model_df:", model_df.shape)
print("\nMissing values by column:")
print(model_df.isnull().sum())

print("\nTarget summary:")
print(model_df[target_col].describe())

print("\nUnique values per feature:")
for col in feature_cols:
    print(f"{col}: {model_df[col].nunique(dropna=True)}")