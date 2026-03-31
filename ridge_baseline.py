import pandas as pd
import numpy as np
import time

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv("GDSC_DATASET.csv")

target_col = "LN_IC50"

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

# Keep only needed columns
model_df = df[feature_cols + [target_col]].copy()

# Drop rows with missing target just in case
model_df = model_df.dropna(subset=[target_col])

X = model_df[feature_cols]
y = model_df[target_col]

# -----------------------------
# Train/test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

# -----------------------------
# Preprocessing
# -----------------------------
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", categorical_transformer, feature_cols)
    ]
)

# -----------------------------
# Model pipeline
# -----------------------------
ridge_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", Ridge(alpha=1.0))
])

# -----------------------------
# Train
# -----------------------------
start_train = time.time()
ridge_pipeline.fit(X_train, y_train)
train_time = time.time() - start_train

# -----------------------------
# Predict
# -----------------------------
start_pred = time.time()
y_pred = ridge_pipeline.predict(X_test)
pred_time = time.time() - start_pred

# -----------------------------
# Metrics
# -----------------------------
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nRidge Baseline Results")
print("----------------------")
print(f"RMSE: {rmse:.4f}")
print(f"MAE : {mae:.4f}")
print(f"R^2 : {r2:.4f}")
print(f"Training time  : {train_time:.4f} seconds")
print(f"Inference time : {pred_time:.4f} seconds")

# Optional: transformed feature count
X_train_transformed = ridge_pipeline.named_steps["preprocessor"].transform(X_train)
print("Transformed feature matrix shape:", X_train_transformed.shape)