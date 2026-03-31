import pandas as pd
import numpy as np
import time

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LassoCV
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

model_df = df[feature_cols + [target_col]].copy()
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
lasso_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("scaler", StandardScaler(with_mean=False)),
    ("model", LassoCV(
        alphas=np.logspace(-4, 0, 10),
        cv=5,
        max_iter=10000,
        random_state=42,
        n_jobs=-1
    ))
])

# -----------------------------
# Train
# -----------------------------
start_train = time.time()
lasso_pipeline.fit(X_train, y_train)
train_time = time.time() - start_train

# -----------------------------
# Predict
# -----------------------------
start_pred = time.time()
y_pred = lasso_pipeline.predict(X_test)
pred_time = time.time() - start_pred

# -----------------------------
# Metrics
# -----------------------------
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

best_alpha = lasso_pipeline.named_steps["model"].alpha_

print("\nLASSO Results")
print("-------------")
print(f"Best alpha: {best_alpha:.6f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE : {mae:.4f}")
print(f"R^2 : {r2:.4f}")
print(f"Training time  : {train_time:.4f} seconds")
print(f"Inference time : {pred_time:.4f} seconds")

# -----------------------------
# Count nonzero coefficients
# -----------------------------
coef = lasso_pipeline.named_steps["model"].coef_
nonzero = np.sum(coef != 0)
total = len(coef)

print(f"Nonzero coefficients: {nonzero} / {total}")