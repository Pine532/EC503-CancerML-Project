from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "GDSC_DATASET.csv"

TARGET_COL = "LN_IC50"
FEATURE_COLS = [
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
    "TARGET_PATHWAY",
]

GROUP_SPLIT_COLUMNS = {
    "drug": "DRUG_ID",
    "cell_line": "COSMIC_ID",
}

SUPPORTED_SPLITS = ("random", "drug", "cell_line")
SPLIT_LABELS = {
    "random": "Random row split",
    "drug": "Grouped-by-drug split",
    "cell_line": "Grouped-by-cell-line split",
}

RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE_WITHIN_TRAIN = 0.25


@dataclass
class DatasetSplit:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series


def load_raw_dataset(data_path: Path = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(data_path)


def load_model_dataframe(
    data_path: Path = DATA_PATH,
    include_group_columns: bool = False,
) -> pd.DataFrame:
    df = load_raw_dataset(data_path)
    selected_cols = FEATURE_COLS + [TARGET_COL]

    if include_group_columns:
        selected_cols = list(GROUP_SPLIT_COLUMNS.values()) + selected_cols

    model_df = df[selected_cols].copy()
    model_df = model_df.dropna(subset=[TARGET_COL])
    return model_df


def build_preprocessor() -> ColumnTransformer:
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, FEATURE_COLS),
        ]
    )


def _to_dense_if_needed(matrix):
    return matrix.toarray() if hasattr(matrix, "toarray") else matrix


def gradient_boosting_backend_name() -> str:
    if XGBRegressor is not None:
        return "XGBoost"

    return "sklearn HistGradientBoostingRegressor"


def build_linear_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", LinearRegression()),
        ]
    )


def build_ridge_pipeline(alpha: float = 1.0) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", Ridge(alpha=alpha)),
        ]
    )


def build_lasso_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("scaler", StandardScaler(with_mean=False)),
            (
                "model",
                LassoCV(
                    alphas=np.logspace(-4, 0, 6),
                    cv=3,
                    max_iter=10000,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def build_random_forest_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def build_gradient_boosting_pipeline() -> Pipeline:
    steps = [("preprocessor", build_preprocessor())]

    if XGBRegressor is not None:
        steps.append(
            (
                "model",
                XGBRegressor(
                    objective="reg:squarederror",
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=5,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    tree_method="hist",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            )
        )
        return Pipeline(steps=steps)

    return Pipeline(
        steps=[
            *steps,
            # sklearn's histogram booster expects dense input after one-hot encoding.
            ("to_dense", FunctionTransformer(_to_dense_if_needed, validate=False)),
            (
                "model",
                HistGradientBoostingRegressor(
                    learning_rate=0.05,
                    max_iter=200,
                    max_depth=5,
                    max_features=0.8,
                    early_stopping=False,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def build_model_registry() -> dict[str, Pipeline]:
    return {
        "Linear": build_linear_pipeline(),
        "Ridge": build_ridge_pipeline(),
        "Lasso": build_lasso_pipeline(),
        "Random Forest": build_random_forest_pipeline(),
        "Gradient Boosting": build_gradient_boosting_pipeline(),
    }


def split_dataset(model_df: pd.DataFrame, split_strategy: str = "random") -> DatasetSplit:
    if split_strategy not in SUPPORTED_SPLITS:
        supported = ", ".join(SUPPORTED_SPLITS)
        raise ValueError(f"Unsupported split strategy '{split_strategy}'. Expected one of: {supported}")

    X = model_df[FEATURE_COLS].copy()
    y = model_df[TARGET_COL].copy()

    if split_strategy == "random":
        X_temp, X_test, y_temp, y_test = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp,
            y_temp,
            test_size=VAL_SIZE_WITHIN_TRAIN,
            random_state=RANDOM_STATE,
        )
        return DatasetSplit(X_train, X_val, X_test, y_train, y_val, y_test)

    group_column = GROUP_SPLIT_COLUMNS[split_strategy]
    groups = model_df[group_column]
    train_val_idx, test_idx = next(
        GroupShuffleSplit(
            n_splits=1,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
        ).split(X, y, groups)
    )

    X_train_val = X.iloc[train_val_idx].copy()
    y_train_val = y.iloc[train_val_idx].copy()
    groups_train_val = groups.iloc[train_val_idx].copy()

    train_idx_rel, val_idx_rel = next(
        GroupShuffleSplit(
            n_splits=1,
            test_size=VAL_SIZE_WITHIN_TRAIN,
            random_state=RANDOM_STATE,
        ).split(X_train_val, y_train_val, groups_train_val)
    )

    return DatasetSplit(
        X_train=X_train_val.iloc[train_idx_rel].copy(),
        X_val=X_train_val.iloc[val_idx_rel].copy(),
        X_test=X.iloc[test_idx].copy(),
        y_train=y_train_val.iloc[train_idx_rel].copy(),
        y_val=y_train_val.iloc[val_idx_rel].copy(),
        y_test=y.iloc[test_idx].copy(),
    )


def regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }
