from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVR

try:
    from xgboost import XGBRegressor
    XGBOOST_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - environment-dependent import path
    XGBRegressor = None
    XGBOOST_IMPORT_ERROR = exc

PROJECT_DIR = Path(__file__).resolve().parent
GDSC_METADATA_DATA_PATH = PROJECT_DIR / "GDSC_DATASET.csv"
GDSC_EXPRESSION_DATA_PATH = PROJECT_DIR / "data" / "model_ready" / "gdsc_metadata_expression_top500.parquet"
SECONDARY_SCREEN_DATA_PATH = PROJECT_DIR / "data" / "secondary_screen" / "secondary_screen_auc_clean.parquet"
RESULTS_DIR = PROJECT_DIR / "results"

GDSC_DATASET_NAME = "gdsc"
SECONDARY_SCREEN_DATASET_NAME = "secondary_screen_auc"

SUPPORTED_DATASETS = (
    "gdsc_metadata_only",
    "gdsc_metadata_plus_expression",
    "secondary_screen_auc",
)
DATASET_LABELS = {
    "gdsc_metadata_only": "GDSC metadata only",
    "gdsc_metadata_plus_expression": "GDSC metadata plus expression",
    "secondary_screen_auc": "Secondary screen AUC",
}
SUPPORTED_FEATURE_SETS = ("metadata_only", "metadata_plus_expression")
FEATURE_SET_LABELS = {
    "metadata_only": "Metadata only",
    "metadata_plus_expression": "Metadata plus expression",
}
LEGACY_FEATURE_SET_TO_DATASET = {
    "metadata_only": "gdsc_metadata_only",
    "metadata_plus_expression": "gdsc_metadata_plus_expression",
}
SUPPORTED_SPLITS = ("random", "drug", "cell_line")
SPLIT_LABELS = {
    "random": "Random row split",
    "drug": "Grouped-by-drug split",
    "cell_line": "Grouped-by-cell-line split",
}

GDSC_TARGET_COL = "LN_IC50"
GDSC_METADATA_FEATURE_COLS = [
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
GDSC_LEAKAGE_COLS = ("AUC", "Z_SCORE")
GDSC_IDENTIFIER_COLS = ("COSMIC_ID", "CELL_LINE_NAME", "DRUG_ID", "DRUG_NAME")
GDSC_GROUP_SPLIT_COLUMNS = {
    "drug": "DRUG_ID",
    "cell_line": "COSMIC_ID",
}

SECONDARY_SCREEN_TARGET_COL = "target_auc"
SECONDARY_SCREEN_CATEGORICAL_FEATURE_COLS = [
    "ccle_tissue",
    "screen_id",
    "name",
    "moa",
    "target",
    "disease.area",
    "indication",
    "phase",
]
SECONDARY_SCREEN_LEAKAGE_COLS = (
    "auc",
    "upper_limit",
    "lower_limit",
    "slope",
    "r2",
    "ec50",
    "ic50",
    "log_ic50",
    "log_ec50",
)
SECONDARY_SCREEN_IDENTIFIER_COLS = (
    "broad_id",
    "depmap_id",
    "ccle_name",
    "smiles",
    "row_name",
)
SECONDARY_SCREEN_GROUP_SPLIT_COLUMNS = {
    "drug": "broad_id",
    "cell_line": "depmap_id",
}

RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE_WITHIN_TRAIN = 0.25


@dataclass
class ModelingDataset:
    dataset_mode: str
    dataset_name: str
    dataset_label: str
    target_col: str
    feature_set: str | None
    group_split_columns: dict[str, str]
    model_df: pd.DataFrame
    categorical_cols: list[str]
    numerical_cols: list[str]
    feature_columns: list[str]


@dataclass
class DatasetSplit:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series


def _to_dense_if_needed(matrix):
    return matrix.toarray() if hasattr(matrix, "toarray") else matrix


def _validate_feature_set(feature_set: str) -> None:
    if feature_set not in SUPPORTED_FEATURE_SETS:
        supported = ", ".join(SUPPORTED_FEATURE_SETS)
        raise ValueError(f"Unsupported feature set '{feature_set}'. Expected one of: {supported}")


def _validate_dataset_mode(dataset_mode: str) -> None:
    if dataset_mode not in SUPPORTED_DATASETS:
        supported = ", ".join(SUPPORTED_DATASETS)
        raise ValueError(f"Unsupported dataset '{dataset_mode}'. Expected one of: {supported}")


def resolve_dataset_mode(
    dataset: str | None = None,
    feature_set: str | None = None,
) -> str:
    if dataset is not None:
        _validate_dataset_mode(dataset)

    if feature_set is not None:
        _validate_feature_set(feature_set)

    if dataset is None and feature_set is None:
        return "gdsc_metadata_only"

    if dataset is None:
        return LEGACY_FEATURE_SET_TO_DATASET[feature_set]

    if feature_set is None:
        return dataset

    if dataset == "secondary_screen_auc":
        raise ValueError("--feature-set cannot be used with --dataset secondary_screen_auc.")

    expected_dataset = LEGACY_FEATURE_SET_TO_DATASET[feature_set]
    if dataset != expected_dataset:
        raise ValueError(
            f"Conflicting dataset/feature-set combination: dataset='{dataset}', feature_set='{feature_set}'."
        )

    return dataset


def feature_set_for_dataset_mode(dataset_mode: str) -> str | None:
    _validate_dataset_mode(dataset_mode)
    if dataset_mode.startswith("gdsc_"):
        return dataset_mode.removeprefix("gdsc_")
    return None


def dataset_name_for_dataset_mode(dataset_mode: str) -> str:
    _validate_dataset_mode(dataset_mode)
    if dataset_mode.startswith("gdsc_"):
        return GDSC_DATASET_NAME
    return SECONDARY_SCREEN_DATASET_NAME


def target_col_for_dataset_mode(dataset_mode: str) -> str:
    _validate_dataset_mode(dataset_mode)
    if dataset_mode.startswith("gdsc_"):
        return GDSC_TARGET_COL
    return SECONDARY_SCREEN_TARGET_COL


def group_split_columns_for_dataset_mode(dataset_mode: str) -> dict[str, str]:
    _validate_dataset_mode(dataset_mode)
    if dataset_mode.startswith("gdsc_"):
        return GDSC_GROUP_SPLIT_COLUMNS
    return SECONDARY_SCREEN_GROUP_SPLIT_COLUMNS


def load_dataset(
    dataset: str | None = None,
    feature_set: str | None = None,
) -> tuple[str, pd.DataFrame]:
    dataset_mode = resolve_dataset_mode(dataset=dataset, feature_set=feature_set)

    if dataset_mode == "gdsc_metadata_only":
        return dataset_mode, pd.read_csv(GDSC_METADATA_DATA_PATH)

    parquet_path = (
        GDSC_EXPRESSION_DATA_PATH
        if dataset_mode == "gdsc_metadata_plus_expression"
        else SECONDARY_SCREEN_DATA_PATH
    )

    if not parquet_path.exists():
        raise FileNotFoundError(f"Missing Parquet dataset: {parquet_path}")

    try:
        return dataset_mode, pd.read_parquet(parquet_path)
    except ImportError as exc:
        raise ImportError(
            "Parquet support requires 'pyarrow' or 'fastparquet'. "
            "Install 'pyarrow' in the environment used for this project."
        ) from exc


def get_feature_columns(
    df: pd.DataFrame,
    dataset_mode: str,
) -> tuple[list[str], list[str]]:
    _validate_dataset_mode(dataset_mode)

    if dataset_mode.startswith("gdsc_"):
        missing_metadata_cols = [col for col in GDSC_METADATA_FEATURE_COLS if col not in df.columns]
        if missing_metadata_cols:
            raise ValueError(f"Dataset is missing required metadata columns: {missing_metadata_cols}")

        categorical_cols = list(GDSC_METADATA_FEATURE_COLS)
        numerical_cols: list[str] = []

        if dataset_mode == "gdsc_metadata_plus_expression":
            excluded_cols = set(categorical_cols)
            excluded_cols.add(GDSC_TARGET_COL)
            excluded_cols.update(GDSC_LEAKAGE_COLS)
            excluded_cols.update(GDSC_IDENTIFIER_COLS)

            candidate_cols = [col for col in df.columns if col not in excluded_cols]
            non_numeric_cols = [col for col in candidate_cols if not is_numeric_dtype(df[col])]

            if non_numeric_cols:
                raise ValueError(
                    "Unexpected non-numeric columns in gdsc_metadata_plus_expression feature set: "
                    f"{non_numeric_cols}"
                )

            numerical_cols = candidate_cols

        feature_columns = categorical_cols + numerical_cols

        if GDSC_TARGET_COL in feature_columns:
            raise ValueError(f"Target column '{GDSC_TARGET_COL}' must not appear in X.")

        leakage_in_features = [col for col in GDSC_LEAKAGE_COLS if col in feature_columns]
        if leakage_in_features:
            raise ValueError(f"Leakage columns detected in X: {leakage_in_features}")

        identifiers_in_features = [col for col in GDSC_IDENTIFIER_COLS if col in feature_columns]
        if identifiers_in_features:
            raise ValueError(f"Identifier columns detected in X: {identifiers_in_features}")

        return categorical_cols, numerical_cols

    missing_secondary_cols = [
        col for col in SECONDARY_SCREEN_CATEGORICAL_FEATURE_COLS if col not in df.columns
    ]
    if missing_secondary_cols:
        raise ValueError(f"Dataset is missing required secondary-screen columns: {missing_secondary_cols}")

    categorical_cols = list(SECONDARY_SCREEN_CATEGORICAL_FEATURE_COLS)
    excluded_cols = set(categorical_cols)
    excluded_cols.add(SECONDARY_SCREEN_TARGET_COL)
    excluded_cols.update(SECONDARY_SCREEN_LEAKAGE_COLS)
    excluded_cols.update(SECONDARY_SCREEN_IDENTIFIER_COLS)

    candidate_cols = [col for col in df.columns if col not in excluded_cols]
    numerical_cols = [col for col in candidate_cols if is_numeric_dtype(df[col])]
    unexpected_non_numeric_cols = [col for col in candidate_cols if col not in numerical_cols]

    if unexpected_non_numeric_cols:
        raise ValueError(
            "Unexpected non-numeric secondary-screen columns outside the declared categorical set: "
            f"{unexpected_non_numeric_cols}"
        )

    feature_columns = categorical_cols + numerical_cols

    if SECONDARY_SCREEN_TARGET_COL in feature_columns:
        raise ValueError(f"Target column '{SECONDARY_SCREEN_TARGET_COL}' must not appear in X.")

    leakage_in_features = [col for col in SECONDARY_SCREEN_LEAKAGE_COLS if col in feature_columns]
    if leakage_in_features:
        raise ValueError(f"Leakage columns detected in X: {leakage_in_features}")

    identifiers_in_features = [col for col in SECONDARY_SCREEN_IDENTIFIER_COLS if col in feature_columns]
    if identifiers_in_features:
        raise ValueError(f"Identifier columns detected in X: {identifiers_in_features}")

    return categorical_cols, numerical_cols


def load_model_dataframe(
    dataset: str | None = None,
    feature_set: str | None = None,
    include_group_columns: bool = False,
) -> ModelingDataset:
    dataset_mode, df = load_dataset(dataset=dataset, feature_set=feature_set)
    dataset_name = dataset_name_for_dataset_mode(dataset_mode)
    resolved_feature_set = feature_set_for_dataset_mode(dataset_mode)
    target_col = target_col_for_dataset_mode(dataset_mode)
    group_split_columns = group_split_columns_for_dataset_mode(dataset_mode)
    categorical_cols, numerical_cols = get_feature_columns(df, dataset_mode=dataset_mode)
    feature_columns = categorical_cols + numerical_cols

    selected_cols = feature_columns + [target_col]
    if include_group_columns:
        selected_cols = list(dict.fromkeys([*group_split_columns.values(), *selected_cols]))

    model_df = df[selected_cols].copy()
    model_df = model_df.dropna(subset=[target_col])

    return ModelingDataset(
        dataset_mode=dataset_mode,
        dataset_name=dataset_name,
        dataset_label=DATASET_LABELS[dataset_mode],
        target_col=target_col,
        feature_set=resolved_feature_set,
        group_split_columns=group_split_columns,
        model_df=model_df,
        categorical_cols=categorical_cols,
        numerical_cols=numerical_cols,
        feature_columns=feature_columns,
    )


def build_preprocessor(
    categorical_cols: list[str],
    numerical_cols: list[str],
) -> ColumnTransformer:
    transformers = []

    if categorical_cols:
        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(("cat", categorical_transformer, categorical_cols))

    if numerical_cols:
        numerical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("num", numerical_transformer, numerical_cols))

    return ColumnTransformer(transformers=transformers)


def build_dummy_mean_pipeline(
    categorical_cols: list[str],
    numerical_cols: list[str],
) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(categorical_cols, numerical_cols)),
            ("model", DummyRegressor(strategy="mean")),
        ]
    )


def build_linear_pipeline(
    categorical_cols: list[str],
    numerical_cols: list[str],
) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(categorical_cols, numerical_cols)),
            ("model", LinearRegression()),
        ]
    )


def build_ridge_pipeline(
    categorical_cols: list[str],
    numerical_cols: list[str],
    alpha: float = 1.0,
) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(categorical_cols, numerical_cols)),
            ("model", Ridge(alpha=alpha)),
        ]
    )


def build_lasso_pipeline(
    categorical_cols: list[str],
    numerical_cols: list[str],
) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(categorical_cols, numerical_cols)),
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


def build_linear_svr_pipeline(
    categorical_cols: list[str],
    numerical_cols: list[str],
) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(categorical_cols, numerical_cols)),
            ("scaler", StandardScaler(with_mean=False)),
            (
                "model",
                LinearSVR(
                    C=1.0,
                    epsilon=0.0,
                    max_iter=10000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def build_random_forest_pipeline(
    categorical_cols: list[str],
    numerical_cols: list[str],
) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(categorical_cols, numerical_cols)),
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


def gradient_boosting_backend_name() -> str:
    if XGBRegressor is not None:
        return "XGBoost"

    return "sklearn HistGradientBoostingRegressor"


def gradient_boosting_backend_detail() -> str | None:
    if XGBRegressor is not None or XGBOOST_IMPORT_ERROR is None:
        return None

    return f"{type(XGBOOST_IMPORT_ERROR).__name__}: {XGBOOST_IMPORT_ERROR}"


def build_gradient_boosting_pipeline(
    categorical_cols: list[str],
    numerical_cols: list[str],
) -> Pipeline:
    steps = [("preprocessor", build_preprocessor(categorical_cols, numerical_cols))]

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


def build_model_registry(
    categorical_cols: list[str],
    numerical_cols: list[str],
) -> dict[str, Pipeline]:
    return {
        "Dummy Mean": build_dummy_mean_pipeline(categorical_cols, numerical_cols),
        "Linear": build_linear_pipeline(categorical_cols, numerical_cols),
        "Ridge": build_ridge_pipeline(categorical_cols, numerical_cols),
        "Lasso": build_lasso_pipeline(categorical_cols, numerical_cols),
        "Linear SVR": build_linear_svr_pipeline(categorical_cols, numerical_cols),
        "Random Forest": build_random_forest_pipeline(categorical_cols, numerical_cols),
        "Gradient Boosting": build_gradient_boosting_pipeline(categorical_cols, numerical_cols),
    }


def split_dataset(
    model_df: pd.DataFrame,
    feature_columns: list[str],
    target_col: str,
    group_split_columns: dict[str, str],
    split_strategy: str = "random",
) -> DatasetSplit:
    if split_strategy not in SUPPORTED_SPLITS:
        supported = ", ".join(SUPPORTED_SPLITS)
        raise ValueError(f"Unsupported split strategy '{split_strategy}'. Expected one of: {supported}")

    X = model_df[feature_columns].copy()
    y = model_df[target_col].copy()

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

    group_column = group_split_columns[split_strategy]
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
