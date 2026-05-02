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
from sklearn.model_selection import train_test_split
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
SECONDARY_SCREEN_AUC_DATA_PATH = PROJECT_DIR / "data" / "secondary_screen" / "secondary_screen_auc_clean.parquet"
SECONDARY_SCREEN_IC50_DATA_PATH = PROJECT_DIR / "data" / "secondary_screen" / "secondary_screen_ic50_clean.parquet"
RESULTS_DIR = PROJECT_DIR / "results"

GDSC_DATASET_NAME = "gdsc"

SUPPORTED_DATASETS = (
    "gdsc_metadata_only",
    "gdsc_metadata_plus_expression",
    "gdsc_auc_metadata_only",
    "gdsc_auc_metadata_plus_expression",
    "secondary_screen_auc",
    "secondary_screen_ic50",
)
DATASET_LABELS = {
    "gdsc_metadata_only": "GDSC metadata only",
    "gdsc_metadata_plus_expression": "GDSC metadata plus expression",
    "gdsc_auc_metadata_only": "GDSC AUC metadata only",
    "gdsc_auc_metadata_plus_expression": "GDSC AUC metadata plus expression",
    "secondary_screen_auc": "Secondary screen AUC",
    "secondary_screen_ic50": "Secondary screen log IC50",
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
SUPPORTED_SPLITS = ("random",)
SPLIT_LABELS = {
    "random": "Random row split",
}

GDSC_TARGET_COL = "LN_IC50"
GDSC_AUC_TARGET_COL = "AUC"
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
GDSC_RESPONSE_COLS = (GDSC_TARGET_COL, GDSC_AUC_TARGET_COL, "Z_SCORE")
GDSC_IDENTIFIER_COLS = ("COSMIC_ID", "CELL_LINE_NAME", "DRUG_ID", "DRUG_NAME")

SECONDARY_SCREEN_AUC_TARGET_COL = "target_auc"
SECONDARY_SCREEN_IC50_TARGET_COL = "target_log_ic50"
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
    SECONDARY_SCREEN_AUC_TARGET_COL,
    SECONDARY_SCREEN_IC50_TARGET_COL,
)
SECONDARY_SCREEN_IDENTIFIER_COLS = (
    "broad_id",
    "depmap_id",
    "ccle_name",
    "smiles",
    "row_name",
)

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
    raw_shape: tuple[int, int]
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

    if dataset.startswith("secondary_screen_"):
        raise ValueError(f"--feature-set cannot be used with --dataset {dataset}.")

    expected_dataset = LEGACY_FEATURE_SET_TO_DATASET[feature_set]
    if dataset != expected_dataset:
        raise ValueError(
            f"Conflicting dataset/feature-set combination: dataset='{dataset}', feature_set='{feature_set}'."
        )

    return dataset


def feature_set_for_dataset_mode(dataset_mode: str) -> str | None:
    _validate_dataset_mode(dataset_mode)
    if dataset_mode in {"gdsc_metadata_only", "gdsc_auc_metadata_only"}:
        return "metadata_only"
    if dataset_mode in {"gdsc_metadata_plus_expression", "gdsc_auc_metadata_plus_expression"}:
        return "metadata_plus_expression"
    return None


def dataset_name_for_dataset_mode(dataset_mode: str) -> str:
    _validate_dataset_mode(dataset_mode)
    if dataset_mode.startswith("gdsc_"):
        return GDSC_DATASET_NAME
    return dataset_mode


def target_col_for_dataset_mode(dataset_mode: str) -> str:
    _validate_dataset_mode(dataset_mode)
    if dataset_mode.startswith("gdsc_auc_"):
        return GDSC_AUC_TARGET_COL
    if dataset_mode.startswith("gdsc_"):
        return GDSC_TARGET_COL
    if dataset_mode == "secondary_screen_auc":
        return SECONDARY_SCREEN_AUC_TARGET_COL
    return SECONDARY_SCREEN_IC50_TARGET_COL


def build_secondary_screen_ic50_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    ic50_df = df.copy()

    if SECONDARY_SCREEN_IC50_TARGET_COL not in ic50_df.columns:
        if "ic50" in ic50_df.columns:
            ic50_values = pd.to_numeric(ic50_df["ic50"], errors="coerce")
            valid_ic50 = ic50_values.gt(0) & np.isfinite(ic50_values)
            ic50_df = ic50_df.loc[valid_ic50].copy()
            ic50_df[SECONDARY_SCREEN_IC50_TARGET_COL] = np.log10(ic50_values.loc[valid_ic50])
        elif "log_ic50" in ic50_df.columns:
            ic50_df[SECONDARY_SCREEN_IC50_TARGET_COL] = pd.to_numeric(
                ic50_df["log_ic50"],
                errors="coerce",
            )
        else:
            raise ValueError(
                "secondary_screen_ic50 requires one of: target_log_ic50, ic50, or log_ic50."
            )

    ic50_df[SECONDARY_SCREEN_IC50_TARGET_COL] = pd.to_numeric(
        ic50_df[SECONDARY_SCREEN_IC50_TARGET_COL],
        errors="coerce",
    )
    ic50_df = ic50_df[np.isfinite(ic50_df[SECONDARY_SCREEN_IC50_TARGET_COL])].copy()

    return ic50_df


def load_secondary_screen_ic50_dataset() -> pd.DataFrame:
    if SECONDARY_SCREEN_IC50_DATA_PATH.exists():
        return pd.read_parquet(SECONDARY_SCREEN_IC50_DATA_PATH)

    if not SECONDARY_SCREEN_AUC_DATA_PATH.exists():
        raise FileNotFoundError(f"Missing Parquet dataset: {SECONDARY_SCREEN_AUC_DATA_PATH}")

    df = pd.read_parquet(SECONDARY_SCREEN_AUC_DATA_PATH)
    ic50_df = build_secondary_screen_ic50_dataframe(df)
    SECONDARY_SCREEN_IC50_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    ic50_df.to_parquet(SECONDARY_SCREEN_IC50_DATA_PATH, index=False, compression="snappy")
    return ic50_df


def load_dataset(
    dataset: str | None = None,
    feature_set: str | None = None,
) -> tuple[str, pd.DataFrame]:
    dataset_mode = resolve_dataset_mode(dataset=dataset, feature_set=feature_set)

    if dataset_mode in {"gdsc_metadata_only", "gdsc_auc_metadata_only"}:
        return dataset_mode, pd.read_csv(GDSC_METADATA_DATA_PATH)

    if dataset_mode == "secondary_screen_ic50":
        try:
            return dataset_mode, load_secondary_screen_ic50_dataset()
        except ImportError as exc:
            raise ImportError(
                "Parquet support requires 'pyarrow' or 'fastparquet'. "
                "Install 'pyarrow' in the environment used for this project."
            ) from exc

    if dataset_mode in {"gdsc_metadata_plus_expression", "gdsc_auc_metadata_plus_expression"}:
        parquet_path = GDSC_EXPRESSION_DATA_PATH
    elif dataset_mode == "secondary_screen_auc":
        parquet_path = SECONDARY_SCREEN_AUC_DATA_PATH
    else:
        parquet_path = GDSC_EXPRESSION_DATA_PATH

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
        target_col = target_col_for_dataset_mode(dataset_mode)
        leakage_cols = tuple(col for col in GDSC_RESPONSE_COLS if col != target_col)

        if feature_set_for_dataset_mode(dataset_mode) == "metadata_plus_expression":
            excluded_cols = set(categorical_cols)
            excluded_cols.update(GDSC_RESPONSE_COLS)
            excluded_cols.update(GDSC_IDENTIFIER_COLS)

            candidate_cols = [col for col in df.columns if col not in excluded_cols]
            non_numeric_cols = [col for col in candidate_cols if not is_numeric_dtype(df[col])]

            if non_numeric_cols:
                raise ValueError(
                    "Unexpected non-numeric columns in GDSC metadata-plus-expression feature set: "
                    f"{non_numeric_cols}"
                )

            numerical_cols = candidate_cols

        feature_columns = categorical_cols + numerical_cols

        if target_col in feature_columns:
            raise ValueError(f"Target column '{target_col}' must not appear in X.")

        leakage_in_features = [col for col in leakage_cols if col in feature_columns]
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
    secondary_target_col = target_col_for_dataset_mode(dataset_mode)
    excluded_cols.add(secondary_target_col)
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

    if secondary_target_col in feature_columns:
        raise ValueError(f"Target column '{secondary_target_col}' must not appear in X.")

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
) -> ModelingDataset:
    dataset_mode, df = load_dataset(dataset=dataset, feature_set=feature_set)
    raw_shape = df.shape
    dataset_name = dataset_name_for_dataset_mode(dataset_mode)
    resolved_feature_set = feature_set_for_dataset_mode(dataset_mode)
    target_col = target_col_for_dataset_mode(dataset_mode)
    categorical_cols, numerical_cols = get_feature_columns(df, dataset_mode=dataset_mode)
    feature_columns = categorical_cols + numerical_cols

    selected_cols = feature_columns + [target_col]

    model_df = df[selected_cols].copy()
    model_df = model_df.dropna(subset=[target_col])

    return ModelingDataset(
        dataset_mode=dataset_mode,
        dataset_name=dataset_name,
        dataset_label=DATASET_LABELS[dataset_mode],
        target_col=target_col,
        feature_set=resolved_feature_set,
        raw_shape=raw_shape,
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
                # LinearSVR is the feasible SVR baseline for these large sparse
                # one-hot feature matrices, but it may still need tuning by split.
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
    split_strategy: str = "random",
) -> DatasetSplit:
    if split_strategy not in SUPPORTED_SPLITS:
        supported = ", ".join(SUPPORTED_SPLITS)
        raise ValueError(f"Unsupported split strategy '{split_strategy}'. Expected one of: {supported}")

    X = model_df[feature_columns].copy()
    y = model_df[target_col].copy()

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


def regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }
