from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.compose import TransformedTargetRegressor

from cancer_ml_utils import (
    RANDOM_STATE,
    RESULTS_DIR,
    SPLIT_LABELS,
    SUPPORTED_DATASETS,
    SUPPORTED_SPLITS,
    build_dummy_mean_pipeline,
    build_gradient_boosting_pipeline,
    build_linear_pipeline,
    build_linear_svr_pipeline,
    build_preprocessor,
    build_random_forest_pipeline,
    build_ridge_pipeline,
    gradient_boosting_backend_name,
    load_model_dataframe,
    regression_metrics,
    split_dataset,
)


MODEL_ARG_TO_NAME = {
    "Dummy Mean": "Dummy Mean",
    "Linear": "Linear",
    "Ridge": "Ridge",
    "Linear SVR": "Linear SVR",
    "Random Forest": "Random Forest",
    "Gradient Boosting": "Gradient Boosting",
    "Neural Network": "Neural Network",
    "dummy_mean": "Dummy Mean",
    "linear": "Linear",
    "ridge": "Ridge",
    "linear_svr": "Linear SVR",
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
    "neural_network": "Neural Network",
    "DummyMean": "Dummy Mean",
    "LinearSVR": "Linear SVR",
    "RandomForest": "Random Forest",
    "GradientBoosting": "Gradient Boosting",
    "NeuralNetwork": "Neural Network",
}

DEFAULT_MODELS = (
    "Dummy Mean",
    "Linear",
    "Ridge",
    "Linear SVR",
    "Gradient Boosting",
)


def to_dense_float32(matrix):
    if sparse.issparse(matrix):
        return matrix.toarray().astype(np.float32)
    return np.asarray(matrix, dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune non-Lasso EC503 cancer drug sensitivity models."
    )
    parser.add_argument("--dataset", choices=SUPPORTED_DATASETS, required=True)
    parser.add_argument("--split", choices=SUPPORTED_SPLITS, default="random")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=tuple(MODEL_ARG_TO_NAME),
        default=None,
        help="Optional model subset. Lasso is intentionally excluded from this tuner.",
    )
    parser.add_argument(
        "--tuning-max-rows",
        type=int,
        default=50000,
        help="Row cap sampled from the training split for hyperparameter search.",
    )
    parser.add_argument(
        "--final-train-max-rows",
        type=int,
        default=None,
        help="Optional row cap for final refit after tuning. Default uses the full training split.",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=12,
        help="RandomizedSearchCV iterations for tunable models.",
    )
    parser.add_argument("--cv", type=int, default=3)
    parser.add_argument("--n-jobs", type=int, default=-1)
    return parser.parse_args()


def selected_model_names(model_args: list[str] | None) -> list[str]:
    if model_args is None:
        return list(DEFAULT_MODELS)
    return list(dict.fromkeys(MODEL_ARG_TO_NAME[arg] for arg in model_args))


def sample_rows(X: pd.DataFrame, y: pd.Series, max_rows: int | None) -> tuple[pd.DataFrame, pd.Series]:
    if max_rows is None or len(X) <= max_rows:
        return X, y
    sampled_index = X.sample(n=max_rows, random_state=RANDOM_STATE).index
    return X.loc[sampled_index].copy(), y.loc[sampled_index].copy()


def build_neural_network_estimator(categorical_cols: list[str], numerical_cols: list[str]):
    regressor = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(categorical_cols, numerical_cols)),
            ("to_dense", FunctionTransformer(to_dense_float32, validate=False)),
            (
                "model",
                MLPRegressor(
                    hidden_layer_sizes=(256, 128),
                    activation="relu",
                    solver="adam",
                    alpha=1e-4,
                    batch_size=2048,
                    learning_rate_init=1e-3,
                    max_iter=30,
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=5,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    return TransformedTargetRegressor(
        regressor=regressor,
        transformer=StandardScaler(),
    )


def build_estimator_and_params(
    model_name: str,
    categorical_cols: list[str],
    numerical_cols: list[str],
) -> tuple[object, dict[str, list[object]]]:
    if model_name == "Dummy Mean":
        return build_dummy_mean_pipeline(categorical_cols, numerical_cols), {}

    if model_name == "Linear":
        return build_linear_pipeline(categorical_cols, numerical_cols), {}

    if model_name == "Ridge":
        return build_ridge_pipeline(categorical_cols, numerical_cols), {
            "model__alpha": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0],
        }

    if model_name == "Linear SVR":
        return build_linear_svr_pipeline(categorical_cols, numerical_cols), {
            "model__C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0],
            "model__epsilon": [0.0, 0.001, 0.01, 0.03, 0.1],
            "model__tol": [1e-4, 1e-3],
        }

    if model_name == "Random Forest":
        return build_random_forest_pipeline(categorical_cols, numerical_cols), {
            "model__n_estimators": [100, 200, 400],
            "model__max_depth": [8, 12, 16, 24, None],
            "model__min_samples_leaf": [1, 2, 5, 10],
            "model__min_samples_split": [2, 5, 10],
            "model__max_features": ["sqrt", "log2", 0.5, 0.8, 1.0],
        }

    if model_name == "Gradient Boosting":
        estimator = build_gradient_boosting_pipeline(categorical_cols, numerical_cols)
        if gradient_boosting_backend_name() == "XGBoost":
            return estimator, {
                "model__n_estimators": [100, 200, 300, 500],
                "model__learning_rate": [0.03, 0.05, 0.08, 0.1],
                "model__max_depth": [3, 4, 5, 6],
                "model__subsample": [0.7, 0.8, 0.9, 1.0],
                "model__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
                "model__reg_lambda": [0.1, 1.0, 3.0, 10.0],
            }
        return estimator, {
            "model__max_iter": [100, 200, 300, 500],
            "model__learning_rate": [0.03, 0.05, 0.08, 0.1],
            "model__max_depth": [3, 5, 8, None],
            "model__l2_regularization": [0.0, 0.01, 0.1, 1.0],
            "model__max_features": [0.5, 0.8, 1.0],
        }

    if model_name == "Neural Network":
        return build_neural_network_estimator(categorical_cols, numerical_cols), {
            "regressor__model__hidden_layer_sizes": [
                (128,),
                (256,),
                (256, 128),
                (512, 256),
            ],
            "regressor__model__alpha": [1e-5, 1e-4, 1e-3],
            "regressor__model__learning_rate_init": [3e-4, 1e-3, 3e-3],
            "regressor__model__batch_size": [1024, 2048, 4096],
            "regressor__model__max_iter": [20, 40, 60],
        }

    raise ValueError(f"Unsupported model for tuning: {model_name}")


def run_tuning(
    estimator,
    param_distributions: dict[str, list[object]],
    X_tune: pd.DataFrame,
    y_tune: pd.Series,
    n_iter: int,
    cv: int,
    n_jobs: int,
) -> tuple[object, dict, float, float, pd.DataFrame]:
    if not param_distributions:
        estimator.fit(X_tune, y_tune)
        return estimator, {}, np.nan, np.nan, pd.DataFrame()

    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=n_jobs,
        verbose=1,
        refit=True,
    )
    search.fit(X_tune, y_tune)
    cv_results = pd.DataFrame(search.cv_results_)
    return search.best_estimator_, search.best_params_, -search.best_score_, search.best_index_, cv_results


def build_output_paths(
    dataset_mode: str,
    split_key: str,
    model_names: list[str],
    tuning_max_rows: int | None,
    n_iter: int,
) -> tuple[Path, Path]:
    model_slug = "-".join(name.lower().replace(" ", "_") for name in model_names)
    row_part = "full" if tuning_max_rows is None else f"tune{tuning_max_rows}"
    stem = f"optimized_model_comparison_{dataset_mode}_{split_key}_{model_slug}_{row_part}_niter{n_iter}"
    return RESULTS_DIR / f"{stem}.csv", RESULTS_DIR / f"{stem}_best_params.csv"


def cv_results_path(results_path: Path, model_name: str) -> Path:
    model_slug = model_name.lower().replace(" ", "_")
    return results_path.with_name(f"{results_path.stem}_{model_slug}_cv_results.csv")


def main() -> None:
    args = parse_args()
    model_names = selected_model_names(args.models)

    model_data = load_model_dataframe(
        dataset=args.dataset,
    )
    data_split = split_dataset(
        model_data.model_df,
        feature_columns=model_data.feature_columns,
        target_col=model_data.target_col,
        split_strategy=args.split,
    )
    X_tune, y_tune = sample_rows(data_split.X_train, data_split.y_train, args.tuning_max_rows)
    X_final, y_final = sample_rows(
        data_split.X_train,
        data_split.y_train,
        args.final_train_max_rows,
    )

    results_path, params_path = build_output_paths(
        dataset_mode=model_data.dataset_mode,
        split_key=args.split,
        model_names=model_names,
        tuning_max_rows=args.tuning_max_rows,
        n_iter=args.n_iter,
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Dataset:", model_data.dataset_name)
    print("Dataset mode:", model_data.dataset_mode)
    print("Target:", model_data.target_col)
    print("Split strategy:", SPLIT_LABELS[args.split], f"({args.split})")
    print("Raw dataset shape:", model_data.raw_shape)
    print("Modeling dataset shape:", model_data.model_df.shape)
    print("Categorical feature count:", len(model_data.categorical_cols))
    print("Numerical feature count:", len(model_data.numerical_cols))
    print("X_train shape:", data_split.X_train.shape)
    print("X_tune shape:", X_tune.shape)
    print("X_final shape:", X_final.shape)
    print("X_val shape:", data_split.X_val.shape)
    print("X_test shape:", data_split.X_test.shape)
    print("Models:", ", ".join(model_names))
    print("Gradient Boosting backend:", gradient_boosting_backend_name())
    print("Results path:", results_path)
    print("Best params path:", params_path)

    result_rows = []
    param_rows = []
    trained_models = {}

    for model_name in model_names:
        print(f"\nOptimizing {model_name}...")
        estimator, param_distributions = build_estimator_and_params(
            model_name,
            model_data.categorical_cols,
            model_data.numerical_cols,
        )

        tune_start = time.time()
        convergence_warnings: list[str] = []
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always", ConvergenceWarning)
            best_estimator, best_params, best_cv_rmse, best_index, cv_results = run_tuning(
                estimator=estimator,
                param_distributions=param_distributions,
                X_tune=X_tune,
                y_tune=y_tune,
                n_iter=args.n_iter,
                cv=args.cv,
                n_jobs=args.n_jobs,
            )
            if not cv_results.empty:
                cv_results.to_csv(cv_results_path(results_path, model_name), index=False)
        for caught in caught_warnings:
            if issubclass(caught.category, ConvergenceWarning):
                convergence_warnings.append(str(caught.message))
        tune_seconds = time.time() - tune_start

        final_estimator = clone(best_estimator)
        final_start = time.time()
        final_estimator.fit(X_final, y_final)
        final_train_seconds = time.time() - final_start

        val_start = time.time()
        val_pred = final_estimator.predict(data_split.X_val)
        predict_seconds = time.time() - val_start
        val_metrics = regression_metrics(data_split.y_val, val_pred)

        test_pred = final_estimator.predict(data_split.X_test)
        test_metrics = regression_metrics(data_split.y_test, test_pred)

        warning_text = " | ".join(dict.fromkeys(convergence_warnings))
        result = {
            "Dataset": model_data.dataset_name,
            "Dataset Mode": model_data.dataset_mode,
            "Target": model_data.target_col,
            "Split": args.split,
            "Feature Set": "" if model_data.feature_set is None else model_data.feature_set,
            "Model": model_name,
            "Tuning Max Rows": "" if args.tuning_max_rows is None else args.tuning_max_rows,
            "Final Train Max Rows": "" if args.final_train_max_rows is None else args.final_train_max_rows,
            "CV Folds": args.cv if param_distributions else "",
            "N Iter": args.n_iter if param_distributions else "",
            "Best CV RMSE": best_cv_rmse,
            "Validation RMSE": val_metrics["RMSE"],
            "Validation MAE": val_metrics["MAE"],
            "Validation R2": val_metrics["R2"],
            "Test RMSE": test_metrics["RMSE"],
            "Test MAE": test_metrics["MAE"],
            "Test R2": test_metrics["R2"],
            "Tune Seconds": tune_seconds,
            "Final Train Seconds": final_train_seconds,
            "Predict Seconds": predict_seconds,
            "Convergence Warnings": warning_text,
        }
        result_rows.append(result)
        trained_models[model_name] = final_estimator

        param_rows.append(
            {
                "Dataset Mode": model_data.dataset_mode,
                "Split": args.split,
                "Model": model_name,
                "Tuned": bool(param_distributions),
                "Best CV RMSE": best_cv_rmse,
                "Best Search Index": best_index,
                "Best Params JSON": json.dumps(best_params, sort_keys=True),
                "Param Search Space JSON": json.dumps(param_distributions, default=str, sort_keys=True),
                "Convergence Warnings": warning_text,
            }
        )

        pd.DataFrame(result_rows).sort_values(
            by=["Validation RMSE", "Validation MAE", "Validation R2"],
            ascending=[True, True, False],
        ).to_csv(results_path, index=False)
        pd.DataFrame(param_rows).to_csv(params_path, index=False)

        print(
            f"{model_name}: val RMSE={val_metrics['RMSE']:.4f}, "
            f"val MAE={val_metrics['MAE']:.4f}, val R2={val_metrics['R2']:.4f}, "
            f"test R2={test_metrics['R2']:.4f}, tune={tune_seconds:.1f}s"
        )
        if best_params:
            print("Best params:", best_params)
        if warning_text:
            print("Warnings:", warning_text)

    results_df = pd.DataFrame(result_rows).sort_values(
        by=["Validation RMSE", "Validation MAE", "Validation R2"],
        ascending=[True, True, False],
    ).reset_index(drop=True)
    results_df.to_csv(results_path, index=False)
    pd.DataFrame(param_rows).to_csv(params_path, index=False)

    print("\nOptimized Validation/Test Results")
    print("---------------------------------")
    print(results_df.to_string(index=False))
    print("\nSaved optimized results to:", results_path)
    print("Saved best parameters to:", params_path)


if __name__ == "__main__":
    main()
