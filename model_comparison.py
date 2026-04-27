from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from cancer_ml_utils import (
    RESULTS_DIR,
    SPLIT_LABELS,
    SUPPORTED_DATASETS,
    SUPPORTED_FEATURE_SETS,
    SUPPORTED_SPLITS,
    build_model_registry,
    gradient_boosting_backend_detail,
    gradient_boosting_backend_name,
    load_model_dataframe,
    regression_metrics,
    resolve_dataset_mode,
    split_dataset,
)


MODEL_ARG_TO_NAME = {
    "Dummy Mean": "Dummy Mean",
    "Linear": "Linear",
    "Ridge": "Ridge",
    "Lasso": "Lasso",
    "Linear SVR": "Linear SVR",
    "Random Forest": "Random Forest",
    "Gradient Boosting": "Gradient Boosting",
    "dummy_mean": "Dummy Mean",
    "linear": "Linear",
    "ridge": "Ridge",
    "lasso": "Lasso",
    "linear_svr": "Linear SVR",
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
    "DummyMean": "Dummy Mean",
    "LinearSVR": "Linear SVR",
    "RandomForest": "Random Forest",
    "GradientBoosting": "Gradient Boosting",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare regressors on EC503 cancer drug sensitivity dataset modes."
    )
    parser.add_argument(
        "--dataset",
        choices=SUPPORTED_DATASETS,
        default=None,
        help=(
            "Dataset mode to evaluate. Defaults to gdsc_metadata_only when omitted. "
            "Use gdsc_metadata_only, gdsc_metadata_plus_expression, or secondary_screen_auc."
        ),
    )
    parser.add_argument(
        "--feature-set",
        choices=SUPPORTED_FEATURE_SETS,
        default=None,
        help="Legacy GDSC-only compatibility alias for --dataset.",
    )
    parser.add_argument(
        "--split",
        choices=SUPPORTED_SPLITS,
        default="random",
        help="Evaluation split to use.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=tuple(MODEL_ARG_TO_NAME),
        default=None,
        help="Optional subset of models to run.",
    )
    return parser.parse_args()


def get_selected_models(
    all_models: dict[str, object],
    model_args: list[str] | None,
) -> dict[str, object]:
    if model_args is None:
        return all_models

    selected_names = list(dict.fromkeys(MODEL_ARG_TO_NAME[arg] for arg in model_args))
    return {name: all_models[name] for name in selected_names}


def build_results_path(
    dataset_name: str,
    split_key: str,
    feature_set: str | None,
    selected_model_names: list[str],
    all_model_names: list[str],
) -> Path:
    file_stem = f"model_comparison_{dataset_name}_{split_key}"

    if feature_set is not None:
        file_stem = f"{file_stem}_{feature_set}"

    if selected_model_names != all_model_names:
        model_slug = "-".join(name.lower().replace(" ", "_") for name in selected_model_names)
        file_stem = f"{file_stem}_{model_slug}"

    return RESULTS_DIR / f"{file_stem}.csv"


def main() -> None:
    args = parse_args()
    dataset_mode = resolve_dataset_mode(dataset=args.dataset, feature_set=args.feature_set)

    model_data = load_model_dataframe(
        dataset=dataset_mode,
        include_group_columns=args.split != "random",
    )
    data_split = split_dataset(
        model_data.model_df,
        feature_columns=model_data.feature_columns,
        target_col=model_data.target_col,
        group_split_columns=model_data.group_split_columns,
        split_strategy=args.split,
    )

    all_models = build_model_registry(
        categorical_cols=model_data.categorical_cols,
        numerical_cols=model_data.numerical_cols,
    )
    models = get_selected_models(all_models, args.models)

    print("Dataset:", model_data.dataset_name)
    print("Dataset mode:", model_data.dataset_mode)
    print("Target:", model_data.target_col)
    print("Split strategy:", SPLIT_LABELS[args.split], f"({args.split})")
    print("Dataset shape:", model_data.model_df.shape)
    print("Target summary:")
    print(model_data.model_df[model_data.target_col].describe().to_string())
    print("Categorical feature count:", len(model_data.categorical_cols))
    print("Categorical features:", ", ".join(model_data.categorical_cols))
    print("Numerical feature count:", len(model_data.numerical_cols))
    if model_data.feature_set is not None:
        print("Feature set:", model_data.feature_set)
    print("X_train shape:", data_split.X_train.shape)
    print("X_val shape:", data_split.X_val.shape)
    print("X_test shape:", data_split.X_test.shape)
    print("Gradient Boosting backend:", gradient_boosting_backend_name())

    backend_detail = gradient_boosting_backend_detail()
    if backend_detail is not None:
        print("Gradient Boosting fallback reason:", backend_detail)

    print("Models:", ", ".join(models))

    results = []
    trained_models = {}
    transformed_shape_printed = False

    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")

        start_train = time.time()
        model.fit(data_split.X_train, data_split.y_train)
        train_time = time.time() - start_train

        if not transformed_shape_printed:
            transformed_train = model.named_steps["preprocessor"].transform(data_split.X_train)
            print("Transformed train feature matrix shape:", transformed_train.shape)
            transformed_shape_printed = True

        start_pred = time.time()
        val_pred = model.predict(data_split.X_val)
        pred_time = time.time() - start_pred

        metrics = regression_metrics(data_split.y_val, val_pred)
        result = {
            "Dataset": model_data.dataset_name,
            "Dataset Mode": model_data.dataset_mode,
            "Target": model_data.target_col,
            "Split": args.split,
            "Feature Set": "" if model_data.feature_set is None else model_data.feature_set,
            "Model": model_name,
            "RMSE": metrics["RMSE"],
            "MAE": metrics["MAE"],
            "R2": metrics["R2"],
            "Train Seconds": train_time,
            "Predict Seconds": pred_time,
        }

        if model_name == "Lasso":
            coef = model.named_steps["model"].coef_
            result["Best Alpha"] = model.named_steps["model"].alpha_
            result["Nonzero Coefficients"] = int(np.count_nonzero(coef))

        print(
            f"{model_name}: RMSE={metrics['RMSE']:.4f}, "
            f"MAE={metrics['MAE']:.4f}, R2={metrics['R2']:.4f}, "
            f"train={train_time:.2f}s, predict={pred_time:.2f}s"
        )

        results.append(result)
        trained_models[model_name] = model

    results_df = pd.DataFrame(results).sort_values(
        by=["RMSE", "MAE", "R2"],
        ascending=[True, True, False],
    ).reset_index(drop=True)

    best_model_name = results_df.loc[0, "Model"]
    best_model = trained_models[best_model_name]
    test_pred = best_model.predict(data_split.X_test)
    test_metrics = regression_metrics(data_split.y_test, test_pred)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = build_results_path(
        dataset_name=model_data.dataset_name,
        split_key=args.split,
        feature_set=model_data.feature_set,
        selected_model_names=list(models),
        all_model_names=list(all_models),
    )
    results_df.to_csv(results_path, index=False)

    print("\nValidation Results")
    print("------------------")
    print(results_df.to_string(index=False))

    print("\nSaved validation results to:", results_path)
    print("\nSelected best model:", best_model_name)
    print("Final Test Performance")
    print("----------------------")
    print(f"RMSE: {test_metrics['RMSE']:.4f}")
    print(f"MAE : {test_metrics['MAE']:.4f}")
    print(f"R^2 : {test_metrics['R2']:.4f}")


if __name__ == "__main__":
    main()
