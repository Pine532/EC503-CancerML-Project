import argparse
import time

import numpy as np
import pandas as pd

from cancer_ml_utils import (
    SPLIT_LABELS,
    SUPPORTED_SPLITS,
    build_model_registry,
    load_model_dataframe,
    regression_metrics,
    split_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare baseline regressors on the canonical EC503 cancer ML feature set."
    )
    parser.add_argument(
        "--split",
        choices=SUPPORTED_SPLITS,
        default="random",
        help="Evaluation split to use.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_df = load_model_dataframe(include_group_columns=args.split != "random")
    data_split = split_dataset(model_df, split_strategy=args.split)
    models = build_model_registry()

    print("Split strategy:", SPLIT_LABELS[args.split])
    print("X_train shape:", data_split.X_train.shape)
    print("X_val shape:", data_split.X_val.shape)
    print("X_test shape:", data_split.X_test.shape)

    results = []
    trained_models = {}

    for model_name, model in models.items():
        start_train = time.time()
        model.fit(data_split.X_train, data_split.y_train)
        train_time = time.time() - start_train

        start_pred = time.time()
        val_pred = model.predict(data_split.X_val)
        pred_time = time.time() - start_pred

        metrics = regression_metrics(data_split.y_val, val_pred)
        result = {
            "Model": model_name,
            "RMSE": metrics["RMSE"],
            "MAE": metrics["MAE"],
            "R2": metrics["R2"],
            "Train Seconds": train_time,
            "Val Predict Seconds": pred_time,
        }

        if model_name == "Lasso":
            coef = model.named_steps["model"].coef_
            result["Best Alpha"] = model.named_steps["model"].alpha_
            result["Nonzero Coefficients"] = int(np.count_nonzero(coef))

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

    print("\nValidation Results")
    print("------------------")
    print(results_df.to_string(index=False))

    print("\nSelected best model:", best_model_name)
    print("Final Test Performance")
    print("----------------------")
    print(f"RMSE: {test_metrics['RMSE']:.4f}")
    print(f"MAE : {test_metrics['MAE']:.4f}")
    print(f"R^2 : {test_metrics['R2']:.4f}")


if __name__ == "__main__":
    main()
