import argparse
import time

import numpy as np

from cancer_ml_utils import (
    SPLIT_LABELS,
    SUPPORTED_SPLITS,
    build_lasso_pipeline,
    load_model_dataframe,
    regression_metrics,
    split_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Lasso baseline on the GDSC metadata features.")
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

    print("Split strategy:", SPLIT_LABELS[args.split])
    print("X_train shape:", data_split.X_train.shape)
    print("X_val shape:", data_split.X_val.shape)
    print("X_test shape:", data_split.X_test.shape)

    lasso_pipeline = build_lasso_pipeline()

    start_train = time.time()
    lasso_pipeline.fit(data_split.X_train, data_split.y_train)
    train_time = time.time() - start_train

    start_pred = time.time()
    y_pred = lasso_pipeline.predict(data_split.X_test)
    pred_time = time.time() - start_pred

    metrics = regression_metrics(data_split.y_test, y_pred)
    best_alpha = lasso_pipeline.named_steps["model"].alpha_
    coef = lasso_pipeline.named_steps["model"].coef_
    nonzero = int(np.count_nonzero(coef))
    total = len(coef)

    print("\nLASSO Results")
    print("-------------")
    print(f"Best alpha: {best_alpha:.6f}")
    print(f"RMSE: {metrics['RMSE']:.4f}")
    print(f"MAE : {metrics['MAE']:.4f}")
    print(f"R^2 : {metrics['R2']:.4f}")
    print(f"Training time  : {train_time:.4f} seconds")
    print(f"Inference time : {pred_time:.4f} seconds")
    print(f"Nonzero coefficients: {nonzero} / {total}")


if __name__ == "__main__":
    main()
