import argparse
import time

from cancer_ml_utils import (
    SPLIT_LABELS,
    SUPPORTED_SPLITS,
    build_ridge_pipeline,
    load_model_dataframe,
    regression_metrics,
    split_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Ridge baseline on the GDSC metadata features.")
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

    ridge_pipeline = build_ridge_pipeline()

    start_train = time.time()
    ridge_pipeline.fit(data_split.X_train, data_split.y_train)
    train_time = time.time() - start_train

    start_pred = time.time()
    y_pred = ridge_pipeline.predict(data_split.X_test)
    pred_time = time.time() - start_pred

    metrics = regression_metrics(data_split.y_test, y_pred)

    print("\nRidge Baseline Results")
    print("----------------------")
    print(f"RMSE: {metrics['RMSE']:.4f}")
    print(f"MAE : {metrics['MAE']:.4f}")
    print(f"R^2 : {metrics['R2']:.4f}")
    print(f"Training time  : {train_time:.4f} seconds")
    print(f"Inference time : {pred_time:.4f} seconds")

    X_train_transformed = ridge_pipeline.named_steps["preprocessor"].transform(data_split.X_train)
    print("Transformed feature matrix shape:", X_train_transformed.shape)


if __name__ == "__main__":
    main()
