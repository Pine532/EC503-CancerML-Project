import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.io import savemat
from sklearn.decomposition import TruncatedSVD

from cancer_ml_utils import (
    PROJECT_DIR,
    RANDOM_STATE,
    SPLIT_LABELS,
    SUPPORTED_SPLITS,
    build_model_registry,
    build_preprocessor,
    load_model_dataframe,
    regression_metrics,
    split_dataset,
)


MODEL_KEY_ALIASES = {
    "linear": "Linear",
    "ridge": "Ridge",
    "lasso": "Lasso",
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export canonical model predictions and a 2D embedding for MATLAB visualization."
    )
    parser.add_argument(
        "--split",
        choices=SUPPORTED_SPLITS,
        default="random",
        help="Evaluation split to export.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=tuple(MODEL_KEY_ALIASES),
        default=tuple(MODEL_KEY_ALIASES),
        help="Subset of models to train and export.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap on dataset rows for faster exploratory exports.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_DIR / "matlab_visualization_data.mat",
        help="Path to the output MAT file.",
    )
    return parser.parse_args()


def maybe_subsample_dataframe(model_df: pd.DataFrame, max_rows: Optional[int]) -> pd.DataFrame:
    if max_rows is None or len(model_df) <= max_rows:
        return model_df

    return model_df.sample(n=max_rows, random_state=RANDOM_STATE).reset_index(drop=True)


def main() -> None:
    args = parse_args()

    model_df = load_model_dataframe(include_group_columns=args.split != "random")
    model_df = maybe_subsample_dataframe(model_df, args.max_rows)
    data_split = split_dataset(model_df, split_strategy=args.split)

    preprocessor = build_preprocessor()
    X_train_encoded = preprocessor.fit_transform(data_split.X_train)
    X_val_encoded = preprocessor.transform(data_split.X_val)
    X_test_encoded = preprocessor.transform(data_split.X_test)

    embedding = TruncatedSVD(n_components=2, random_state=RANDOM_STATE)
    train_embedding = embedding.fit_transform(X_train_encoded)
    val_embedding = embedding.transform(X_val_encoded)
    test_embedding = embedding.transform(X_test_encoded)

    model_registry = build_model_registry()
    selected_model_names = [MODEL_KEY_ALIASES[key] for key in args.models]

    train_predictions = []
    val_predictions = []
    test_predictions = []
    metrics_rows = []
    best_result = None

    for model_name in selected_model_names:
        model = model_registry[model_name]
        model.fit(data_split.X_train, data_split.y_train)

        train_pred = model.predict(data_split.X_train)
        val_pred = model.predict(data_split.X_val)
        test_pred = model.predict(data_split.X_test)
        metrics = regression_metrics(data_split.y_val, val_pred)

        train_predictions.append(train_pred)
        val_predictions.append(val_pred)
        test_predictions.append(test_pred)

        row = {
            "Model": model_name,
            "RMSE": metrics["RMSE"],
            "MAE": metrics["MAE"],
            "R2": metrics["R2"],
        }
        metrics_rows.append(row)

        candidate = (metrics["RMSE"], metrics["MAE"], -metrics["R2"], model_name)
        if best_result is None or candidate < best_result:
            best_result = candidate

    metrics_df = pd.DataFrame(metrics_rows).sort_values(
        by=["RMSE", "MAE", "R2"],
        ascending=[True, True, False],
    ).reset_index(drop=True)

    best_model_name = metrics_df.loc[0, "Model"]
    best_model_index = selected_model_names.index(best_model_name) + 1

    mat_payload = {
        "split_key": args.split,
        "split_label": SPLIT_LABELS[args.split],
        "embedding_method": "TruncatedSVD on one-hot encoded canonical features",
        "explained_variance_ratio": embedding.explained_variance_ratio_,
        "model_names": np.array(selected_model_names, dtype=object),
        "best_model_name": best_model_name,
        "best_model_index": best_model_index,
        "train_embedding": train_embedding,
        "val_embedding": val_embedding,
        "test_embedding": test_embedding,
        "y_train": data_split.y_train.to_numpy(),
        "y_val": data_split.y_val.to_numpy(),
        "y_test": data_split.y_test.to_numpy(),
        "train_predictions": np.column_stack(train_predictions),
        "val_predictions": np.column_stack(val_predictions),
        "test_predictions": np.column_stack(test_predictions),
        "metrics_rmse": metrics_df["RMSE"].to_numpy(),
        "metrics_mae": metrics_df["MAE"].to_numpy(),
        "metrics_r2": metrics_df["R2"].to_numpy(),
        "metrics_model_names": np.array(metrics_df["Model"].tolist(), dtype=object),
        "n_train": len(data_split.X_train),
        "n_val": len(data_split.X_val),
        "n_test": len(data_split.X_test),
        "max_rows_used": -1 if args.max_rows is None else args.max_rows,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    savemat(args.output, mat_payload)

    print(f"Saved MATLAB visualization data to: {args.output}")
    print("Split strategy:", SPLIT_LABELS[args.split])
    print("Models exported:", ", ".join(selected_model_names))
    print("Best validation model:", best_model_name)
    print(f"Rows used: train={len(data_split.X_train)}, val={len(data_split.X_val)}, test={len(data_split.X_test)}")


if __name__ == "__main__":
    main()
