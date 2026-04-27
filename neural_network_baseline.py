from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from cancer_ml_utils import (
    RANDOM_STATE,
    RESULTS_DIR,
    SPLIT_LABELS,
    SUPPORTED_DATASETS,
    SUPPORTED_SPLITS,
    build_preprocessor,
    load_model_dataframe,
    split_dataset,
)

try:
    import torch
    from torch import nn

    TORCH_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - depends on local environment
    torch = None
    nn = None
    TORCH_IMPORT_ERROR = exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a simple neural-network baseline on EC503 dataset modes."
    )
    parser.add_argument("--dataset", choices=SUPPORTED_DATASETS, required=True)
    parser.add_argument("--split", choices=SUPPORTED_SPLITS, default="random")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional row cap for smoke tests.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    return parser.parse_args()


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def cap_rows(model_df: pd.DataFrame, max_rows: int | None) -> pd.DataFrame:
    if max_rows is None or len(model_df) <= max_rows:
        return model_df

    return model_df.sample(n=max_rows, random_state=RANDOM_STATE).reset_index(drop=True)


def to_float32_array(matrix):
    if sparse.issparse(matrix):
        return matrix.astype(np.float32)

    return np.asarray(matrix, dtype=np.float32)


def iter_batches(matrix, y: np.ndarray, batch_size: int, rng: np.random.Generator):
    indices = np.arange(y.shape[0])
    rng.shuffle(indices)

    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start : start + batch_size]
        X_batch = matrix[batch_idx]
        if sparse.issparse(X_batch):
            X_batch = X_batch.toarray()
        yield np.asarray(X_batch, dtype=np.float32), y[batch_idx].astype(np.float32)


def train_torch_mlp(
    X_train,
    y_train_scaled: np.ndarray,
    X_val,
    input_dim: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> np.ndarray:
    torch.manual_seed(RANDOM_STATE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = nn.Sequential(
        nn.Linear(input_dim, 256),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(128, 1),
    ).to(device)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    rng = np.random.default_rng(RANDOM_STATE)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = []
        for X_batch, y_batch in iter_batches(X_train, y_train_scaled, batch_size, rng):
            X_tensor = torch.from_numpy(X_batch).to(device)
            y_tensor = torch.from_numpy(y_batch.reshape(-1, 1)).to(device)

            optimizer.zero_grad()
            loss = loss_fn(model(X_tensor), y_tensor)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))

        print(f"Epoch {epoch}/{epochs} - train MSE: {np.mean(epoch_losses):.6f}")

    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, X_val.shape[0], batch_size):
            X_batch = X_val[start : start + batch_size]
            if sparse.issparse(X_batch):
                X_batch = X_batch.toarray()
            X_tensor = torch.from_numpy(np.asarray(X_batch, dtype=np.float32)).to(device)
            predictions.append(model(X_tensor).cpu().numpy().ravel())

    return np.concatenate(predictions)


def train_sklearn_mlp(
    X_train,
    y_train_scaled: np.ndarray,
    X_val,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> np.ndarray:
    model = MLPRegressor(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=batch_size,
        learning_rate_init=learning_rate,
        max_iter=epochs,
        random_state=RANDOM_STATE,
        early_stopping=False,
        verbose=True,
    )
    model.fit(X_train, y_train_scaled)
    return model.predict(X_val)


def build_results_path(dataset_mode: str, split_key: str) -> Path:
    return RESULTS_DIR / f"neural_network_{dataset_mode}_{split_key}.csv"


def main() -> None:
    args = parse_args()

    model_data = load_model_dataframe(
        dataset=args.dataset,
        include_group_columns=args.split != "random",
    )
    model_df = cap_rows(model_data.model_df, args.max_rows)
    data_split = split_dataset(
        model_df,
        feature_columns=model_data.feature_columns,
        target_col=model_data.target_col,
        group_split_columns=model_data.group_split_columns,
        split_strategy=args.split,
    )

    preprocessor = build_preprocessor(model_data.categorical_cols, model_data.numerical_cols)
    X_train = to_float32_array(preprocessor.fit_transform(data_split.X_train))
    X_val = to_float32_array(preprocessor.transform(data_split.X_val))

    target_scaler = StandardScaler()
    y_train = data_split.y_train.to_numpy(dtype=np.float32).reshape(-1, 1)
    y_train_scaled = target_scaler.fit_transform(y_train).ravel()

    print("Dataset mode:", model_data.dataset_mode)
    print("Target:", model_data.target_col)
    print("Split strategy:", SPLIT_LABELS[args.split], f"({args.split})")
    print("Raw dataset shape:", model_data.raw_shape)
    print("Rows after filtering target:", len(model_data.model_df))
    print("Rows used:", len(model_df))
    print("Target summary:")
    print(model_df[model_data.target_col].describe().to_string())
    print("Categorical feature count:", len(model_data.categorical_cols))
    print("Categorical features:", ", ".join(model_data.categorical_cols))
    print("Numerical feature count:", len(model_data.numerical_cols))
    if model_data.numerical_cols:
        print("Numerical features preview:", ", ".join(model_data.numerical_cols[:10]))
    print("X_train shape:", data_split.X_train.shape)
    print("X_val shape:", data_split.X_val.shape)
    print("X_test shape:", data_split.X_test.shape)
    print("Input feature dimension:", X_train.shape[1])

    backend = "PyTorch" if torch is not None else "sklearn MLPRegressor"
    print("Neural network backend:", backend)
    if TORCH_IMPORT_ERROR is not None:
        print("PyTorch unavailable, using sklearn fallback:", TORCH_IMPORT_ERROR)

    start_train = time.time()
    if torch is not None:
        val_pred_scaled = train_torch_mlp(
            X_train=X_train,
            y_train_scaled=y_train_scaled,
            X_val=X_val,
            input_dim=X_train.shape[1],
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
        )
    else:
        val_pred_scaled = train_sklearn_mlp(
            X_train=X_train,
            y_train_scaled=y_train_scaled,
            X_val=X_val,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
        )
    train_seconds = time.time() - start_train

    val_pred = target_scaler.inverse_transform(val_pred_scaled.reshape(-1, 1)).ravel()
    y_val = data_split.y_val.to_numpy(dtype=np.float32)
    metrics = regression_metrics(y_val, val_pred)

    result = {
        "Dataset": model_data.dataset_name,
        "Dataset Mode": model_data.dataset_mode,
        "Target": model_data.target_col,
        "Split": args.split,
        "Feature Set": "" if model_data.feature_set is None else model_data.feature_set,
        "Model": "Neural Network",
        "Backend": backend,
        "Rows Used": len(model_df),
        "Input Dim": X_train.shape[1],
        "Epochs": args.epochs,
        "Batch Size": args.batch_size,
        "Learning Rate": args.learning_rate,
        "RMSE": metrics["RMSE"],
        "MAE": metrics["MAE"],
        "R2": metrics["R2"],
        "Train Seconds": train_seconds,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = build_results_path(model_data.dataset_mode, args.split)
    pd.DataFrame([result]).to_csv(results_path, index=False)

    print("\nValidation Performance")
    print("----------------------")
    print(f"RMSE: {metrics['RMSE']:.4f}")
    print(f"MAE : {metrics['MAE']:.4f}")
    print(f"R^2 : {metrics['R2']:.4f}")
    print(f"Train seconds: {train_seconds:.2f}")
    print("Saved neural-network results to:", results_path)


if __name__ == "__main__":
    main()
