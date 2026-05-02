# Model Optimization Workflow

This project now has two model-running paths:

| Script | Purpose |
| --- | --- |
| `model_comparison.py` | Baseline model comparison with fixed model definitions |
| `optimize_models.py` | Tuned non-Lasso comparison with randomized hyperparameter search |

`optimize_models.py` intentionally skips Lasso because `LassoCV` has been too slow for the larger datasets.

## Models Covered

| Model | Optimization status |
| --- | --- |
| `Dummy Mean` | No hyperparameters; fitted as a baseline |
| `Linear` | No hyperparameters; fitted as a baseline |
| `Ridge` | Tunes `alpha` |
| `Linear SVR` | Tunes `C`, `epsilon`, and `tol` |
| `Random Forest` | Available only if explicitly requested; abandoned for main optimized runs because it is too slow and underperformed in baseline tests |
| `Gradient Boosting` | Tunes XGBoost parameters when XGBoost is installed; otherwise tunes sklearn `HistGradientBoostingRegressor` parameters |
| `Neural Network` | Available only if explicitly requested; abandoned for main optimized runs because it is too slow for the current workflow |

## What Gets Saved

Each optimization run saves two CSVs in `results/`:

| Output | Contents |
| --- | --- |
| `optimized_model_comparison_*.csv` | Validation/test RMSE, MAE, R2, timing, row caps, warnings |
| `optimized_model_comparison_*_best_params.csv` | Best hyperparameters and searched parameter space |

The script also prints the selected best parameters for each tuned model.

## Important Runtime Controls

| Argument | Meaning |
| --- | --- |
| `--tuning-max-rows` | Samples this many rows from the training split for hyperparameter search |
| `--final-train-max-rows` | Optional row cap for final refit after tuning; omit to refit on the full training split |
| `--n-iter` | Number of sampled hyperparameter combinations per tunable model |
| `--cv` | Number of cross-validation folds during tuning |
| `--n-jobs` | Parallel workers for sklearn randomized search |

For presentation-quality runs, use larger `--tuning-max-rows` and `--n-iter`. For quick debugging, use small values.

## Example Full Commands

Tune the practical non-Lasso suite for GDSC AUC metadata-only:

```bash
python optimize_models.py \
  --dataset gdsc_auc_metadata_only \
  --split random \
  --tuning-max-rows 50000 \
  --n-iter 12 \
  --cv 3
```

Tune the practical non-Lasso suite for GDSC AUC with expression:

```bash
python optimize_models.py \
  --dataset gdsc_auc_metadata_plus_expression \
  --split random \
  --tuning-max-rows 50000 \
  --n-iter 12 \
  --cv 3
```

Run a faster boosting-focused search:

```bash
python optimize_models.py \
  --dataset gdsc_auc_metadata_plus_expression \
  --split random \
  --models Ridge "Gradient Boosting" \
  --tuning-max-rows 50000 \
  --n-iter 20 \
  --cv 3
```

Random Forest can still be run explicitly, but it is no longer recommended for the main optimized result table:

```bash
python optimize_models.py \
  --dataset gdsc_auc_metadata_plus_expression \
  --split random \
  --models "Random Forest" \
  --tuning-max-rows 10000 \
  --n-iter 5 \
  --cv 3
```

Run SVR separately because it can be slow and may produce convergence warnings:

```bash
python optimize_models.py \
  --dataset gdsc_auc_metadata_plus_expression \
  --split random \
  --models "Linear SVR" \
  --tuning-max-rows 20000 \
  --n-iter 10 \
  --cv 3
```

Neural Network can still be run explicitly, but it is no longer recommended for the main optimized result table:

```bash
python optimize_models.py \
  --dataset gdsc_auc_metadata_plus_expression \
  --split random \
  --models "Neural Network" \
  --tuning-max-rows 50000 \
  --n-iter 12 \
  --cv 3
```

The main practical optimized run should focus on Ridge and Gradient Boosting:

```bash
python optimize_models.py \
  --dataset gdsc_auc_metadata_plus_expression \
  --split random \
  --models Ridge "Gradient Boosting" \
  --tuning-max-rows 50000 \
  --n-iter 20 \
  --cv 3
```

## Smoke Tests Already Run

These were validation/debug runs, not final optimized results.

| Dataset | Models | Tuning rows | Final train rows | Iterations | Result file |
| --- | --- | ---: | ---: | ---: | --- |
| `gdsc_auc_metadata_only` | Ridge, Random Forest, Gradient Boosting | 5,000 | 10,000 | 2 | `results/optimized_model_comparison_gdsc_auc_metadata_only_random_ridge-random_forest-gradient_boosting_tune5000_niter2.csv` |
| `gdsc_auc_metadata_only` | Linear SVR, Neural Network | 1,000 | 2,000 | 1 | `results/optimized_model_comparison_gdsc_auc_metadata_only_random_linear_svr-neural_network_tune1000_niter1.csv` |

Observed smoke-test warnings:

| Model | Warning |
| --- | --- |
| `Linear SVR` | Liblinear failed to converge in the tiny smoke test |
| `Neural Network` | Max iterations reached in the tiny smoke test |

These warnings are saved in the result CSV. They are useful evidence that SVR and Neural Network results should not be over-interpreted until tuned with adequate iterations and row counts.
