# EC503 CancerML Project

This project compares machine-learning baselines for cancer drug sensitivity prediction across GDSC and cleaned secondary-screen datasets.

## Dataset Modes

- `gdsc_metadata_only`: predicts `LN_IC50` from 12 categorical GDSC metadata features.
- `gdsc_metadata_plus_expression`: predicts `LN_IC50` from the same metadata plus the top 500 continuous gene-expression features.
- `secondary_screen_auc`: predicts `target_auc` from safe categorical secondary-screen predictors.
- `secondary_screen_ic50`: predicts `target_log_ic50`, derived from secondary-screen IC50/log IC50 values.

## Features and Leakage Controls

GDSC metadata features exclude `LN_IC50`, `AUC`, `Z_SCORE`, `COSMIC_ID`, `CELL_LINE_NAME`, `DRUG_ID`, and `DRUG_NAME`.

Secondary-screen predictors are:

- `ccle_tissue`
- `screen_id`
- `name`
- `moa`
- `target`
- `disease.area`
- `indication`
- `phase`

Secondary-screen models exclude dose-response outputs and identifiers as predictors, including `auc`, `ic50`, `ec50`, `slope`, `r2`, `upper_limit`, `lower_limit`, `log_ic50`, `log_ec50`, `target_auc`, `target_log_ic50`, `broad_id`, `depmap_id`, `ccle_name`, `smiles`, and `row_name`.

## Regenerating Cleaned Data

Run these in order when rebuilding generated datasets:

```powershell
.\.venv\Scripts\python.exe clean_expression_data.py
.\.venv\Scripts\python.exe merge_expression_features.py
.\.venv\Scripts\python.exe optimize_merged_dataset.py
.\.venv\Scripts\python.exe clean_secondary_screen.py
```

`secondary_screen_ic50_clean.parquet` is created on demand by `model_comparison.py` from the cleaned secondary-screen AUC parquet.

## Classical Experiments

Core GDSC comparisons:

```powershell
.\.venv\Scripts\python.exe model_comparison.py --dataset gdsc_metadata_only --split random --models "Dummy Mean" Linear Ridge
.\.venv\Scripts\python.exe model_comparison.py --dataset gdsc_metadata_plus_expression --split random --models "Dummy Mean" Linear Ridge
```

Secondary AUC:

```powershell
.\.venv\Scripts\python.exe model_comparison.py --dataset secondary_screen_auc --split random --models "Dummy Mean" Linear Ridge "Gradient Boosting"
```

Secondary IC50 with train-derived target clipping:

```powershell
.\.venv\Scripts\python.exe model_comparison.py --dataset secondary_screen_ic50 --split random --models "Dummy Mean" Linear Ridge "Gradient Boosting" --clip-target-quantiles 0.01 0.99
```

Sampled SVR experiments:

```powershell
.\.venv\Scripts\python.exe model_comparison.py --dataset secondary_screen_auc --split random --models "Linear SVR" --max-rows 100000
.\.venv\Scripts\python.exe model_comparison.py --dataset secondary_screen_ic50 --split random --models "Linear SVR" --max-rows 100000 --clip-target-quantiles 0.01 0.99
```

## Neural Network Baseline

The neural-network script uses PyTorch if installed. If PyTorch is unavailable, it uses sklearn `MLPRegressor` and reports that fallback in the output CSV.

```powershell
.\.venv\Scripts\python.exe neural_network_baseline.py --dataset secondary_screen_auc --split random --max-rows 100000 --epochs 20
.\.venv\Scripts\python.exe neural_network_baseline.py --dataset secondary_screen_ic50 --split random --max-rows 100000 --epochs 20
```

## Combining Results and Plots

```powershell
.\.venv\Scripts\python.exe combine_results.py
.\.venv\Scripts\python.exe generate_final_plots.py
```

Outputs:

- `results/all_results_with_optimized.csv`
- `figures/r2_by_model_dataset_split.png`
- `figures/rmse_by_model_dataset_split.png`
- `figures/gdsc_metadata_vs_expression.png`
- `figures/secondary_auc_vs_ic50.png`

## Runtime Notes

- Full-dataset `Linear SVR` can be slow; use `--max-rows` for controlled experiments.
- `LassoCV` can also be slow on large one-hot matrices.
- Gradient Boosting uses XGBoost when installed and otherwise falls back to sklearn `HistGradientBoostingRegressor`.
- `secondary_screen_ic50` has large finite target outliers; `--clip-target-quantiles 0.01 0.99` clips targets using training-set thresholds only.
- Generated data, results, figures, and local virtual environments are ignored by git.
- The canonical pipeline now supports random splits only.
