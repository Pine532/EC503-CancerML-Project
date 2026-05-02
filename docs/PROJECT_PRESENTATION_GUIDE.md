# EC503 CancerML Presentation Guide

This guide reflects the current random-split-only project state.

## Current Scope

The presentation should focus on GDSC only:

| Dataset mode | Target | Features |
| --- | --- | --- |
| `gdsc_metadata_only` | `LN_IC50` | 12 categorical metadata features |
| `gdsc_metadata_plus_expression` | `LN_IC50` | Same 12 metadata features plus 500 expression features |
| `gdsc_auc_metadata_only` | `AUC` | Same 12 categorical metadata features |
| `gdsc_auc_metadata_plus_expression` | `AUC` | Same 12 metadata features plus 500 expression features |

The secondary/PRISM dataset is not part of the main presentation.

## Split Policy

The canonical pipeline now supports only the `random` split.

All reported GDSC metrics should be described as random train/validation/test split results. Do not average metrics across split types.

## Main Results To Present

| Task | Dataset | Best model | RMSE | R2 |
| --- | --- | --- | ---: | ---: |
| `LN_IC50` | Metadata only | Ridge | 1.5736 | 0.6714 |
| `LN_IC50` | Metadata + expression | Ridge | 1.4671 | 0.7154 |
| `AUC` | Metadata + expression | Optimized Gradient Boosting | 0.0958 test | 0.5716 test |

## Optimized AUC Model

Best current optimized model:

```text
Dataset: gdsc_auc_metadata_plus_expression
Target: AUC
Model: XGBoost / Gradient Boosting
Split: random
Test RMSE: 0.0958
Test R2: 0.5716
```

Best parameters:

```text
n_estimators = 500
learning_rate = 0.08
max_depth = 5
subsample = 0.7
colsample_bytree = 1.0
reg_lambda = 10.0
```

## Runtime Notes

The baseline timing results were measured on the current local machine using the
canonical random split. For the metadata-only GDSC datasets, the model trained
on about `n = 145,221` rows with `k = 358` encoded features after one-hot
encoding. For the metadata-plus-expression datasets, the model trained on about
`n = 142,074` rows with `k = 868` encoded features, consisting of the same
encoded metadata features plus 500 continuous expression features. Linear
Regression and Ridge were computationally fastest because they train directly
on this encoded feature matrix; Ridge adds only a small regularization step.
LASSO was excluded from the final runs because its iterative regularized
optimization was too slow for the larger feature sets. Random Forest was also
excluded from the final results because it was slow relative to its benefit.
Gradient Boosting was slower than the linear models because XGBoost stores and
evaluates many decision trees, with cost depending on the number of rows, number
of columns, number of trees, and tree depth. GBM tuning was the largest
bottleneck: the main AUC tuning run used a 50,000-row tuning sample, 3-fold
cross-validation, and 20 random-search iterations, which required 60 GBM fits.
The metadata-plus-expression GBM tuning run took about `453.4 s` for search
plus `40.3 s` for final refit; the metadata-only GBM tuning run took about
`19.1 s` for search plus `1.1 s` for final refit. Detailed pseudocode for the
shared model-comparison pipeline is contained in Appendix A.

| Target | Feature set | Model | Train seconds | Predict seconds |
| --- | --- | --- | ---: | ---: |
| `LN_IC50` | Metadata only | Dummy Mean | 0.27 | 0.06 |
| `LN_IC50` | Metadata only | Linear Regression | 0.48 | 0.06 |
| `LN_IC50` | Metadata only | Ridge | 0.43 | 0.06 |
| `LN_IC50` | Metadata only | Gradient Boosting | 0.60 | 0.09 |
| `LN_IC50` | Metadata + expression | Dummy Mean | 7.52 | 0.32 |
| `LN_IC50` | Metadata + expression | Linear Regression | 13.38 | 0.37 |
| `LN_IC50` | Metadata + expression | Ridge | 7.61 | 0.36 |
| `LN_IC50` | Metadata + expression | Gradient Boosting | 19.93 | 0.37 |
| `AUC` | Metadata only | Dummy Mean | 0.27 | 0.06 |
| `AUC` | Metadata only | Linear Regression | 0.46 | 0.06 |
| `AUC` | Metadata only | Ridge | 0.44 | 0.06 |
| `AUC` | Metadata only | Gradient Boosting | 0.58 | 0.08 |
| `AUC` | Metadata + expression | Dummy Mean | 7.62 | 0.33 |
| `AUC` | Metadata + expression | Linear Regression | 16.28 | 0.43 |
| `AUC` | Metadata + expression | Ridge | 8.36 | 0.39 |
| `AUC` | Metadata + expression | Gradient Boosting | 20.42 | 0.71 |

## Key Talking Points

- `LN_IC50` and `AUC` are separate GDSC targets.
- Lower `LN_IC50` means stronger drug sensitivity.
- AUC summarizes the dose-response curve differently from IC50.
- Categorical metadata is converted to numeric features with imputation and one-hot encoding.
- Expression features add continuous cell-line molecular information.
- Adding expression improves `LN_IC50` random-split R2 from about `0.671` to `0.715`.
- Optimized Gradient Boosting gives the strongest current GDSC AUC result with test R2 about `0.572`.

## Commands

Run baseline GDSC `LN_IC50`:

```bash
python model_comparison.py --dataset gdsc_metadata_only --split random --models "Dummy Mean" Linear Ridge
python model_comparison.py --dataset gdsc_metadata_plus_expression --split random --models "Dummy Mean" Linear Ridge
```

Run optimized GDSC `AUC`:

```bash
python optimize_models.py --dataset gdsc_auc_metadata_plus_expression --split random --models Ridge "Gradient Boosting" --tuning-max-rows 50000 --n-iter 20 --cv 3
```
