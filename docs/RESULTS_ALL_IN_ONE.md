# Results Summary

This file consolidates the current baseline results, optimized results, and optimized hyperparameters.

The full machine-readable table is `results/all_results_with_optimized.csv`.

## Best Result Per Dataset/Split

| Dataset Mode | Target | Split | Experiment Type | Model | Primary RMSE | Primary R2 | Best Params Compact |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gdsc_auc_metadata_only | AUC | random | Optimized | Gradient Boosting | 0.0998 | 0.5460 | colsample_bytree=1.0, learning_rate=0.08, max_depth=5, n_estimators=500, reg_lambda=10.0, subsample=0.7 |
| gdsc_auc_metadata_plus_expression | AUC | random | Optimized | Gradient Boosting | 0.0958 | 0.5716 | colsample_bytree=1.0, learning_rate=0.08, max_depth=5, n_estimators=500, reg_lambda=10.0, subsample=0.7 |
| gdsc_metadata_only | LN_IC50 | random | Baseline | Ridge | 1.5736 | 0.6714 |  |
| gdsc_metadata_plus_expression | LN_IC50 | random | Baseline | Ridge | 1.4671 | 0.7154 |  |

## Optimized Runs

| Dataset Mode | Target | Split | Model | Tuning Max Rows | N Iter | CV Folds | Best CV RMSE | RMSE | R2 | Test RMSE | Test R2 | Best Params Compact |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gdsc_auc_metadata_only | AUC | random | Gradient Boosting | 50000.0 | 20.0000 | 3.0000 | 0.0999 | 0.0995 | 0.5348 | 0.0998 | 0.5460 | colsample_bytree=1.0, learning_rate=0.08, max_depth=5, n_estimators=500, reg_lambda=10.0, subsample=0.7 |
| gdsc_auc_metadata_only | AUC | random | Ridge | 50000.0 | 20.0000 | 3.0000 | 0.1025 | 0.1033 | 0.4991 | 0.1032 | 0.5142 | alpha=0.3 |
| gdsc_auc_metadata_plus_expression | AUC | random | Gradient Boosting | 50000.0 | 20.0000 | 3.0000 | 0.0983 | 0.0947 | 0.5742 | 0.0958 | 0.5716 | colsample_bytree=1.0, learning_rate=0.08, max_depth=5, n_estimators=500, reg_lambda=10.0, subsample=0.7 |
| gdsc_auc_metadata_plus_expression | AUC | random | Ridge | 50000.0 | 20.0000 | 3.0000 | 0.1021 | 0.0999 | 0.5263 | 0.1006 | 0.5272 | alpha=1.0 |

## GDSC AUC Results

| Experiment Type | Dataset Mode | Target | Split | Model | RMSE | MAE | R2 | Test RMSE | Test MAE | Test R2 | Best Params Compact |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | gdsc_auc_metadata_only | AUC | random | Ridge | 0.1033 | 0.0666 | 0.4991 |  |  |  |  |
| Baseline | gdsc_auc_metadata_only | AUC | random | Linear | 0.1033 | 0.0666 | 0.4991 |  |  |  |  |
| Baseline | gdsc_auc_metadata_only | AUC | random | Gradient Boosting | 0.1082 | 0.0752 | 0.4498 |  |  |  |  |
| Baseline | gdsc_auc_metadata_only | AUC | random | Dummy Mean | 0.1459 | 0.1026 | -0.0000 |  |  |  |  |
| Optimized | gdsc_auc_metadata_only | AUC | random | Gradient Boosting | 0.0995 | 0.0642 | 0.5348 | 0.0998 | 0.0641 | 0.5460 | colsample_bytree=1.0, learning_rate=0.08, max_depth=5, n_estimators=500, reg_lambda=10.0, subsample=0.7 |
| Optimized | gdsc_auc_metadata_only | AUC | random | Ridge | 0.1033 | 0.0666 | 0.4991 | 0.1032 | 0.0662 | 0.5142 | alpha=0.3 |
| Baseline | gdsc_auc_metadata_plus_expression | AUC | random | Ridge | 0.0999 | 0.0656 | 0.5263 |  |  |  |  |
| Baseline | gdsc_auc_metadata_plus_expression | AUC | random | Linear | 0.0999 | 0.0656 | 0.5262 |  |  |  |  |
| Baseline | gdsc_auc_metadata_plus_expression | AUC | random | Gradient Boosting | 0.1054 | 0.0738 | 0.4727 |  |  |  |  |
| Baseline | gdsc_auc_metadata_plus_expression | AUC | random | Dummy Mean | 0.1451 | 0.1023 | -0.0001 |  |  |  |  |
| Optimized | gdsc_auc_metadata_plus_expression | AUC | random | Gradient Boosting | 0.0947 | 0.0610 | 0.5742 | 0.0958 | 0.0610 | 0.5716 | colsample_bytree=1.0, learning_rate=0.08, max_depth=5, n_estimators=500, reg_lambda=10.0, subsample=0.7 |
| Optimized | gdsc_auc_metadata_plus_expression | AUC | random | Ridge | 0.0999 | 0.0656 | 0.5263 | 0.1006 | 0.0656 | 0.5272 | alpha=1.0 |

## GDSC LN_IC50 Results

| Experiment Type | Dataset Mode | Target | Split | Model | RMSE | MAE | R2 | Test RMSE | Test MAE | Test R2 | Best Params Compact |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | gdsc_metadata_only | LN_IC50 | random | Ridge | 1.5736 | 1.1843 | 0.6714 |  |  |  |  |
| Baseline | gdsc_metadata_only | LN_IC50 | random | Linear | 1.5736 | 1.1842 | 0.6714 |  |  |  |  |
| Baseline | gdsc_metadata_only | LN_IC50 | random | Gradient Boosting | 1.8013 | 1.4365 | 0.5694 |  |  |  |  |
| Baseline | gdsc_metadata_only | LN_IC50 | random | Dummy Mean | 2.7450 | 2.0850 | -0.0000 |  |  |  |  |
| Baseline | gdsc_metadata_plus_expression | LN_IC50 | random | Ridge | 1.4671 | 1.1001 | 0.7154 |  |  |  |  |
| Baseline | gdsc_metadata_plus_expression | LN_IC50 | random | Linear | 1.4671 | 1.1000 | 0.7154 |  |  |  |  |
| Baseline | gdsc_metadata_plus_expression | LN_IC50 | random | Gradient Boosting | 1.7378 | 1.3973 | 0.6007 |  |  |  |  |
| Baseline | gdsc_metadata_plus_expression | LN_IC50 | random | Dummy Mean | 2.7503 | 2.0913 | -0.0002 |  |  |  |  |

## Secondary Screen Results

_No rows._

## Notes

- Baseline `RMSE`, `MAE`, and `R2` are validation metrics saved by `model_comparison.py` and `neural_network_baseline.py`.
- Optimized rows include validation metrics plus final held-out test metrics.
- Random Forest and Neural Network were abandoned for the main optimized workflow because they were slow relative to their current benefit.
- Lasso is excluded from optimization because it takes too long on these datasets.
- Optimized Gradient Boosting uses XGBoost in this environment.

Total baseline rows: 16.
Total optimized rows: 4.