# GBM Optimization Plots

Run:

```bash
python visualize_gbm_optimization.py
```

The script reads `results/all_results_with_optimized.csv` plus the optimized
best-parameter CSVs and writes plots to `figures/`.

Generated plots:

- `gbm_auc_rmse_optimization.png`: compares baseline Gradient Boosting RMSE,
  optimized validation RMSE, and optimized held-out test RMSE for GDSC AUC.
- `gbm_auc_rmse_optimization_line.png`: shows the same RMSE comparison as a
  line graph across baseline validation, optimized validation, and optimized
  held-out test.
- `gbm_auc_r2_optimization.png`: compares baseline Gradient Boosting R2,
  optimized validation R2, and optimized held-out test R2 for GDSC AUC.
- `gbm_auc_r2_optimization_line.png`: shows the same R2 comparison as a line
  graph across baseline validation, optimized validation, and optimized held-out
  test.
- `gbm_auc_best_hyperparameters.png`: shows the selected XGBoost settings for
  metadata-only and metadata-plus-expression AUC models.
- `gbm_auc_random_search_cv_trace.png`: shows the actual GBM random-search
  candidates tried for AUC. Each point is one hyperparameter combination scored
  by cross-validation RMSE.
- `gbm_auc_random_search_parameter_trace.png`: shows the actual AUC GBM
  hyperparameter values tried at each random-search candidate.
- `gbm_lnic50_random_search_cv_trace.png`: same candidate-level optimization
  trace for LN_IC50.
- `gbm_lnic50_random_search_parameter_trace.png`: same candidate-level
  hyperparameter-value trace for LN_IC50.

Lower RMSE is better. Higher R2 is better.

The optimizer saves per-candidate cross-validation results in
`results/*gradient_boosting_cv_results.csv`. The visualization script also
writes a combined table:

- `results/gbm_random_search_trials.csv`
