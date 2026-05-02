# EC503 CancerML Dataset Guide

This is the single consolidated dataset reference for the project. It combines the project context, dataset modes, targets, feature sets, categorical/continuous handling, leakage controls, and the key statistics needed for a presentation.

Full generated profile tables are stored in `docs/dataset_profile/`.

## One-Sentence Project Summary

This project predicts cancer drug response from GDSC and secondary-screen data using shared preprocessing, train/test split logic, and model-comparison code.

For the main GDSC task, the target is `LN_IC50`. The model learns from cell-line metadata, drug metadata, and, in the expression-enhanced dataset, 500 continuous gene-expression features.

## Dataset Modes

| Dataset mode | Source file | Target | Task |
| --- | --- | --- | --- |
| `gdsc_metadata_only` | `GDSC_DATASET.csv` | `LN_IC50` | Original GDSC task using only 12 categorical metadata features |
| `gdsc_metadata_plus_expression` | `data/model_ready/gdsc_metadata_expression_top500.parquet` | `LN_IC50` | Same GDSC task, but with 500 added continuous gene-expression features |
| `gdsc_auc_metadata_only` | `GDSC_DATASET.csv` | `AUC` | GDSC AUC task using the same 12 categorical metadata features |
| `gdsc_auc_metadata_plus_expression` | `data/model_ready/gdsc_metadata_expression_top500.parquet` | `AUC` | GDSC AUC task with the same 12 metadata features plus 500 expression features |
| `secondary_screen_auc` | `data/secondary_screen/secondary_screen_auc_clean.parquet` | `target_auc` | Separate secondary-screen task predicting response AUC |
| `secondary_screen_ic50` | `data/secondary_screen/secondary_screen_ic50_clean.parquet` | `target_log_ic50` | Separate secondary-screen task predicting log IC50 |

The secondary-screen datasets are not appended to GDSC. They are separate dataset modes with different targets and different identifiers.

## Dataset Size And Feature Counts

| Dataset mode | Raw rows | Raw columns | Modeling rows | Modeling columns | Predictors before encoding | Categorical predictors | Continuous predictors | Estimated one-hot columns | Estimated encoded features |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `gdsc_metadata_only` | 242,035 | 19 | 242,035 | 15 | 12 | 12 | 0 | 358 | 358 |
| `gdsc_metadata_plus_expression` | 236,791 | 519 | 236,791 | 515 | 512 | 12 | 500 | 368 | 868 |
| `gdsc_auc_metadata_only` | 242,035 | 19 | 242,035 | 15 | 12 | 12 | 0 | 358 | 358 |
| `gdsc_auc_metadata_plus_expression` | 236,791 | 519 | 236,791 | 515 | 512 | 12 | 500 | 368 | 868 |
| `secondary_screen_auc` | 690,192 | 15 | 690,192 | 11 | 8 | 8 | 0 | 3,241 | 3,241 |
| `secondary_screen_ic50` | 355,784 | 16 | 355,784 | 11 | 8 | 8 | 0 | 3,172 | 3,172 |

The expression-enhanced GDSC dataset has fewer rows than metadata-only GDSC because only rows with matched expression features are available in the model-ready Parquet.

## Target Variables

| Dataset mode | Target | Mean | Std | Min | 25% | Median | 75% | Max |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `gdsc_metadata_only` | `LN_IC50` | 2.817 | 2.762 | -8.748 | 1.508 | 3.237 | 4.700 | 13.820 |
| `gdsc_metadata_plus_expression` | `LN_IC50` | 2.826 | 2.763 | -8.748 | 1.516 | 3.245 | 4.708 | 13.820 |
| `gdsc_auc_metadata_only` | `AUC` | 0.883 | 0.147 | 0.006 | 0.849 | 0.944 | 0.975 | 0.999 |
| `gdsc_auc_metadata_plus_expression` | `AUC` | 0.883 | 0.147 | 0.006 | 0.850 | 0.945 | 0.975 | 0.999 |
| `secondary_screen_auc` | `target_auc` | 0.958 | 0.282 | 0.004 | 0.806 | 0.916 | 1.127 | 4.889 |
| `secondary_screen_ic50` | `target_log_ic50` | -0.027 | 3.072 | -240.448 | -0.379 | 0.176 | 0.501 | 299.167 |

For GDSC, lower `LN_IC50` means the drug is more potent because less drug is needed to inhibit the cell line.

GDSC `AUC` is a separate response target. It should not be used as a predictor when predicting `LN_IC50`, but it can be predicted directly in the `gdsc_auc_*` dataset modes. When predicting `AUC`, the pipeline excludes `LN_IC50` and `Z_SCORE` as response-derived leakage columns.

For secondary-screen AUC, the target is `target_auc`, not `LN_IC50`. This is a related drug-response prediction problem, but it should not be described as directly improving the GDSC LN_IC50 model unless you explicitly train a transfer-learning or data-fusion method.

## The 12 Canonical GDSC Metadata Features

These 12 features are used in both GDSC modes. They are all categorical before preprocessing.

| Feature | Meaning | Unique values in metadata-only | Missing count | Most common values |
| --- | --- | ---: | ---: | --- |
| `TCGA_DESC` | TCGA-style cancer/tissue label | 32 | 1,067 | `UNCLASSIFIED` 18.9%, `LUAD` 6.5%, `SCLC` 5.6% |
| `GDSC Tissue descriptor 1` | Broad GDSC tissue group | 19 | 9,366 | `lung_NSCLC` 11.1%, `urogenital_system` 10.6%, `leukemia` 8.5% |
| `GDSC Tissue descriptor 2` | More specific GDSC tissue subtype | 54 | 9,366 | `lung_NSCLC_adenocarcinoma` 6.7%, `lung_small_cell_carcinoma` 5.7%, `breast` 5.5% |
| `Cancer Type (matching TCGA label)` | Cancer type mapped to TCGA naming | 31 | 51,446 | missing 21.3%, `LUAD` 6.4%, `SCLC` 5.7% |
| `Microsatellite instability Status (MSI)` | MSI phenotype | 2 | 12,353 | `MSS/MSI-L` 88.5%, `MSI-H` 6.4%, missing 5.1% |
| `Screen Medium` | Experimental growth medium | 2 | 9,366 | `R` 53.6%, `D/F12` 42.5%, missing 3.9% |
| `Growth Properties` | Cell growth behavior | 3 | 9,366 | `Adherent` 69.6%, `Suspension` 23.5%, `Semi-Adherent` 3.1% |
| `CNA` | Whether copy-number data is available | 2 | 9,366 | `Y` 95.7%, `N` 0.4%, missing 3.9% |
| `Gene Expression` | Whether gene-expression data is available | 2 | 9,366 | `Y` 94.2%, `N` 2.0%, missing 3.9% |
| `Methylation` | Whether methylation data is available | 2 | 9,366 | `Y` 93.0%, `N` 3.1%, missing 3.9% |
| `TARGET` | Drug target annotation | 185 | 27,155 | missing 11.2%, `PARP1, PARP2` 1.9%, `MEK1, MEK2` 1.9% |
| `TARGET_PATHWAY` | Drug target pathway | 24 | 0 | `Unclassified` 10.3%, `PI3K/MTOR signaling` 9.4%, `Other` 8.8% |

For categorical variables, "range" means category cardinality and category distribution. Numeric min/max does not apply until after one-hot encoding.

## GDSC Metadata Plus Expression

`gdsc_metadata_plus_expression` keeps the exact same 12 GDSC categorical metadata predictors and adds 500 continuous gene-expression predictors.

The same feature layout is used for `gdsc_auc_metadata_plus_expression`; only the target changes from `LN_IC50` to `AUC`.

| Feature type | Count | Examples | Preprocessing |
| --- | ---: | --- | --- |
| Categorical metadata | 12 | `TCGA_DESC`, `TARGET`, `TARGET_PATHWAY` | Most-frequent imputation, one-hot encoding |
| Continuous expression | 500 | `RPS4Y1`, `KRT19`, `VIM`, `S100P`, `TACSTD2` | Median imputation, standard scaling |

The model is still predicting `LN_IC50`. The new information is molecular cell-line state. In plain language, the model can now learn that certain gene-expression patterns combined with certain drug targets/pathways are associated with higher or lower sensitivity.

Top expression features by standard deviation in the current model-ready file:

| Gene | Mean | Std | Min | Median | Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| `RPS4Y1` | 5.958 | 3.933 | 2.509 | 3.450 | 13.423 |
| `KRT19` | 7.618 | 3.932 | 2.761 | 6.535 | 13.489 |
| `VIM` | 9.682 | 3.714 | 2.599 | 11.651 | 13.408 |
| `S100P` | 6.304 | 3.676 | 2.831 | 3.988 | 13.730 |
| `TACSTD2` | 5.773 | 3.605 | 2.642 | 3.278 | 12.534 |
| `TGFBI` | 7.028 | 3.470 | 2.942 | 7.019 | 13.141 |
| `TM4SF1` | 7.827 | 3.464 | 2.546 | 9.155 | 13.217 |
| `SRGN` | 5.857 | 3.414 | 2.557 | 3.623 | 13.217 |
| `CAV1` | 8.289 | 3.409 | 2.832 | 9.510 | 13.000 |
| `C19orf33` | 6.212 | 3.395 | 2.252 | 4.159 | 11.956 |

The full statistics for all 500 expression features are in `docs/dataset_profile/continuous_feature_summary.csv`.

## Secondary-Screen Features

The secondary-screen modes use 8 categorical predictors and no continuous predictors in the current baseline.

| Feature | Meaning | AUC unique values | AUC most common values |
| --- | --- | ---: | --- |
| `ccle_tissue` | Cell-line tissue | 20 | `LUNG` 19.4%, `TRACT` 10.9%, `SKIN` 8.5% |
| `screen_id` | Screening batch/source | 4 | `HTS002` 85.9%, `MTS010` 9.2%, `MTS006` 4.8% |
| `name` | Drug name | 1,448 | top drug is only 0.2%, so this is high-cardinality |
| `moa` | Drug mechanism of action | 531 | `Unknown` 4.4%, `EGFR inhibitor` 2.9%, `HDAC inhibitor` 1.9% |
| `target` | Drug target | 791 | `Unknown` 18.3%, `MTOR` 1.2%, `EGFR` 1.1% |
| `disease.area` | Drug disease area | 100 | `Unknown` 65.1%, `oncology` 7.0% |
| `indication` | Drug indication | 339 | `Unknown` 65.1%, `breast cancer` 1.0% |
| `phase` | Clinical phase/status | 8 | `Launched` 37.9%, `Preclinical` 26.0%, `Phase 2` 16.0% |

These features describe the cell-line tissue and drug annotations. They do not include dose-response outputs like `auc`, `ec50`, `ic50`, or curve-fit parameters as predictors.

## How Categorical Features Become Numerical

Linear regression and most ML algorithms cannot directly consume strings like `lung_NSCLC`, `PARP1, PARP2`, or `Phase 2`. The pipeline converts categorical features into numbers before fitting models.

Pipeline steps:

1. Missing categorical values are filled using `SimpleImputer(strategy="most_frequent")`.
2. Categories are converted with `OneHotEncoder(handle_unknown="ignore")`.
3. Continuous expression features, when present, are imputed with the median and scaled with `StandardScaler()`.
4. The model receives the transformed numeric matrix.

Example:

| Raw `Screen Medium` value | `Screen Medium_R` | `Screen Medium_D/F12` |
| --- | ---: | ---: |
| `R` | 1 | 0 |
| `D/F12` | 0 | 1 |

After this transformation, Linear Regression, Ridge, Lasso, Linear SVR, Neural Network models, Random Forest, and Gradient Boosting all receive numerical features.

## Why Encoded Feature Counts Are Larger Than Raw Feature Counts

Categorical features expand during one-hot encoding. For example, the single GDSC feature `TARGET` has 185 categories in metadata-only GDSC, so it can contribute roughly 185 binary columns by itself.

That is why:

| Dataset mode | Raw predictors | Encoded features |
| --- | ---: | ---: |
| `gdsc_metadata_only` | 12 | 358 |
| `gdsc_metadata_plus_expression` | 512 | 868 |
| `gdsc_auc_metadata_only` | 12 | 358 |
| `gdsc_auc_metadata_plus_expression` | 512 | 868 |
| `secondary_screen_auc` | 8 | 3,241 |
| `secondary_screen_ic50` | 8 | 3,172 |

The secondary-screen datasets have only 8 raw predictors, but `name`, `moa`, and `target` are high-cardinality categorical variables, so one-hot encoding creates thousands of columns.

## Leakage Controls

The pipeline deliberately excludes response-derived columns and identifiers that would make the task unrealistically easy.

GDSC excluded columns:

| Excluded columns | Reason |
| --- | --- |
| `LN_IC50` | Target, cannot be included in predictors |
| `AUC`, `Z_SCORE` | Response-derived leakage columns |
| `COSMIC_ID`, `CELL_LINE_NAME`, `DRUG_ID`, `DRUG_NAME` | Identifiers/names, not canonical predictors |

Secondary-screen excluded columns:

| Excluded columns | Reason |
| --- | --- |
| `auc`, `upper_limit`, `lower_limit`, `slope`, `r2`, `ec50`, `ic50`, `log_ic50`, `log_ec50` | Dose-response outputs or derived from outputs |
| `broad_id`, `depmap_id`, `ccle_name`, `smiles`, `row_name` | Identifiers or raw names not used as predictors |

`broad_id` and `depmap_id` are not used as predictors.

## Split Types

The canonical pipeline now uses only the `random` split. Rows are randomly split into train, validation, and test sets with fixed `random_state=42`.

We do not average metrics across split types. All current presentation metrics should be described as random-split results.

## What Each Dataset Is Useful For

| Dataset mode | Best use |
| --- | --- |
| `gdsc_metadata_only` | Baseline comparison using the original approved feature set |
| `gdsc_metadata_plus_expression` | Tests whether cell-line molecular information improves `LN_IC50` prediction |
| `gdsc_auc_metadata_only` | Tests whether the original 12 metadata features can predict GDSC AUC |
| `gdsc_auc_metadata_plus_expression` | Tests whether expression features improve GDSC AUC prediction |
| `secondary_screen_auc` | Large-scale separate benchmark for predicting secondary-screen AUC response |
| `secondary_screen_ic50` | Separate benchmark for predicting log IC50 from secondary screen |

For the presentation, keep the distinction clear: the original GDSC task predicts `LN_IC50`; the new GDSC AUC modes predict `AUC`; secondary-screen AUC predicts `target_auc`; secondary-screen IC50 predicts `target_log_ic50`.

## GDSC AUC Smoke-Test Results

Fast random-split smoke tests were run with `Dummy Mean`, `Linear`, and `Ridge`.

| Dataset mode | Best fast model | Validation RMSE | Validation MAE | Validation R2 | Test RMSE | Test MAE | Test R2 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `gdsc_auc_metadata_only` | Ridge | 0.1033 | 0.0666 | 0.4991 | 0.1032 | 0.0662 | 0.5142 |
| `gdsc_auc_metadata_plus_expression` | Ridge | 0.0999 | 0.0656 | 0.5263 | 0.1006 | 0.0656 | 0.5272 |

The expression features gave a modest improvement for AUC prediction in this fast Linear/Ridge comparison.

## Presentation Answers To Likely Questions

**What are the inputs for the original GDSC model?**

The original model uses 12 categorical metadata features describing tissue/cancer type, screening context, whether molecular data exists, and drug target/pathway annotations.

**How can linear regression use categorical features?**

It does not use raw strings. The pipeline applies imputation and one-hot encoding first, which turns each category into numeric binary indicator columns.

**What changed with the new Parquet dataset?**

The new GDSC Parquet keeps the same 12 categorical metadata features and adds 500 continuous gene-expression columns. The target remains `LN_IC50`.

**Can we predict GDSC AUC too?**

Yes. The code now supports `gdsc_auc_metadata_only` and `gdsc_auc_metadata_plus_expression`. These modes use the same GDSC feature setup but make `AUC` the target and exclude `LN_IC50` plus `Z_SCORE` from the predictors.

**Does the secondary-screen dataset help predict GDSC LN_IC50 directly?**

Not directly in the current code. It is a separate dataset mode with separate targets. It helps the project by adding a large second drug-response prediction benchmark.

**Why are `AUC` and `Z_SCORE` excluded from GDSC predictors?**

They are response-derived columns. Including them would leak information about the answer into the input features.

**Why are IDs excluded?**

IDs like `DRUG_ID`, `COSMIC_ID`, `broad_id`, and `depmap_id` can let models memorize entities instead of learning generalizable patterns, so they are not used as predictors.

**Which split do we report?**

We report the random split only. This keeps the experimental setup simple and makes the GDSC `LN_IC50` and GDSC `AUC` results directly comparable.

## Full Generated Data Artifacts

| File or directory | Contents |
| --- | --- |
| `docs/dataset_profile/dataset_summary.csv` | Dataset-level row counts, feature counts, and target summaries |
| `docs/dataset_profile/categorical_feature_summary.csv` | One row per categorical feature with unique counts, missing counts, and top values |
| `docs/dataset_profile/all_categorical_value_counts.csv` | Full histogram/value-count table for every categorical feature |
| `docs/dataset_profile/categorical_value_counts/` | One CSV per categorical feature |
| `docs/dataset_profile/continuous_feature_summary.csv` | Statistics for all 500 expression features |
| `docs/profile_datasets.py` | Script used to regenerate these artifacts from the canonical loaders |

To regenerate the profile:

```bash
PYTHONPYCACHEPREFIX=/tmp .venv/bin/python docs/profile_datasets.py
```
