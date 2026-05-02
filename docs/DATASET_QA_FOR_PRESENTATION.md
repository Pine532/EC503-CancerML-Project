# Dataset Questions To Be Ready For

This document summarizes the datasets as the canonical pipeline currently loads them. Full value-count tables are in `docs/dataset_profile/`.

## Dataset Modes

| Dataset mode | Source file | Target | Rows used | Raw columns | Predictor columns before encoding | Categorical predictors | Continuous predictors | Encoded feature columns |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `gdsc_metadata_only` | `GDSC_DATASET.csv` | `LN_IC50` | 242,035 | 19 | 12 | 12 | 0 | 358 |
| `gdsc_metadata_plus_expression` | `data/model_ready/gdsc_metadata_expression_top500.parquet` | `LN_IC50` | 236,791 | 519 | 512 | 12 | 500 | 868 |
| `gdsc_auc_metadata_only` | `GDSC_DATASET.csv` | `AUC` | 242,035 | 19 | 12 | 12 | 0 | 358 |
| `gdsc_auc_metadata_plus_expression` | `data/model_ready/gdsc_metadata_expression_top500.parquet` | `AUC` | 236,791 | 519 | 512 | 12 | 500 | 868 |
| `secondary_screen_auc` | `data/secondary_screen/secondary_screen_auc_clean.parquet` | `target_auc` | 690,192 | 15 | 8 | 8 | 0 | 3,241 |
| `secondary_screen_ic50` | `data/secondary_screen/secondary_screen_ic50_clean.parquet` | `target_log_ic50` | 355,784 | 16 | 8 | 8 | 0 | 3,172 |

## GDSC Target

For both GDSC modes, the target is `LN_IC50`. Smaller `LN_IC50` generally means the drug is more potent against that cell line because less drug is needed to inhibit growth.

GDSC target summary:

| Dataset mode | Mean | Std | Min | 25% | Median | 75% | Max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `gdsc_metadata_only` | 2.817 | 2.762 | -8.748 | 1.508 | 3.237 | 4.700 | 13.820 |
| `gdsc_metadata_plus_expression` | 2.826 | 2.763 | -8.748 | 1.516 | 3.245 | 4.708 | 13.820 |

The expression dataset has fewer rows because only rows with matched expression features are available in the merged Parquet.

GDSC AUC can also be predicted directly using `gdsc_auc_metadata_only` or `gdsc_auc_metadata_plus_expression`. In those modes, `AUC` is the target and `LN_IC50` plus `Z_SCORE` are excluded as leakage columns.

## The 12 GDSC Categorical Features

These are the approved metadata features used in both GDSC modes:

| Feature | Meaning | Unique values in metadata-only | Most common values |
| --- | --- | ---: | --- |
| `TCGA_DESC` | TCGA-style cancer/tissue label | 32 | `UNCLASSIFIED` 18.9%, `LUAD` 6.5%, `SCLC` 5.6% |
| `GDSC Tissue descriptor 1` | Broad GDSC tissue group | 19 | `lung_NSCLC` 11.1%, `urogenital_system` 10.6%, `leukemia` 8.5% |
| `GDSC Tissue descriptor 2` | More specific GDSC tissue subtype | 54 | `lung_NSCLC_adenocarcinoma` 6.7%, `lung_small_cell_carcinoma` 5.7%, `breast` 5.5% |
| `Cancer Type (matching TCGA label)` | Cancer type mapped to TCGA naming | 31 | missing 21.3%, `LUAD` 6.4%, `SCLC` 5.7% |
| `Microsatellite instability Status (MSI)` | MSI phenotype | 2 | `MSS/MSI-L` 88.5%, `MSI-H` 6.4%, missing 5.1% |
| `Screen Medium` | Experimental growth medium | 2 | `R` 53.6%, `D/F12` 42.5%, missing 3.9% |
| `Growth Properties` | Cell growth behavior | 3 | `Adherent` 69.6%, `Suspension` 23.5%, `Semi-Adherent` 3.1% |
| `CNA` | Whether copy-number data is available | 2 | `Y` 95.7%, `N` 0.4%, missing 3.9% |
| `Gene Expression` | Whether gene-expression data is available | 2 | `Y` 94.2%, `N` 2.0%, missing 3.9% |
| `Methylation` | Whether methylation data is available | 2 | `Y` 93.0%, `N` 3.1%, missing 3.9% |
| `TARGET` | Drug target annotation | 185 | missing 11.2%, `PARP1, PARP2` 1.9%, `MEK1, MEK2` 1.9% |
| `TARGET_PATHWAY` | Drug target pathway | 24 | `Unclassified` 10.3%, `PI3K/MTOR signaling` 9.4%, `Other` 8.8% |

For categorical features, “range” is best discussed as cardinality and category distribution, not numeric min/max.

## How Linear Regression Uses Categorical Features

Linear regression does not receive raw strings. The pipeline converts all categorical predictors into numbers first:

1. Missing category values are filled with the most frequent value using `SimpleImputer(strategy="most_frequent")`.
2. Each categorical feature is one-hot encoded using `OneHotEncoder(handle_unknown="ignore")`.
3. The model receives the resulting numerical matrix.

Example: if `Screen Medium` has values `R` and `D/F12`, one-hot encoding creates binary indicator columns such as `Screen Medium_R` and `Screen Medium_D/F12`. A row has 1 in the column matching its category and 0 in the others.

This is why linear regression, Ridge, Lasso, Linear SVR, and tree models can all use the same categorical metadata pipeline.

## What Changes In GDSC Metadata Plus Expression

`gdsc_metadata_plus_expression` keeps the same 12 categorical metadata features and adds 500 continuous gene-expression features.

The expression features are gene-level continuous values. They are treated differently from categorical metadata:

| Feature type | Columns | Preprocessing |
| --- | ---: | --- |
| Categorical metadata | 12 | Most-frequent imputation, one-hot encoding |
| Continuous expression | 500 | Median imputation, standard scaling |

The model is still predicting `LN_IC50`. The difference is that the model now has molecular information about the cell line, so it can learn relationships like: certain expression patterns plus a drug target/pathway are associated with higher or lower drug sensitivity.

The pipeline does not include `LN_IC50`, `AUC`, `Z_SCORE`, cell-line IDs, or drug IDs as predictors.

## Secondary Screen Dataset

The secondary screen is a separate dataset mode, not appended to GDSC.

Current secondary predictors:

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

The secondary AUC mode predicts `target_auc`; the secondary IC50 mode predicts `target_log_ic50`. It does not directly improve GDSC `LN_IC50` prediction because the pipeline treats it as a separate task with a different target and different identifiers.

The secondary dataset helps answer a related question: can we predict drug response in a much larger secondary screen from cell-line tissue and drug annotations?

## Leakage Controls

The GDSC models exclude:

| Excluded columns | Reason |
| --- | --- |
| `LN_IC50` | Target, cannot be in predictors |
| `AUC`, `Z_SCORE` | Response-derived leakage columns |
| `COSMIC_ID`, `CELL_LINE_NAME`, `DRUG_ID`, `DRUG_NAME` | Identifiers/names, not canonical predictors |

The secondary-screen models exclude:

| Excluded columns | Reason |
| --- | --- |
| `auc`, `upper_limit`, `lower_limit`, `slope`, `r2`, `ec50`, `ic50`, `log_ic50`, `log_ec50` | Dose-response outputs or derived response variables |
| `broad_id`, `depmap_id`, `ccle_name`, `smiles`, `row_name` | Identifiers or raw names not used as predictors |

`broad_id` and `depmap_id` are not used as predictors.

## Split Interpretation

The canonical pipeline now uses only the `random` split. Rows are randomly split into train, validation, and test sets with fixed `random_state=42`.

Do not average metrics across split types. The current presentation metrics should be described as random-split results.

## Where The Full Histogram Data Is

Per-feature category histograms are saved as CSV value-count tables:

| Artifact | Contents |
| --- | --- |
| `docs/dataset_profile/dataset_summary.csv` | Row counts, feature counts, target summaries |
| `docs/dataset_profile/categorical_feature_summary.csv` | One row per categorical feature with unique counts, missing counts, and top categories |
| `docs/dataset_profile/all_categorical_value_counts.csv` | Full histogram table for every categorical feature |
| `docs/dataset_profile/categorical_value_counts/` | One CSV per categorical feature |
| `docs/dataset_profile/continuous_feature_summary.csv` | Summary statistics for all 500 expression features |
