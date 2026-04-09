# EC503-CancerML-Project

Baseline machine learning experiments for predicting cancer drug response (`LN_IC50`) from GDSC metadata.

## Canonical pipeline

The repo now uses one shared experiment definition in `cancer_ml_utils.py`:

- dataset path resolution relative to the repo
- the canonical 12-feature metadata list
- shared preprocessing with `SimpleImputer(strategy="most_frequent")` and `OneHotEncoder(handle_unknown="ignore")`
- canonical LASSO preprocessing with sparse-safe scaling
- supported split strategies: `random`, `drug`, and `cell_line`

Use the Python scripts as the source of truth. The notebooks mirror those scripts for exploration and plotting.

## What changed

This cleanup pass addressed the main issues from the earlier review:

- `ML_model_implementation.ipynb` no longer hard-codes `best_model = rf`; the best model is selected programmatically from validation metrics.
- LASSO now has one canonical implementation path: shared preprocessing plus sparse-safe scaling before `LassoCV`.
- The feature list is centralized in `cancer_ml_utils.py` and reused across scripts and notebooks.
- Notebook and script preprocessing are aligned through the shared helper module.
- The exploratory notebook no longer builds features from a wider temporary set that includes `AUC` and `Z_SCORE`.
- Dataset loading is now path-safe and resolved relative to the repo instead of depending on the caller's current working directory.
- A stricter evaluation path was added with grouped splits by drug or cell line.
- A new `model_comparison.py` script provides one reproducible baseline comparison workflow outside the notebooks.

## Files

- `dataset_analyzer.py`: basic missing-value, target, and cardinality summary for the canonical feature set
- `ridge_baseline.py`: Ridge baseline on the canonical feature set
- `lasso_baseline.py`: LASSO baseline on the canonical feature set
- `model_comparison.py`: compares Linear, Ridge, LASSO, and Random Forest on the same preprocessing and split logic, then selects the best validation model programmatically
- `Project.ipynb`: exploratory notebook using the canonical feature list and split utilities
- `ML_model_implementation.ipynb`: notebook version of the canonical model-comparison workflow

## Setup

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

## Run

Default random-row baseline:

```bash
python dataset_analyzer.py
python ridge_baseline.py
python lasso_baseline.py
python model_comparison.py
```

Stricter grouped splits:

```bash
python ridge_baseline.py --split drug
python ridge_baseline.py --split cell_line
python model_comparison.py --split drug
python model_comparison.py --split cell_line
```

## Notes

- `random` is still useful as a quick baseline, but it is optimistic because similar drug or cell-line contexts can land in both train and test.
- `drug` groups by `DRUG_ID`.
- `cell_line` groups by `COSMIC_ID`.
- `AUC` and `Z_SCORE` are not used in the canonical modeling pipeline.
- Notebook and script preprocessing are intentionally aligned now; results should be directly comparable when run with the same split strategy.
