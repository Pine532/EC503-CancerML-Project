from cancer_ml_utils import FEATURE_COLS, TARGET_COL, load_model_dataframe


def main() -> None:
    model_df = load_model_dataframe()

    print("Shape of model_df:", model_df.shape)
    print("\nMissing values by column:")
    print(model_df.isnull().sum())

    print("\nTarget summary:")
    print(model_df[TARGET_COL].describe())

    print("\nUnique values per feature:")
    for col in FEATURE_COLS:
        print(f"{col}: {model_df[col].nunique(dropna=True)}")


if __name__ == "__main__":
    main()
