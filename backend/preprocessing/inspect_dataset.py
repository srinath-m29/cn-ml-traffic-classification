from pathlib import Path
import pandas as pd

DATASET_DIR = Path("dataset/raw/MachineLearningCVE")

csv_files = sorted(DATASET_DIR.glob("*.csv"))

print(f"Found {len(csv_files)} CSV files\n")

for file in csv_files:
    print("=" * 80)
    print(f"FILE: {file.name}")

    df = pd.read_csv(file)

    print(f"Rows    : {len(df):,}")
    print(f"Columns : {len(df.columns)}")

    print("\nLabels:")
    print(df[" Label"].value_counts())

    print("\nMissing values:")
    print(df.isnull().sum().sum())

    print("\nDuplicate rows:")
    print(df.duplicated().sum())