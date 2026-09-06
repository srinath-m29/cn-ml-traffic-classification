from pathlib import Path
import pandas as pd
import numpy as np

DATASET_DIR = Path("dataset/raw/MachineLearningCVE")
OUTPUT_DIR = Path("dataset/processed")
OUTPUT_FILE = OUTPUT_DIR / "network_traffic_cleaned.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

csv_files = sorted(DATASET_DIR.glob("*.csv"))

if not csv_files:
    raise FileNotFoundError(f"No CSV files found in {DATASET_DIR.resolve()}")

frames = []

for file in csv_files:
    print(f"Reading: {file.name}")
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip()
    frames.append(df)

print("\nCombining datasets...")
data = pd.concat(frames, ignore_index=True)

print(f"Original rows: {len(data):,}")
print(f"Original columns: {len(data.columns)}")

print("\nReplacing infinite values...")
data.replace([np.inf, -np.inf], np.nan, inplace=True)

print("Removing rows containing missing values...")
missing_before = data.isna().sum().sum()
data.dropna(inplace=True)
print(f"Missing values removed: {missing_before:,}")

print("\nRemoving duplicate flows...")
duplicates = data.duplicated().sum()
data.drop_duplicates(inplace=True)
print(f"Duplicate rows removed: {duplicates:,}")

print("\nFinal dataset:")
print(f"Rows    : {len(data):,}")
print(f"Columns : {len(data.columns)}")

print("\nTraffic classes:")
print(data["Label"].value_counts())

print(f"\nSaving cleaned dataset to:\n{OUTPUT_FILE.resolve()}")

data.to_csv(OUTPUT_FILE, index=False)

print("\nCleaning completed successfully.")