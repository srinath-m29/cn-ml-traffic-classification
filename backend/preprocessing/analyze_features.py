from pathlib import Path
import pandas as pd
import numpy as np

DATASET_FILE = Path("dataset/processed/network_traffic_cleaned.csv")

print(f"Loading: {DATASET_FILE.resolve()}")

df = pd.read_csv(DATASET_FILE)

print("\nDataset shape:")
print(df.shape)

print("\nData types:")
print(df.dtypes.value_counts())

print("\nNon-numeric columns:")
non_numeric = df.select_dtypes(exclude=np.number).columns
for column in non_numeric:
    print(f"- {column}")

print("\nConstant features:")
feature_columns = df.drop(columns=["Label"]).columns
constant_features = []

for column in feature_columns:
    if df[column].nunique(dropna=False) <= 1:
        constant_features.append(column)

for column in constant_features:
    print(f"- {column}")

print(f"\nTotal constant features: {len(constant_features)}")

print("\nMissing values:")
missing = df.isna().sum()
missing = missing[missing > 0]

if missing.empty:
    print("No missing values")
else:
    print(missing)

print("\nInfinite values:")
numeric_df = df[feature_columns].select_dtypes(include=np.number)
infinite_count = np.isinf(numeric_df).sum().sum()
print(f"Total infinite values: {infinite_count}")

print("\nFeature summary:")
print(df[feature_columns].describe().T.to_string())

print("\nFeature analysis completed.")