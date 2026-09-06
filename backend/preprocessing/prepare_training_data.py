from pathlib import Path
import pandas as pd

DATASET_FILE = Path("dataset/processed/network_traffic_cleaned.csv")
FEATURE_FILE = Path("dataset/processed/selected_features.txt")
OUTPUT_FILE = Path("dataset/processed/training_dataset.csv")

features = FEATURE_FILE.read_text(encoding="utf-8").splitlines()

# TCP window features are excluded because of widespread negative values.
PROBLEMATIC_FEATURES = [
    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward",
]

features = [
    feature for feature in features
    if feature not in PROBLEMATIC_FEATURES
]

print(f"Final feature count: {len(features)}")

columns = features + ["Label"]

print("\nLoading required columns...")

df = pd.read_csv(DATASET_FILE, usecols=columns)

print(f"Initial rows: {len(df):,}")

# Features where negative values are considered invalid for this experiment.
NON_NEGATIVE_FEATURES = [
    "Flow Duration",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Mean",
    "Flow IAT Max",
    "Flow IAT Min",
    "Fwd IAT Min",
    "Fwd Header Length",
    "Bwd Header Length",
    "min_seg_size_forward",
]

invalid_mask = (df[NON_NEGATIVE_FEATURES] < 0).any(axis=1)

invalid_rows = invalid_mask.sum()

print(f"Rows containing suspicious negative values: {invalid_rows:,}")

df = df.loc[~invalid_mask].copy()

print(f"Rows after negative-value filtering: {len(df):,}")

print("\nFinal shape:")
print(df.shape)

print("\nClass distribution:")
print(df["Label"].value_counts())

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(OUTPUT_FILE, index=False)

print(f"\nSaved training dataset to:")
print(OUTPUT_FILE.resolve())