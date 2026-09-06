from pathlib import Path
import pandas as pd

DATASET_FILE = Path("dataset/processed/network_traffic_cleaned.csv")
FEATURE_FILE = Path("dataset/processed/selected_features.txt")

df = pd.read_csv(DATASET_FILE)

features = FEATURE_FILE.read_text(encoding="utf-8").splitlines()

print(f"Checking {len(features)} selected features...\n")

negative_counts = {}

for feature in features:
    count = (df[feature] < 0).sum()

    if count > 0:
        negative_counts[feature] = count

print("Features containing negative values:")
print("-" * 60)

if negative_counts:
    for feature, count in negative_counts.items():
        percentage = count / len(df) * 100
        print(f"{feature:<35} {count:>10,} ({percentage:.4f}%)")
else:
    print("No negative values found.")

print("\nTotal affected features:", len(negative_counts))