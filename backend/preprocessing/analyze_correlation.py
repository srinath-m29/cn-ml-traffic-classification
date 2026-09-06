from pathlib import Path
import pandas as pd
import numpy as np

DATASET_FILE = Path("dataset/processed/network_traffic_cleaned.csv")

print(f"Loading: {DATASET_FILE.resolve()}")

df = pd.read_csv(DATASET_FILE)

features = df.drop(columns=["Label"])

constant_features = [
    column for column in features.columns
    if features[column].nunique(dropna=False) <= 1
]

features = features.drop(columns=constant_features)

print(f"\nOriginal features: {len(df.columns) - 1}")
print(f"Constant features removed: {len(constant_features)}")
print(f"Remaining features: {len(features.columns)}")

print("\nCalculating correlation matrix...")

correlation_matrix = features.corr().abs()

upper_triangle = correlation_matrix.where(
    np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
)

threshold = 0.95

high_correlation_pairs = []

for column in upper_triangle.columns:
    for row in upper_triangle.index:
        value = upper_triangle.loc[row, column]

        if pd.notna(value) and value >= threshold:
            high_correlation_pairs.append(
                (row, column, value)
            )

high_correlation_pairs.sort(key=lambda x: x[2], reverse=True)

print(f"\nHighly correlated feature pairs (>= {threshold}):")
print(f"Total pairs: {len(high_correlation_pairs)}\n")

for feature_a, feature_b, correlation in high_correlation_pairs:
    print(f"{feature_a} <-> {feature_b} : {correlation:.4f}")

print("\nCorrelation analysis completed.")