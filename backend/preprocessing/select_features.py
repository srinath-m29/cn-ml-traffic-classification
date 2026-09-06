from pathlib import Path
import pandas as pd

DATASET_FILE = Path("dataset/processed/network_traffic_cleaned.csv")
FEATURE_FILE = Path("dataset/processed/selected_features.txt")

# Features that are constant across the entire dataset
CONSTANT_FEATURES = [
    "Bwd PSH Flags",
    "Bwd URG Flags",
    "Fwd Avg Bytes/Bulk",
    "Fwd Avg Packets/Bulk",
    "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk",
    "Bwd Avg Packets/Bulk",
    "Bwd Avg Bulk Rate",
]

# Clear duplicate/redundant features identified from correlation analysis
REDUNDANT_FEATURES = [
    "Subflow Fwd Packets",
    "Subflow Bwd Packets",
    "Subflow Fwd Bytes",
    "Subflow Bwd Bytes",
    "Fwd Header Length.1",
    "Avg Fwd Segment Size",
    "Avg Bwd Segment Size",
]

df = pd.read_csv(DATASET_FILE)

all_features = [column for column in df.columns if column != "Label"]

selected_features = [
    feature
    for feature in all_features
    if feature not in CONSTANT_FEATURES
    and feature not in REDUNDANT_FEATURES
]

print(f"Original features : {len(all_features)}")
print(f"Constant removed  : {len(CONSTANT_FEATURES)}")
print(f"Redundant removed : {len(REDUNDANT_FEATURES)}")
print(f"Selected features : {len(selected_features)}")

print("\nSelected features:")
for index, feature in enumerate(selected_features, start=1):
    print(f"{index:02d}. {feature}")

FEATURE_FILE.parent.mkdir(parents=True, exist_ok=True)

FEATURE_FILE.write_text(
    "\n".join(selected_features),
    encoding="utf-8"
)

print(f"\nFeature list saved to:\n{FEATURE_FILE.resolve()}")