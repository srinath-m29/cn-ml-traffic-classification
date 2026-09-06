from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

RAW_DIR = (
    BASE_DIR
    / "dataset"
    / "raw"
    / "MachineLearningCVE"
)

TRAINING_DATASET = (
    BASE_DIR
    / "dataset"
    / "processed"
    / "training_dataset.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "models"
    / "temporal_analysis"
)


# ============================================================
# FILES
# ============================================================

TRAIN_FILES = [
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
]

TEST_FILES = [
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
]


# ============================================================
# LOAD FEATURE LIST
# ============================================================

print("Loading feature list...")

header = pd.read_csv(
    TRAINING_DATASET,
    nrows=0,
)

FEATURES = [
    column
    for column in header.columns
    if column not in ["Label", "Target"]
]

print(
    f"Features available: {len(FEATURES)}"
)


# ============================================================
# IMPORTANT NETWORK FEATURES
# ============================================================

IMPORTANT_FEATURES = [
    "Destination Port",
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Fwd Packet Length Mean",
    "Bwd Packet Length Mean",
    "Bwd Packet Length Std",
    "Max Packet Length",
    "Packet Length Mean",
    "Packet Length Std",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Max",
    "Fwd IAT Mean",
    "Bwd IAT Mean",
    "Fwd Packets/s",
    "Bwd Packets/s",
    "FIN Flag Count",
    "SYN Flag Count",
    "RST Flag Count",
    "PSH Flag Count",
    "ACK Flag Count",
    "Average Packet Size",
    "Down/Up Ratio",
]


IMPORTANT_FEATURES = [
    feature
    for feature in IMPORTANT_FEATURES
    if feature in FEATURES
]

print(
    f"Important features analyzed: "
    f"{len(IMPORTANT_FEATURES)}"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_files(files, name):
    frames = []

    print(
        f"\nLoading {name} traffic..."
    )

    for filename in files:
        path = RAW_DIR / filename

        print(
            f"  Reading: {filename}"
        )

        # Read header and normalize spaces.
        raw_columns = pd.read_csv(
            path,
            nrows=0,
            encoding="latin1",
        ).columns.tolist()

        column_map = {}

        for raw_column in raw_columns:
            cleaned = raw_column.strip()

            if cleaned not in column_map:
                column_map[cleaned] = raw_column

        missing = [
            feature
            for feature in IMPORTANT_FEATURES
            if feature not in column_map
        ]

        if missing:
            raise ValueError(
                f"Missing features in {filename}: "
                f"{missing}"
            )

        use_columns = [
            column_map[feature]
            for feature in IMPORTANT_FEATURES
        ]

        df = pd.read_csv(
            path,
            usecols=use_columns,
            encoding="latin1",
            low_memory=False,
        )

        df.columns = [
            column.strip()
            for column in df.columns
        ]

        # Numeric conversion.
        for feature in IMPORTANT_FEATURES:
            df[feature] = pd.to_numeric(
                df[feature],
                errors="coerce",
            )

        # Remove infinite values.
        df = df.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        # Remove missing values.
        df = df.dropna(
            subset=IMPORTANT_FEATURES
        )

        frames.append(df)

        print(
            f"    Valid rows: {len(df):,}"
        )

    return pd.concat(
        frames,
        ignore_index=True,
    )


# ============================================================
# LOAD TRAINING AND FRIDAY DATA
# ============================================================

train_df = load_files(
    TRAIN_FILES,
    "Monday–Thursday",
)

test_df = load_files(
    TEST_FILES,
    "Friday",
)


print("\nDataset sizes:")
print(
    f"Monday–Thursday: {len(train_df):,}"
)

print(
    f"Friday         : {len(test_df):,}"
)


# ============================================================
# STATISTICAL DISTRIBUTION COMPARISON
# ============================================================

print(
    "\nCalculating traffic distribution shift..."
)

results = []

for feature in IMPORTANT_FEATURES:

    train_values = train_df[feature]
    test_values = test_df[feature]

    train_mean = train_values.mean()
    test_mean = test_values.mean()

    train_median = train_values.median()
    test_median = test_values.median()

    train_std = train_values.std()
    test_std = test_values.std()

    # Relative change in mean.
    denominator = abs(train_mean)

    if denominator < 1e-12:
        mean_change = np.nan
    else:
        mean_change = (
            abs(test_mean - train_mean)
            / denominator
            * 100
        )

    # Standardized mean difference.
    pooled_std = np.sqrt(
        (
            train_std ** 2
            + test_std ** 2
        ) / 2
    )

    if pooled_std < 1e-12:
        standardized_difference = 0
    else:
        standardized_difference = (
            abs(test_mean - train_mean)
            / pooled_std
        )

    results.append({
        "Feature": feature,
        "Train Mean": train_mean,
        "Friday Mean": test_mean,
        "Train Median": train_median,
        "Friday Median": test_median,
        "Train Std": train_std,
        "Friday Std": test_std,
        "Mean Change %": mean_change,
        "Standardized Difference":
            standardized_difference,
    })


results_df = pd.DataFrame(
    results
)


# ============================================================
# SORT BY DISTRIBUTION CHANGE
# ============================================================

results_df = results_df.sort_values(
    "Standardized Difference",
    ascending=False,
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print(
    "\n" + "=" * 100
)

print(
    "NETWORK TRAFFIC DISTRIBUTION SHIFT"
)

print(
    "=" * 100
)

display_columns = [
    "Feature",
    "Train Mean",
    "Friday Mean",
    "Mean Change %",
    "Standardized Difference",
]

print(
    results_df[
        display_columns
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


# ============================================================
# TOP FEATURES
# ============================================================

print(
    "\n" + "=" * 100
)

print(
    "TOP 10 FEATURES WITH LARGEST DISTRIBUTION SHIFT"
)

print(
    "=" * 100
)

top10 = results_df.head(10)

for index, row in enumerate(
    top10.itertuples(),
    start=1,
):
    print(
        f"{index:02d}. "
        f"{row.Feature:<35} "
        f"Change: {row._8:>10.2f}%  "
        f"Standardized: {row._9:.4f}"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

output_file = (
    OUTPUT_DIR
    / "traffic_distribution_shift.csv"
)

results_df.to_csv(
    output_file,
    index=False,
)

print(
    f"\nSaved analysis to:"
)

print(
    output_file
)


# ============================================================
# SUMMARY
# ============================================================

print(
    "\nTraffic distribution analysis completed."
)