from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

RAW_DIR = BASE_DIR / "dataset" / "raw" / "MachineLearningCVE"

OUTPUT_DIR = BASE_DIR / "models" / "temporal_analysis"


# ============================================================
# DATA FILES
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
# FEATURES
# ============================================================

FEATURES = [
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


# ============================================================
# LOAD CSV FILES
# ============================================================

def load_files(files, dataset_name):
    frames = []

    print(f"\nLoading {dataset_name} traffic...")

    for filename in files:
        path = RAW_DIR / filename

        print(f"  Reading: {filename}")

        # Read only header first.
        header = pd.read_csv(
            path,
            nrows=0,
            encoding="latin1",
        )

        # Remove accidental spaces from CIC-IDS headers.
        column_map = {
            column.strip(): column
            for column in header.columns
        }

        missing = [
            feature
            for feature in FEATURES
            if feature not in column_map
        ]

        if missing:
            raise ValueError(
                f"Missing features in {filename}:\n{missing}"
            )

        original_columns = [
            column_map[feature]
            for feature in FEATURES
        ]

        df = pd.read_csv(
            path,
            usecols=original_columns,
            encoding="latin1",
            low_memory=False,
        )

        df.columns = [
            column.strip()
            for column in df.columns
        ]

        # Convert everything to numeric.
        for feature in FEATURES:
            df[feature] = pd.to_numeric(
                df[feature],
                errors="coerce",
            )

        # Replace infinite values.
        df = df.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        # Remove invalid rows.
        df = df.dropna(
            subset=FEATURES
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
# LOAD DATA
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
# SAMPLE DATA
# ============================================================
#
# KS testing does not require millions of observations.
# Sampling makes the analysis faster and easier on RAM.
#

SAMPLE_SIZE = 100_000

rng = np.random.default_rng(42)


def sample_values(series):
    values = series.to_numpy()

    if len(values) > SAMPLE_SIZE:
        indices = rng.choice(
            len(values),
            SAMPLE_SIZE,
            replace=False,
        )
        values = values[indices]

    return values


# ============================================================
# KS TEST
# ============================================================

print(
    "\nCalculating Kolmogorov–Smirnov statistics..."
)

results = []

for feature in FEATURES:

    train_values = sample_values(
        train_df[feature]
    )

    friday_values = sample_values(
        test_df[feature]
    )

    statistic, p_value = ks_2samp(
        train_values,
        friday_values,
    )

    results.append({
        "Feature": feature,
        "KS Statistic": statistic,
        "P-Value": p_value,
    })


results_df = pd.DataFrame(results)


# ============================================================
# INTERPRETATION
# ============================================================

def interpret_ks(statistic):
    if statistic >= 0.5:
        return "Very large shift"
    elif statistic >= 0.3:
        return "Large shift"
    elif statistic >= 0.2:
        return "Moderate shift"
    elif statistic >= 0.1:
        return "Small shift"
    else:
        return "Very small shift"


results_df["Interpretation"] = (
    results_df["KS Statistic"]
    .apply(interpret_ks)
)


# ============================================================
# SORT
# ============================================================

results_df = results_df.sort_values(
    "KS Statistic",
    ascending=False,
).reset_index(drop=True)


# ============================================================
# DISPLAY
# ============================================================

print(
    "\n" + "=" * 100
)

print(
    "KOLMOGOROV–SMIRNOV DISTRIBUTION SHIFT"
)

print(
    "=" * 100
)

print(
    results_df.to_string(
        index=False,
        formatters={
            "KS Statistic":
                lambda x: f"{x:.4f}",
            "P-Value":
                lambda x: f"{x:.4e}",
        },
    )
)


# ============================================================
# TOP 10
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

for index, row in results_df.head(10).iterrows():

    print(
        f"{index + 1:02d}. "
        f"{row['Feature']:<35} "
        f"KS = {row['KS Statistic']:.4f}    "
        f"p = {row['P-Value']:.2e}    "
        f"{row['Interpretation']}"
    )


# ============================================================
# SIGNIFICANT FEATURES
# ============================================================

significant = results_df[
    results_df["P-Value"] < 0.05
]

print(
    "\nStatistically significant shifts "
    f"(p < 0.05): {len(significant)} / {len(FEATURES)}"
)


# ============================================================
# SAVE
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

output_file = (
    OUTPUT_DIR
    / "ks_distribution_test.csv"
)

results_df.to_csv(
    output_file,
    index=False,
)

print(
    f"\nSaved results to:\n{output_file}"
)

print(
    "\nKS distribution analysis completed."
)