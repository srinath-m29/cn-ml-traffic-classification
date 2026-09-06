from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parents[2]

RAW_DIR = BASE_DIR / "dataset" / "raw" / "MachineLearningCVE"

OUTPUT_DIR = BASE_DIR / "models" / "temporal_analysis" / "plots"


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


FEATURES = [
    "Fwd Packet Length Mean",
    "Total Length of Fwd Packets",
    "PSH Flag Count",
    "Average Packet Size",
]


def load_files(files, dataset_name):
    frames = []

    print(f"\nLoading {dataset_name}...")

    for filename in files:
        path = RAW_DIR / filename

        print(f"  Reading: {filename}")

        header = pd.read_csv(
            path,
            nrows=0,
            encoding="latin1",
        )

        column_map = {
            column.strip(): column
            for column in header.columns
        }

        use_columns = [
            column_map[feature]
            for feature in FEATURES
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

        for feature in FEATURES:
            df[feature] = pd.to_numeric(
                df[feature],
                errors="coerce",
            )

        df = df.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        df = df.dropna(
            subset=FEATURES
        )

        frames.append(df)

    return pd.concat(
        frames,
        ignore_index=True,
    )


train_df = load_files(
    TRAIN_FILES,
    "Monday–Thursday traffic",
)

friday_df = load_files(
    TEST_FILES,
    "Friday traffic",
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# PLOT DISTRIBUTIONS
# ============================================================

for feature in FEATURES:

    train_values = train_df[feature].to_numpy()
    friday_values = friday_df[feature].to_numpy()

    # Remove extreme values so the plot remains readable.
    combined = np.concatenate(
        [train_values, friday_values]
    )

    lower = np.percentile(
        combined,
        1,
    )

    upper = np.percentile(
        combined,
        99,
    )

    train_values = train_values[
        (train_values >= lower)
        & (train_values <= upper)
    ]

    friday_values = friday_values[
        (friday_values >= lower)
        & (friday_values <= upper)
    ]

    plt.figure(
        figsize=(10, 6)
    )

    plt.hist(
        train_values,
        bins=80,
        density=True,
        alpha=0.5,
        label="Monday–Thursday",
    )

    plt.hist(
        friday_values,
        bins=80,
        density=True,
        alpha=0.5,
        label="Friday",
    )

    plt.xlabel(
        feature
    )

    plt.ylabel(
        "Density"
    )

    plt.title(
        f"Traffic Distribution Shift: {feature}"
    )

    plt.legend()

    plt.grid(
        alpha=0.2
    )

    filename = (
        feature
        .lower()
        .replace("/", "_per_")
        .replace(" ", "_")
    )

    output_file = (
        OUTPUT_DIR
        / f"{filename}.png"
    )

    plt.tight_layout()

    plt.savefig(
        output_file,
        dpi=200,
    )

    plt.close()

    print(
        f"Saved: {output_file}"
    )


print(
    "\nTemporal distribution plots completed."
)