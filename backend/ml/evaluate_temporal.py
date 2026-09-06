from pathlib import Path
import time

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

RAW_DIR = BASE_DIR / "dataset" / "raw" / "MachineLearningCVE"
TRAINING_DATASET = (
    BASE_DIR
    / "dataset"
    / "processed"
    / "training_dataset.csv"
)

MODEL_DIR = BASE_DIR / "models"

MODEL_FILE = (
    MODEL_DIR
    / "xgboost_temporal_binary.json"
)

RANDOM_STATE = 42


# ============================================================
# CIC-IDS-2017 TRAIN / TEST DAYS
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
# LOAD EXACT FEATURES USED BY OUR PROJECT
# ============================================================

print("Loading feature list...")

training_header = pd.read_csv(
    TRAINING_DATASET,
    nrows=0,
)

FEATURES = [
    column
    for column in training_header.columns
    if column not in ["Label", "Target"]
]

print(f"Features used: {len(FEATURES)}")


# ============================================================
# LOAD RAW CIC-IDS-2017 FILES
# Handles spaces in original column names
# ============================================================

def load_files(files, name):
    frames = []

    print(f"\nLoading {name} data...")

    for filename in files:
        path = RAW_DIR / filename

        print(f"  Reading: {filename}")

        if not path.exists():
            raise FileNotFoundError(
                f"File not found:\n{path}"
            )

        # ----------------------------------------------------
        # Read header only.
        # CIC-IDS-2017 contains extra spaces in some headers.
        # ----------------------------------------------------

        raw_columns = pd.read_csv(
            path,
            nrows=0,
            encoding="latin1",
        ).columns.tolist()

        # Map cleaned column name -> original column name.
        cleaned_to_raw = {}

        for raw_column in raw_columns:
            cleaned_column = raw_column.strip()

            if cleaned_column not in cleaned_to_raw:
                cleaned_to_raw[cleaned_column] = raw_column

        required_columns = FEATURES + ["Label"]

        missing_columns = [
            column
            for column in required_columns
            if column not in cleaned_to_raw
        ]

        if missing_columns:
            print("\nMissing columns:")
            for column in missing_columns:
                print(f"  {column}")

            raise ValueError(
                f"Required columns are missing from:\n{path}"
            )

        # ----------------------------------------------------
        # Convert our cleaned feature names to the actual
        # names present inside the raw CSV.
        # ----------------------------------------------------

        use_columns = [
            cleaned_to_raw[column]
            for column in required_columns
        ]

        # ----------------------------------------------------
        # Read required columns only.
        # ----------------------------------------------------

        df = pd.read_csv(
            path,
            usecols=use_columns,
            encoding="latin1",
            low_memory=False,
        )

        # Remove leading/trailing spaces.
        df.columns = [
            column.strip()
            for column in df.columns
        ]

        # Clean labels.
        df["Label"] = (
            df["Label"]
            .astype(str)
            .str.strip()
        )

        frames.append(df)

        print(
            f"    Rows: {len(df):,}"
        )

    result = pd.concat(
        frames,
        ignore_index=True,
    )

    return result


# ============================================================
# LOAD TRAINING DATA
# ============================================================

train_df = load_files(
    TRAIN_FILES,
    "TRAINING",
)


# ============================================================
# LOAD UNSEEN FRIDAY DATA
# ============================================================

test_df = load_files(
    TEST_FILES,
    "TEST",
)


print("\nRaw dataset sizes:")
print(
    f"Training: {len(train_df):,}"
)
print(
    f"Testing : {len(test_df):,}"
)


# ============================================================
# BINARY TARGET
# ============================================================

print("\nCreating binary targets...")

train_df["Target"] = (
    train_df["Label"] != "BENIGN"
).astype(np.int8)

test_df["Target"] = (
    test_df["Label"] != "BENIGN"
).astype(np.int8)


# ============================================================
# CONVERT FEATURES TO NUMERIC
# ============================================================

print("\nCleaning numerical features...")

X_train = train_df[FEATURES].apply(
    pd.to_numeric,
    errors="coerce",
)

X_test = test_df[FEATURES].apply(
    pd.to_numeric,
    errors="coerce",
)

y_train = train_df["Target"]

y_test = test_df["Target"]


# ============================================================
# HANDLE INFINITE VALUES
# ============================================================

print("Replacing infinite values...")

X_train = X_train.replace(
    [np.inf, -np.inf],
    np.nan,
)

X_test = X_test.replace(
    [np.inf, -np.inf],
    np.nan,
)


# ============================================================
# REMOVE MISSING VALUES
# ============================================================

print("Removing rows containing missing values...")

train_valid = X_train.notna().all(axis=1)

test_valid = X_test.notna().all(axis=1)

X_train = X_train.loc[train_valid]

y_train = y_train.loc[train_valid]

X_test = X_test.loc[test_valid]

y_test = y_test.loc[test_valid]


# ============================================================
# REMOVE SUSPICIOUS NEGATIVE VALUES
# ============================================================

negative_check_features = [
    "Flow Duration",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Mean",
    "Flow IAT Max",
    "Flow IAT Min",
    "Fwd IAT Min",
    "Fwd Header Length",
    "Bwd Header Length",
    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward",
    "min_seg_size_forward",
]

negative_check_features = [
    feature
    for feature in negative_check_features
    if feature in FEATURES
]


print(
    "\nChecking suspicious negative values..."
)

train_negative = (
    X_train[negative_check_features] < 0
).any(axis=1)

test_negative = (
    X_test[negative_check_features] < 0
).any(axis=1)


train_negative_count = train_negative.sum()

test_negative_count = test_negative.sum()


print(
    f"Training rows removed: "
    f"{train_negative_count:,}"
)

print(
    f"Testing rows removed : "
    f"{test_negative_count:,}"
)


X_train = X_train.loc[
    ~train_negative
]

y_train = y_train.loc[
    ~train_negative
]

X_test = X_test.loc[
    ~test_negative
]

y_test = y_test.loc[
    ~test_negative
]


# ============================================================
# FINAL DATASET INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("TEMPORAL DATASET")
print("=" * 70)

print(
    f"\nTraining rows: {len(X_train):,}"
)

print(
    f"Testing rows : {len(X_test):,}"
)

print(
    f"Features     : {len(FEATURES)}"
)


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print("\nTraining class distribution:")

train_distribution = (
    y_train.value_counts()
    .sort_index()
    .rename(
        index={
            0: "BENIGN",
            1: "ATTACK",
        }
    )
)

print(train_distribution)


print("\nTesting class distribution:")

test_distribution = (
    y_test.value_counts()
    .sort_index()
    .rename(
        index={
            0: "BENIGN",
            1: "ATTACK",
        }
    )
)

print(test_distribution)


# ============================================================
# CLASS WEIGHT
# ============================================================

attack_count = (
    y_train == 1
).sum()

benign_count = (
    y_train == 0
).sum()

scale_pos_weight = (
    benign_count / attack_count
)

print(
    f"\nscale_pos_weight: "
    f"{scale_pos_weight:.4f}"
)


# ============================================================
# XGBOOST MODEL
# ============================================================

print(
    "\nTraining XGBoost on RTX 3050..."
)

print("CUDA device enabled.")


model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    scale_pos_weight=scale_pos_weight,
    tree_method="hist",
    device="cuda",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)


# ============================================================
# TRAIN
# ============================================================

start_time = time.time()

model.fit(
    X_train,
    y_train,
)

training_time = (
    time.time() - start_time
)

print(
    f"\nTraining completed in "
    f"{training_time:.2f} seconds."
)


# ============================================================
# PREDICTION
# ============================================================

print(
    "\nPredicting unseen Friday traffic..."
)

y_pred = model.predict(
    X_test
)

y_prob = model.predict_proba(
    X_test
)[:, 1]


# ============================================================
# EVALUATION
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred,
)

roc_auc = roc_auc_score(
    y_test,
    y_prob,
)


print("\n" + "=" * 70)
print("UNSEEN-DAY EVALUATION")
print("=" * 70)

print(
    f"\nAccuracy : {accuracy:.4f}"
)

print(
    f"ROC-AUC  : {roc_auc:.4f}"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print(
    "\nClassification Report:"
)

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "BENIGN",
            "ATTACK",
        ],
        digits=4,
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred,
)

print(
    "\nConfusion Matrix:"
)

print(cm)


# ============================================================
# SAVE MODEL
# ============================================================

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

model.save_model(
    MODEL_FILE
)


print(
    "\nTemporal model saved to:"
)

print(
    MODEL_FILE
)


# ============================================================
# FINAL
# ============================================================

print(
    "\nTemporal evaluation completed successfully."
)