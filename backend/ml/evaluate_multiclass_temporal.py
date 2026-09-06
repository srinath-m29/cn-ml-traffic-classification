import os
import json
import time
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

RAW_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "raw",
    "MachineLearningCVE"
)

PROCESSED_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "processed"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "xgboost_multiclass_gpu.json"
)

LABEL_MAP_PATH = os.path.join(
    MODEL_DIR,
    "multiclass_label_mapping.json"
)

FEATURE_PATH = os.path.join(
    PROCESSED_DIR,
    "selected_features.txt"
)

OUTPUT_DIR = os.path.join(
    MODEL_DIR,
    "temporal_multiclass"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# FRIDAY DATASETS
# ============================================================

FRIDAY_FILES = [
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"
]


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("MULTICLASS TEMPORAL EVALUATION")
print("=" * 70)


# ============================================================
# LOAD FEATURE LIST
# ============================================================

print("\nLoading feature list...")

if not os.path.exists(FEATURE_PATH):
    raise FileNotFoundError(
        f"Feature list not found:\n{FEATURE_PATH}"
    )

with open(FEATURE_PATH, "r", encoding="utf-8") as f:
    selected_features = [
        line.strip()
        for line in f
        if line.strip()
    ]

print(
    f"Features in selected_features.txt: "
    f"{len(selected_features)}"
)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading multiclass XGBoost model...")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found:\n{MODEL_PATH}"
    )

model = xgb.XGBClassifier()

model.load_model(MODEL_PATH)

print("Model loaded successfully.")


# ============================================================
# GET EXACT MODEL FEATURES
# ============================================================

print("\nChecking model feature configuration...")

booster = model.get_booster()

model_features = booster.feature_names

if model_features is None:
    raise RuntimeError(
        "The XGBoost model does not contain feature names."
    )

print(
    f"Model expects: {len(model_features)} features"
)

print(
    f"Exact model features used for evaluation: "
    f"{len(model_features)}"
)


# ============================================================
# PRINT MODEL FEATURES
# ============================================================

print("\nModel feature order:")

for i, feature in enumerate(model_features, 1):
    print(f"{i:02d}. {feature}")


# ============================================================
# LOAD LABEL MAPPING
# ============================================================

print("\nLoading label mapping...")

if not os.path.exists(LABEL_MAP_PATH):
    raise FileNotFoundError(
        f"Label mapping not found:\n{LABEL_MAP_PATH}"
    )

with open(
    LABEL_MAP_PATH,
    "r",
    encoding="utf-8"
) as f:
    mapping_data = json.load(f)


# ------------------------------------------------------------
# Support both possible JSON formats
# ------------------------------------------------------------

if "id_to_label" in mapping_data:

    id_to_label = {
        int(k): v
        for k, v in mapping_data["id_to_label"].items()
    }

elif "label_to_id" in mapping_data:

    id_to_label = {
        int(v): k
        for k, v in mapping_data["label_to_id"].items()
    }

else:

    # fallback for simple {"0":"BENIGN", ...} format
    try:
        id_to_label = {
            int(k): v
            for k, v in mapping_data.items()
        }
    except Exception as e:
        raise ValueError(
            "Unsupported label mapping format."
        ) from e


num_classes = len(id_to_label)

print(
    f"Number of classes: {num_classes}"
)

print("\nClass mapping:")

for class_id in sorted(id_to_label):
    print(
        f"{class_id:2d} -> "
        f"{id_to_label[class_id]}"
    )


# ============================================================
# LOAD FRIDAY DATA
# ============================================================

print("\n" + "=" * 70)
print("LOADING UNSEEN FRIDAY TRAFFIC")
print("=" * 70)


def find_file(filename):

    possible_locations = [

        os.path.join(
            RAW_DIR,
            filename
        ),

        os.path.join(
            BASE_DIR,
            "dataset",
            filename
        ),

        os.path.join(
            BASE_DIR,
            filename
        )
    ]

    for path in possible_locations:

        if os.path.exists(path):
            return path

    # recursive search as final fallback
    for root, dirs, files in os.walk(
        os.path.join(BASE_DIR, "dataset")
    ):

        if filename in files:

            return os.path.join(
                root,
                filename
            )

    return None


# ============================================================
# LOAD CSV FILES
# ============================================================

frames = []

for filename in FRIDAY_FILES:

    print(
        f"\nSearching for: {filename}"
    )

    file_path = find_file(filename)

    if file_path is None:

        raise FileNotFoundError(
            f"\nDataset file not found:\n{filename}\n\n"
            f"Searched under:\n{BASE_DIR}\\dataset"
        )

    print(
        f"Found: {file_path}"
    )

    print(
        f"\nReading: {filename}"
    )

    df = pd.read_csv(
        file_path,
        encoding="utf-8",
        encoding_errors="replace",
        low_memory=False
    )

    print(
        f"Raw rows: {len(df):,}"
    )

    frames.append(df)


# ============================================================
# COMBINE
# ============================================================

print("\nCombining Friday datasets...")

friday_df = pd.concat(
    frames,
    ignore_index=True
)

del frames

print(
    f"Combined rows: {len(friday_df):,}"
)


# ============================================================
# CLEAN COLUMN NAMES
# ============================================================

friday_df.columns = [
    str(col).strip()
    for col in friday_df.columns
]


# ============================================================
# REMOVE DUPLICATE COLUMNS
# ============================================================

if friday_df.columns.duplicated().any():

    duplicated_columns = (
        friday_df.columns[
            friday_df.columns.duplicated()
        ]
        .tolist()
    )

    print(
        "\nWARNING: Duplicate columns detected:"
    )

    print(duplicated_columns)

    friday_df = friday_df.loc[
        :,
        ~friday_df.columns.duplicated()
    ]


# ============================================================
# CHECK LABEL
# ============================================================

if "Label" not in friday_df.columns:

    raise ValueError(
        "Label column not found in Friday dataset."
    )


# ============================================================
# CLEAN LABEL
# ============================================================

friday_df["Label"] = (
    friday_df["Label"]
    .astype(str)
    .str.strip()
)


# ============================================================
# REMOVE UNKNOWN LABELS
# ============================================================

valid_labels = set(
    id_to_label.values()
)

unknown_labels = sorted(
    set(friday_df["Label"]) -
    valid_labels
)

if unknown_labels:

    print(
        "\nWARNING: Unknown labels found:"
    )

    for label in unknown_labels:
        print(
            f"  {repr(label)}"
        )

    friday_df = friday_df[
        friday_df["Label"].isin(valid_labels)
    ].copy()


# ============================================================
# CLEAN NUMERICAL FEATURES
# ============================================================

print("\nCleaning numerical features...")

# ------------------------------------------------------------
# Add missing model features
# ------------------------------------------------------------

print(
    "\nChecking required model features..."
)

missing_model_features = [
    feature
    for feature in model_features
    if feature not in friday_df.columns
]

if missing_model_features:

    print(
        "\nMissing features in Friday dataset:"
    )

    for feature in missing_model_features:
        print(
            f"  - {feature}"
        )

    print(
        "\nThese features will be added with value 0."
    )

    for feature in missing_model_features:

        friday_df[feature] = 0.0


# ============================================================
# CONVERT MODEL FEATURES TO NUMERIC
# ============================================================

print(
    "\nConverting model features to numeric..."
)

for feature in model_features:

    friday_df[feature] = pd.to_numeric(
        friday_df[feature],
        errors="coerce"
    )


# ============================================================
# INFINITE VALUES
# ============================================================

print(
    "Replacing infinite values..."
)

friday_df[model_features] = (
    friday_df[model_features]
    .replace(
        [np.inf, -np.inf],
        np.nan
    )
)


# ============================================================
# REMOVE MISSING VALUES
# ============================================================

missing_before = (
    friday_df[model_features]
    .isna()
    .any(axis=1)
    .sum()
)

print(
    f"Rows containing missing values: "
    f"{missing_before:,}"
)

if missing_before > 0:

    friday_df = friday_df.dropna(
        subset=model_features
    ).copy()


# ============================================================
# NEGATIVE VALUE CHECK
# ============================================================

print(
    "\nChecking suspicious negative values..."
)

negative_columns = [
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
    "min_seg_size_forward"
]

negative_columns = [
    col
    for col in negative_columns
    if col in model_features
]


negative_mask = (
    friday_df[negative_columns] < 0
).any(axis=1)


negative_count = int(
    negative_mask.sum()
)

print(
    f"Rows with suspicious negative values: "
    f"{negative_count:,}"
)

if negative_count > 0:

    friday_df = friday_df[
        ~negative_mask
    ].copy()


# ============================================================
# CREATE TARGET
# ============================================================

print(
    "\nChecking labels..."
)

friday_df["Target"] = (
    friday_df["Label"]
    .map(
        {
            label: class_id
            for class_id, label
            in id_to_label.items()
        }
    )
)


# ============================================================
# REMOVE UNKNOWN TARGETS
# ============================================================

unknown_target = (
    friday_df["Target"].isna()
)

if unknown_target.any():

    print(
        "Removing rows with unknown targets:"
        f" {unknown_target.sum():,}"
    )

    friday_df = friday_df[
        ~unknown_target
    ].copy()


friday_df["Target"] = (
    friday_df["Target"]
    .astype(int)
)


# ============================================================
# FINAL FEATURE ORDER
# ============================================================

print(
    "\nAligning features with trained model..."
)

# IMPORTANT:
# Use EXACT model feature order.
#
# This fixes:
#   expected Idle Max, Idle Min
#   missing Init_Win_bytes_forward/backward
#
# and prevents feature-order mismatch.

X = friday_df[
    model_features
].copy()

y = friday_df[
    "Target"
].copy()


# ============================================================
# FINAL SANITY CHECK
# ============================================================

print(
    "\nFinal feature verification:"
)

print(
    f"X shape: {X.shape}"
)

print(
    f"Expected features: "
    f"{len(model_features)}"
)

print(
    f"Actual features: "
    f"{X.shape[1]}"
)


if list(X.columns) != list(model_features):

    raise RuntimeError(
        "Feature order mismatch after alignment."
    )


# ============================================================
# DATASET SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("UNSEEN FRIDAY DATASET")
print("=" * 70)

print(
    f"Rows     : {len(X):,}"
)

print(
    f"Features : {X.shape[1]}"
)

print(
    "\nClass distribution:"
)

class_counts = (
    y.value_counts()
    .sort_index()
)

for class_id, count in class_counts.items():

    class_name = id_to_label.get(
        int(class_id),
        "UNKNOWN"
    )

    print(
        f"{class_name:<35} "
        f"{count:,}"
    )


# ============================================================
# PREDICTION
# ============================================================

print(
    "\nGenerating predictions..."
)

start_time = time.time()


# ------------------------------------------------------------
# IMPORTANT:
#
# The model is trained on CUDA but X is a CPU pandas dataframe.
#
# Calling model.predict() can trigger:
#
# "Falling back to prediction using DMatrix"
#
# This is safe but slower.
#
# We therefore explicitly use the booster prediction path.
# ------------------------------------------------------------

booster = model.get_booster()

# Convert to DMatrix with EXACT feature names

dtest = xgb.DMatrix(
    X,
    feature_names=model_features
)

probabilities = booster.predict(
    dtest
)

predictions = np.argmax(
    probabilities,
    axis=1
)


prediction_time = (
    time.time() - start_time
)

print(
    f"Prediction completed in "
    f"{prediction_time:.2f} seconds."
)


# ============================================================
# EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("UNSEEN-DAY MULTICLASS EVALUATION")
print("=" * 70)


accuracy = accuracy_score(
    y,
    predictions
)

macro_f1 = f1_score(
    y,
    predictions,
    average="macro",
    zero_division=0
)

weighted_f1 = f1_score(
    y,
    predictions,
    average="weighted",
    zero_division=0
)


print(
    f"\nAccuracy    : {accuracy:.4f}"
)

print(
    f"Macro F1    : {macro_f1:.4f}"
)

print(
    f"Weighted F1 : {weighted_f1:.4f}"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

class_ids = sorted(
    id_to_label.keys()
)

target_names = [
    id_to_label[class_id]
    for class_id in class_ids
]

print(
    "\nClassification Report:"
)

report = classification_report(
    y,
    predictions,
    labels=class_ids,
    target_names=target_names,
    zero_division=0
)

print(report)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print(
    "\nConfusion Matrix:"
)

cm = confusion_matrix(
    y,
    predictions,
    labels=class_ids
)

print(cm)


# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

cm_path = os.path.join(
    OUTPUT_DIR,
    "multiclass_temporal_confusion_matrix.csv"
)

cm_df = pd.DataFrame(
    cm,
    index=target_names,
    columns=target_names
)

cm_df.to_csv(
    cm_path,
    encoding="utf-8"
)

print(
    f"\nConfusion matrix saved to:\n"
    f"{cm_path}"
)


# ============================================================
# PER-CLASS RESULTS
# ============================================================

print(
    "\nCalculating per-class metrics..."
)

report_dict = classification_report(
    y,
    predictions,
    labels=class_ids,
    target_names=target_names,
    output_dict=True,
    zero_division=0
)

per_class_rows = []

for class_id in class_ids:

    class_name = id_to_label[class_id]

    metrics = report_dict[class_name]

    per_class_rows.append(
        {
            "class_id": class_id,
            "class": class_name,
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1-score"],
            "support": int(metrics["support"])
        }
    )


per_class_df = pd.DataFrame(
    per_class_rows
)

per_class_path = os.path.join(
    OUTPUT_DIR,
    "per_class_metrics.csv"
)

per_class_df.to_csv(
    per_class_path,
    index=False,
    encoding="utf-8"
)

print(
    f"Per-class metrics saved to:\n"
    f"{per_class_path}"
)


# ============================================================
# SAVE SUMMARY JSON
# ============================================================

summary = {
    "evaluation_type": "unseen_friday_multiclass",
    "model": MODEL_PATH,
    "features": len(model_features),
    "classes": num_classes,
    "samples": int(len(X)),
    "accuracy": float(accuracy),
    "macro_f1": float(macro_f1),
    "weighted_f1": float(weighted_f1),
    "prediction_time_seconds": float(prediction_time),
    "class_distribution": {
        id_to_label[int(k)]: int(v)
        for k, v in class_counts.items()
    }
}

summary_path = os.path.join(
    OUTPUT_DIR,
    "temporal_multiclass_metrics.json"
)

with open(
    summary_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=4,
        ensure_ascii=False
    )


print(
    f"Summary metrics saved to:\n"
    f"{summary_path}"
)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

print(
    "\nSaving predictions..."
)

prediction_output = pd.DataFrame(
    {
        "Actual_ID": y.values,
        "Actual_Label": [
            id_to_label[int(v)]
            for v in y.values
        ],
        "Predicted_ID": predictions,
        "Predicted_Label": [
            id_to_label[int(v)]
            for v in predictions
        ]
    }
)

prediction_path = os.path.join(
    OUTPUT_DIR,
    "friday_predictions.csv"
)

prediction_output.to_csv(
    prediction_path,
    index=False,
    encoding="utf-8"
)

print(
    f"Predictions saved to:\n"
    f"{prediction_path}"
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("MULTICLASS TEMPORAL EVALUATION COMPLETED")
print("=" * 70)

print(
    f"\nFeatures       : {len(model_features)}"
)

print(
    f"Classes        : {num_classes}"
)

print(
    f"Friday samples : {len(X):,}"
)

print(
    f"Accuracy       : {accuracy:.4f}"
)

print(
    f"Macro F1       : {macro_f1:.4f}"
)

print(
    f"Weighted F1    : {weighted_f1:.4f}"
)

print(
    "\nOutput directory:"
)

print(
    OUTPUT_DIR
)

print("\nDone.")