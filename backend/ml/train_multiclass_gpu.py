import os
import json
import time
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)

from xgboost import XGBClassifier


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

DATASET_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "processed",
    "training_dataset.csv"
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

METRICS_PATH = os.path.join(
    MODEL_DIR,
    "multiclass_validation_metrics.json"
)

CONFUSION_MATRIX_PATH = os.path.join(
    MODEL_DIR,
    "multiclass_confusion_matrix.csv"
)

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# CHECK DATASET
# ============================================================

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(
        f"Training dataset not found:\n{DATASET_PATH}"
    )


# ============================================================
# LOAD FEATURE LIST FROM DATASET
# ============================================================

print("Loading feature list from processed dataset...")

dataset_columns = pd.read_csv(
    DATASET_PATH,
    nrows=0
).columns.tolist()

dataset_columns = [
    column.strip()
    for column in dataset_columns
]

if "Label" not in dataset_columns:
    raise ValueError(
        "Label column not found in training_dataset.csv"
    )

FEATURES = [
    column
    for column in dataset_columns
    if column != "Label"
]

print(f"Features used: {len(FEATURES)}")

if len(FEATURES) != 61:
    print("\nWARNING:")
    print(
        f"Expected 61 features, "
        f"but found {len(FEATURES)}."
    )

    print("\nDetected features:")

    for i, feature in enumerate(FEATURES, 1):
        print(f"{i:02d}. {feature}")

    raise ValueError(
        f"Expected 61 features but found {len(FEATURES)}."
    )


# ============================================================
# LOAD DATASET
# ============================================================

print("\nLoading training dataset...")
print(f"Reading: {os.path.basename(DATASET_PATH)}")

usecols = FEATURES + ["Label"]

df = pd.read_csv(
    DATASET_PATH,
    usecols=usecols,
    low_memory=False
)

print(f"Rows loaded: {len(df):,}")


# ============================================================
# CLEAN COLUMN NAMES
# ============================================================

df.columns = df.columns.str.strip()

df["Label"] = (
    df["Label"]
    .astype(str)
    .str.strip()
)


# ============================================================
# ORIGINAL CLASS DISTRIBUTION
# ============================================================

print("\nOriginal class distribution:")

print(
    df["Label"]
    .value_counts()
)


# ============================================================
# CLEAN NUMERICAL FEATURES
# ============================================================

print("\nCleaning numerical features...")

X = df[FEATURES].apply(
    pd.to_numeric,
    errors="coerce"
)

# Replace infinity
X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

y = df["Label"]


# ============================================================
# REMOVE INVALID ROWS
# ============================================================

print("Removing rows containing missing values...")

valid_mask = (
    X.notna().all(axis=1)
    & y.notna()
)

removed_rows = int((~valid_mask).sum())

X = X.loc[
    valid_mask
].reset_index(drop=True)

y = y.loc[
    valid_mask
].reset_index(drop=True)

print(f"Rows removed: {removed_rows:,}")
print(f"Rows remaining: {len(X):,}")


# ============================================================
# CLASS ENCODING
# ============================================================

print("\nEncoding classes...")

classes = sorted(
    y.unique()
)

label_to_id = {
    label: index
    for index, label in enumerate(classes)
}

id_to_label = {
    index: label
    for label, index in label_to_id.items()
}

y_encoded = (
    y.map(label_to_id)
    .astype(np.int32)
)

print(f"Number of classes: {len(classes)}")

print("\nClass mapping:")

for class_id, label in id_to_label.items():
    print(
        f"{class_id:2d} -> {label}"
    )


# ============================================================
# SAVE LABEL MAPPING
# ============================================================

with open(
    LABEL_MAP_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        {
            "label_to_id": label_to_id,
            "id_to_label": {
                str(k): v
                for k, v in id_to_label.items()
            }
        },
        f,
        indent=4,
        ensure_ascii=False
    )

print("\nLabel mapping saved to:")
print(LABEL_MAP_PATH)


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

print("\nCreating stratified validation split...")

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)

print(f"Training rows  : {len(X_train):,}")
print(f"Validation rows: {len(X_val):,}")


# ============================================================
# TRAINING CLASS DISTRIBUTION
# ============================================================

train_counts = np.bincount(
    y_train,
    minlength=len(classes)
)

print("\nTraining class distribution:")

for class_id, count in enumerate(train_counts):

    print(
        f"{class_id:2d} -> "
        f"{id_to_label[class_id]:35s} "
        f"{count:,}"
    )


# ============================================================
# CLASS WEIGHTS
# ============================================================

print("\nCalculating class weights...")

max_count = train_counts.max()

class_weights = {}

for class_id, count in enumerate(train_counts):

    if count == 0:
        weight = 1.0

    else:
        # Square-root weighting.
        #
        # This gives minority classes more importance
        # without giving extremely rare classes
        # excessively large weights.

        weight = np.sqrt(
            max_count / count
        )

    # Maximum weight is capped at 20.

    weight = min(
        weight,
        20.0
    )

    class_weights[class_id] = float(
        weight
    )


print("\nClass weights:")

for class_id, weight in class_weights.items():

    print(
        f"{class_id:2d} -> "
        f"{id_to_label[class_id]:35s} "
        f"weight = {weight:.4f}"
    )


# ============================================================
# CREATE SAMPLE WEIGHTS
# ============================================================

sample_weights = np.array(
    [
        class_weights[int(label)]
        for label in y_train
    ],
    dtype=np.float32
)


# ============================================================
# XGBOOST MODEL
# ============================================================

print("\n")
print("=" * 70)
print("TRAINING MULTICLASS XGBOOST ON RTX 3050")
print("=" * 70)

model = XGBClassifier(
    objective="multi:softprob",

    num_class=len(classes),

    n_estimators=600,

    max_depth=8,

    learning_rate=0.08,

    subsample=0.85,

    colsample_bytree=0.85,

    min_child_weight=3,

    gamma=0.0,

    reg_alpha=0.0,

    reg_lambda=1.0,

    tree_method="hist",

    device="cuda",

    eval_metric="mlogloss",

    random_state=42,

    n_jobs=-1
)


# ============================================================
# TRAIN MODEL
# ============================================================

print("\nStarting training...")
print("CUDA device requested.")

start_time = time.time()

model.fit(
    X_train,
    y_train,

    sample_weight=sample_weights,

    eval_set=[
        (X_val, y_val)
    ],

    verbose=50
)

training_time = (
    time.time()
    - start_time
)

print(
    f"\nTraining completed in "
    f"{training_time:.2f} seconds."
)


# ============================================================
# VALIDATION PREDICTION
# ============================================================

print("\nGenerating validation predictions...")

y_pred = model.predict(
    X_val
)

y_pred = np.asarray(
    y_pred
).astype(np.int32)


# ============================================================
# BASIC METRICS
# ============================================================

accuracy = accuracy_score(
    y_val,
    y_pred
)

macro_f1 = f1_score(
    y_val,
    y_pred,
    average="macro",
    zero_division=0
)

weighted_f1 = f1_score(
    y_val,
    y_pred,
    average="weighted",
    zero_division=0
)


# ============================================================
# RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("MULTICLASS VALIDATION RESULTS")
print("=" * 70)

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

print("\nClassification Report:\n")

report = classification_report(
    y_val,
    y_pred,

    labels=list(
        range(len(classes))
    ),

    target_names=classes,

    digits=4,

    zero_division=0
)

print(report)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_val,
    y_pred,

    labels=list(
        range(len(classes))
    )
)

print("\nConfusion Matrix:")

print(cm)


# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

cm_df = pd.DataFrame(
    cm,

    index=classes,

    columns=classes
)

cm_df.to_csv(
    CONFUSION_MATRIX_PATH
)

print("\nConfusion matrix saved to:")
print(CONFUSION_MATRIX_PATH)


# ============================================================
# SAVE MODEL
# ============================================================

model.save_model(
    MODEL_PATH
)

print("\nMulticlass model saved to:")
print(MODEL_PATH)


# ============================================================
# SAVE METRICS
# ============================================================

metrics = {
    "accuracy": float(
        accuracy
    ),

    "macro_f1": float(
        macro_f1
    ),

    "weighted_f1": float(
        weighted_f1
    ),

    "training_time_seconds": float(
        training_time
    ),

    "num_classes": int(
        len(classes)
    ),

    "num_features": int(
        len(FEATURES)
    ),

    "training_rows": int(
        len(X_train)
    ),

    "validation_rows": int(
        len(X_val)
    ),

    "classes": classes,

    "class_weights": {
        str(k): float(v)
        for k, v in class_weights.items()
    }
}


with open(
    METRICS_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        metrics,
        f,
        indent=4,
        ensure_ascii=False
    )


print("\nMetrics saved to:")
print(METRICS_PATH)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("MULTICLASS TRAINING COMPLETED SUCCESSFULLY")
print("=" * 70)

print(
    f"\nFeatures       : {len(FEATURES)}"
)

print(
    f"Classes        : {len(classes)}"
)

print(
    f"Training rows  : {len(X_train):,}"
)

print(
    f"Validation rows: {len(X_val):,}"
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
    f"\nModel:"
)

print(
    MODEL_PATH
)

print("\nDone.")