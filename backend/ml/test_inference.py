import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = ROOT_DIR / "models" / "xgboost_multiclass_gpu.json"
MAPPING_PATH = ROOT_DIR / "models" / "multiclass_label_mapping.json"
DATASET_PATH = ROOT_DIR / "dataset" / "processed" / "training_dataset.csv"


# ============================================================
# LOAD LABEL MAPPING
# ============================================================

def load_label_mapping():
    print("\nLoading label mapping...")

    with open(MAPPING_PATH, "r", encoding="utf-8") as file:
        mapping = json.load(file)

    # Current mapping format
    if "id_to_label" in mapping:
        id_to_label = {
            int(k): v
            for k, v in mapping["id_to_label"].items()
        }

    # Alternative format
    elif all(str(k).isdigit() for k in mapping.keys()):
        id_to_label = {
            int(k): v
            for k, v in mapping.items()
        }

    else:
        raise ValueError(
            "Unable to determine ID-to-label mapping."
        )

    return id_to_label


# ============================================================
# MAIN
# ============================================================

print("=" * 70)
print("XGBOOST MULTICLASS INFERENCE TEST")
print("=" * 70)


# ============================================================
# CHECK FILES
# ============================================================

print("\nChecking required files...")

required_files = {
    "Model": MODEL_PATH,
    "Label mapping": MAPPING_PATH,
    "Dataset": DATASET_PATH,
}

for name, path in required_files.items():

    if not path.exists():
        raise FileNotFoundError(
            f"{name} not found:\n{path}"
        )

    print(f"{name}: {path}")


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading XGBoost multiclass model...")

model = xgb.XGBClassifier()

model.load_model(str(MODEL_PATH))

print("Model loaded successfully.")


# ============================================================
# GET EXACT MODEL FEATURES
# ============================================================

print("\nChecking model feature configuration...")

booster = model.get_booster()

model_features = booster.feature_names

if model_features is None:
    raise RuntimeError(
        "The trained model does not contain feature names."
    )

model_features = list(model_features)

print(
    f"Features expected by model: {len(model_features)}"
)

print("\nModel feature order:")

for index, feature in enumerate(
    model_features,
    start=1,
):
    print(f"{index:02d}. {feature}")


# ============================================================
# LOAD LABEL MAPPING
# ============================================================

id_to_label = load_label_mapping()

print(
    f"\nNumber of classes: {len(id_to_label)}"
)

print("\nClass mapping:")

for class_id in sorted(id_to_label):
    print(
        f"{class_id:2d} -> {id_to_label[class_id]}"
    )


# ============================================================
# LOAD SAMPLE DATA
# ============================================================

print("\n" + "=" * 70)
print("LOADING SAMPLE NETWORK TRAFFIC")
print("=" * 70)

print(
    f"\nReading: {DATASET_PATH.name}"
)

# Load only required features + actual label
usecols = model_features + ["Label"]

df = pd.read_csv(
    DATASET_PATH,
    usecols=usecols,
    nrows=1000,
)

print(
    f"Rows loaded: {len(df):,}"
)


# ============================================================
# CLEAN DATA
# ============================================================

print("\nCleaning numerical features...")

X = df[model_features].copy()

# Convert all features to numeric
for column in model_features:
    X[column] = pd.to_numeric(
        X[column],
        errors="coerce",
    )

print("Replacing infinite values...")

X = X.replace(
    [np.inf, -np.inf],
    np.nan,
)

# Find valid rows
valid_mask = X.notna().all(axis=1)

rows_removed = (~valid_mask).sum()

if rows_removed > 0:
    print(
        f"Rows removed because of missing values: "
        f"{rows_removed}"
    )

X = X.loc[valid_mask].copy()

y_true = df.loc[
    valid_mask,
    "Label",
].copy()

if len(X) == 0:
    raise RuntimeError(
        "No valid samples remain after cleaning."
    )

print(
    f"Valid samples: {len(X):,}"
)


# ============================================================
# VERIFY FEATURES
# ============================================================

print("\nChecking feature alignment...")

if list(X.columns) != model_features:
    raise RuntimeError(
        "Feature order mismatch between dataset and model."
    )

print(
    f"X shape: {X.shape}"
)

print(
    f"Expected features: {len(model_features)}"
)

print(
    f"Actual features:   {X.shape[1]}"
)

print("Feature alignment verified successfully.")


# ============================================================
# PREDICTION
# ============================================================

print("\n" + "=" * 70)
print("RUNNING INFERENCE")
print("=" * 70)

print(
    f"\nGenerating predictions for "
    f"{len(X):,} samples..."
)

start_time = time.perf_counter()

predicted_ids = model.predict(X)

prediction_time = (
    time.perf_counter() - start_time
)

print(
    f"Prediction completed in "
    f"{prediction_time:.4f} seconds."
)


# ============================================================
# PREDICTION PROBABILITIES
# ============================================================

print("\nGenerating class probabilities...")

start_time = time.perf_counter()

probabilities = model.predict_proba(X)

probability_time = (
    time.perf_counter() - start_time
)

print(
    f"Probability calculation completed in "
    f"{probability_time:.4f} seconds."
)


# ============================================================
# SAMPLE PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("SAMPLE PREDICTIONS")
print("=" * 70)

samples_to_show = min(
    10,
    len(X),
)

for i in range(samples_to_show):

    predicted_id = int(
        predicted_ids[i]
    )

    predicted_label = id_to_label.get(
        predicted_id,
        f"UNKNOWN({predicted_id})",
    )

    confidence = float(
        probabilities[i][predicted_id]
    )

    actual_label = str(
        y_true.iloc[i]
    )

    print(
        f"\nSample {i + 1}"
    )

    print(
        f"Actual     : {actual_label}"
    )

    print(
        f"Predicted  : {predicted_label}"
    )

    print(
        f"Confidence : {confidence:.6f}"
    )


# ============================================================
# TOP-3 PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("TOP-3 PREDICTIONS FOR FIRST SAMPLE")
print("=" * 70)

first_probabilities = probabilities[0]

top_indices = np.argsort(
    first_probabilities
)[::-1][:3]

for rank, class_id in enumerate(
    top_indices,
    start=1,
):

    class_id = int(class_id)

    label = id_to_label.get(
        class_id,
        f"UNKNOWN({class_id})",
    )

    probability = float(
        first_probabilities[class_id]
    )

    print(
        f"{rank}. "
        f"{label:<35} "
        f"{probability:.6f}"
    )


# ============================================================
# PREDICTION DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("PREDICTION DISTRIBUTION")
print("=" * 70)

prediction_counts = (
    pd.Series(predicted_ids)
    .value_counts()
    .sort_index()
)

for class_id, count in prediction_counts.items():

    class_id = int(class_id)

    label = id_to_label.get(
        class_id,
        f"UNKNOWN({class_id})",
    )

    percentage = (
        count / len(predicted_ids)
    ) * 100

    print(
        f"{label:<35} "
        f"{count:>8,} "
        f"({percentage:6.2f}%)"
    )


# ============================================================
# INFERENCE PERFORMANCE
# ============================================================

print("\n" + "=" * 70)
print("INFERENCE PERFORMANCE")
print("=" * 70)

samples_per_second = (
    len(X) / prediction_time
)

milliseconds_per_sample = (
    prediction_time
    / len(X)
    * 1000
)

print(
    f"Samples processed : {len(X):,}"
)

print(
    f"Prediction time   : "
    f"{prediction_time:.4f} seconds"
)

print(
    f"Samples / second  : "
    f"{samples_per_second:,.2f}"
)

print(
    f"Time / sample     : "
    f"{milliseconds_per_sample:.4f} ms"
)


# ============================================================
# FINAL STATUS
# ============================================================

print("\n" + "=" * 70)
print("INFERENCE TEST COMPLETED SUCCESSFULLY")
print("=" * 70)

print(
    f"\nModel     : {MODEL_PATH}"
)

print(
    f"Features  : {len(model_features)}"
)

print(
    f"Classes   : {len(id_to_label)}"
)

print(
    f"Samples   : {len(X):,}"
)

print("\nDone.")