import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import xgboost as xgb


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = ROOT / "models" / "xgboost_multiclass_gpu.json"
DATASET_PATH = ROOT / "dataset" / "processed" / "training_dataset.csv"
MAPPING_PATH = ROOT / "models" / "multiclass_label_mapping.json"

OUTPUT_DIR = ROOT / "models" / "shap_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

SAMPLE_SIZE = 5000
RANDOM_STATE = 42
TOP_FEATURES = 20


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("XGBOOST MULTICLASS SHAP ANALYSIS")
print("=" * 70)


# ============================================================
# CHECK FILES
# ============================================================

print("\nChecking required files...")

for path in [MODEL_PATH, DATASET_PATH, MAPPING_PATH]:
    if not path.exists():
        raise FileNotFoundError(
            f"\nRequired file not found:\n{path}"
        )

print(f"Model   : {MODEL_PATH}")
print(f"Dataset : {DATASET_PATH}")
print(f"Mapping : {MAPPING_PATH}")


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading XGBoost model...")

model = xgb.XGBClassifier()
model.load_model(str(MODEL_PATH))

booster = model.get_booster()

model_features = booster.feature_names

if model_features is None:
    raise ValueError(
        "Model does not contain feature names."
    )

print(f"Model features: {len(model_features)}")


# ============================================================
# LOAD LABEL MAPPING
# ============================================================

print("\nLoading label mapping...")

with open(
    MAPPING_PATH,
    "r",
    encoding="utf-8"
) as f:
    mapping = json.load(f)


# Support both mapping formats used in the project
if "id_to_label" in mapping:
    id_to_label = {
        int(k): v
        for k, v in mapping["id_to_label"].items()
    }

elif "label_to_id" in mapping:
    id_to_label = {
        int(v): k
        for k, v in mapping["label_to_id"].items()
    }

else:
    id_to_label = {
        int(k): v
        for k, v in mapping.items()
        if str(k).isdigit()
    }


print(f"Number of classes: {len(id_to_label)}")

for class_id in sorted(id_to_label):
    print(f"{class_id:2d} -> {id_to_label[class_id]}")


# ============================================================
# LOAD DATASET
# ============================================================

print("\n" + "=" * 70)
print("LOADING SAMPLE DATA")
print("=" * 70)

print(f"\nReading: {DATASET_PATH.name}")

df = pd.read_csv(
    DATASET_PATH,
    nrows=100000
)

print(f"Rows initially loaded: {len(df):,}")


# ============================================================
# VERIFY FEATURES
# ============================================================

print("\nChecking model features...")

missing_features = [
    feature
    for feature in model_features
    if feature not in df.columns
]

if missing_features:
    raise ValueError(
        "Dataset is missing model features:\n"
        + "\n".join(missing_features)
    )

print("All 61 model features found.")


# ============================================================
# PREPARE DATA
# ============================================================

print("\nPreparing numerical features...")

X = df[model_features].copy()

for column in model_features:
    X[column] = pd.to_numeric(
        X[column],
        errors="coerce"
    )

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

valid_mask = X.notna().all(axis=1)

X = X.loc[valid_mask].reset_index(drop=True)

print(f"Valid rows: {len(X):,}")


# ============================================================
# SAMPLE DATA
# ============================================================

if len(X) > SAMPLE_SIZE:

    print(
        f"\nSampling {SAMPLE_SIZE:,} rows "
        f"for SHAP analysis..."
    )

    X_sample = X.sample(
        n=SAMPLE_SIZE,
        random_state=RANDOM_STATE
    ).reset_index(drop=True)

else:

    X_sample = X.copy()

print(
    f"SHAP samples: {len(X_sample):,}"
)

print(
    f"Features: {X_sample.shape[1]}"
)


# ============================================================
# SHAP EXPLAINER
# ============================================================

print("\n" + "=" * 70)
print("CREATING SHAP EXPLAINER")
print("=" * 70)

print("\nCreating TreeExplainer...")

explainer = shap.TreeExplainer(model)

print("SHAP explainer created successfully.")


# ============================================================
# CALCULATE SHAP VALUES
# ============================================================

print("\nCalculating SHAP values...")

start_time = time.time()

shap_values = explainer.shap_values(
    X_sample
)

elapsed = time.time() - start_time

print(
    f"SHAP calculation completed in "
    f"{elapsed:.2f} seconds."
)


# ============================================================
# HANDLE SHAP OUTPUT FORMAT
# ============================================================

print("\nChecking SHAP output format...")

if isinstance(shap_values, list):

    print(
        f"SHAP returned a list with "
        f"{len(shap_values)} class arrays."
    )

    class_shap_values = shap_values

else:

    print(
        f"SHAP returned array with shape: "
        f"{np.asarray(shap_values).shape}"
    )

    shap_array = np.asarray(shap_values)

    # Possible format:
    # samples x features x classes
    if shap_array.ndim == 3:

        if shap_array.shape[2] == len(id_to_label):

            class_shap_values = [
                shap_array[:, :, class_id]
                for class_id in range(
                    len(id_to_label)
                )
            ]

        elif shap_array.shape[1] == len(id_to_label):

            class_shap_values = [
                shap_array[:, class_id, :]
                for class_id in range(
                    len(id_to_label)
                )
            ]

        else:
            raise ValueError(
                "Unable to determine SHAP class dimension."
            )

    elif shap_array.ndim == 2:

        # Binary/single-output fallback
        class_shap_values = [shap_array]

    else:

        raise ValueError(
            "Unexpected SHAP output dimensions."
        )


# ============================================================
# GLOBAL SHAP IMPORTANCE
# ============================================================

print("\n" + "=" * 70)
print("GLOBAL SHAP FEATURE IMPORTANCE")
print("=" * 70)

all_class_values = []

for values in class_shap_values:

    values = np.asarray(values)

    if values.ndim != 2:
        continue

    all_class_values.append(
        np.abs(values)
    )


if not all_class_values:
    raise ValueError(
        "No valid SHAP arrays were produced."
    )


combined_abs_shap = np.mean(
    np.stack(all_class_values),
    axis=0
)

mean_abs_shap = np.mean(
    combined_abs_shap,
    axis=0
)


global_df = pd.DataFrame(
    {
        "Feature": model_features,
        "Mean_Absolute_SHAP": mean_abs_shap,
    }
)

global_df = global_df.sort_values(
    "Mean_Absolute_SHAP",
    ascending=False
).reset_index(drop=True)

global_df["Rank"] = np.arange(
    1,
    len(global_df) + 1
)

total_shap = global_df[
    "Mean_Absolute_SHAP"
].sum()

if total_shap > 0:

    global_df["Importance_Percentage"] = (
        global_df["Mean_Absolute_SHAP"]
        / total_shap
        * 100
    )

else:

    global_df["Importance_Percentage"] = 0.0


global_df = global_df[
    [
        "Rank",
        "Feature",
        "Mean_Absolute_SHAP",
        "Importance_Percentage",
    ]
]


# ============================================================
# SAVE GLOBAL CSV
# ============================================================

GLOBAL_CSV = (
    OUTPUT_DIR /
    "global_shap_feature_importance.csv"
)

global_df.to_csv(
    GLOBAL_CSV,
    index=False
)

print("\nGlobal SHAP results saved:")
print(GLOBAL_CSV)


# ============================================================
# PRINT TOP FEATURES
# ============================================================

print(
    f"\nTOP {TOP_FEATURES} FEATURES BY SHAP IMPORTANCE"
)

for _, row in global_df.head(
    TOP_FEATURES
).iterrows():

    print(
        f"{int(row['Rank']):02d}. "
        f"{row['Feature']:<35} "
        f"SHAP = "
        f"{row['Importance_Percentage']:8.4f}%"
    )


# ============================================================
# GLOBAL SHAP BAR PLOT
# ============================================================

print("\nGenerating global SHAP plot...")

top_global = global_df.head(
    TOP_FEATURES
).sort_values(
    "Importance_Percentage",
    ascending=True
)

plt.figure(
    figsize=(12, 10)
)

plt.barh(
    top_global["Feature"],
    top_global["Importance_Percentage"]
)

plt.xlabel(
    "Mean Absolute SHAP Importance (%)"
)

plt.ylabel(
    "Feature"
)

plt.title(
    "Top 20 Features by Global SHAP Importance"
)

plt.tight_layout()

GLOBAL_PLOT = (
    OUTPUT_DIR /
    "global_shap_importance.png"
)

plt.savefig(
    GLOBAL_PLOT,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"Saved: {GLOBAL_PLOT}")


# ============================================================
# SHAP SUMMARY PLOT
# ============================================================

print("\nGenerating SHAP summary plot...")

# Use the mean magnitude across classes for a
# multiclass global explanation.

mean_shap_signed = np.mean(
    np.stack(
        [
            np.asarray(v)
            for v in class_shap_values
            if np.asarray(v).ndim == 2
        ]
    ),
    axis=0
)

plt.figure(
    figsize=(12, 10)
)

shap.summary_plot(
    mean_shap_signed,
    X_sample,
    feature_names=model_features,
    max_display=TOP_FEATURES,
    show=False
)

plt.tight_layout()

SUMMARY_PLOT = (
    OUTPUT_DIR /
    "shap_summary_plot.png"
)

plt.savefig(
    SUMMARY_PLOT,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"Saved: {SUMMARY_PLOT}")


# ============================================================
# PER-CLASS SHAP IMPORTANCE
# ============================================================

print("\n" + "=" * 70)
print("PER-CLASS SHAP ANALYSIS")
print("=" * 70)

per_class_dir = OUTPUT_DIR / "per_class"

per_class_dir.mkdir(
    parents=True,
    exist_ok=True
)

for class_id, values in enumerate(
    class_shap_values
):

    values = np.asarray(values)

    if values.ndim != 2:
        continue

    if class_id not in id_to_label:
        class_name = f"Class_{class_id}"
    else:
        class_name = id_to_label[class_id]

    class_importance = np.mean(
        np.abs(values),
        axis=0
    )

    class_df = pd.DataFrame(
        {
            "Feature": model_features,
            "Mean_Absolute_SHAP": class_importance,
        }
    )

    class_df = class_df.sort_values(
        "Mean_Absolute_SHAP",
        ascending=False
    ).reset_index(drop=True)

    class_df["Rank"] = np.arange(
        1,
        len(class_df) + 1
    )

    safe_name = (
        class_name
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace(" ", "")
    )

    csv_path = (
        per_class_dir /
        f"class_{class_id}_{safe_name}.csv"
    )

    class_df.to_csv(
        csv_path,
        index=False
    )

    # Top 10 plot
    top_class = class_df.head(
        10
    ).sort_values(
        "Mean_Absolute_SHAP",
        ascending=True
    )

    plt.figure(
        figsize=(12, 7)
    )

    plt.barh(
        top_class["Feature"],
        top_class["Mean_Absolute_SHAP"]
    )

    plt.xlabel(
        "Mean Absolute SHAP Value"
    )

    plt.ylabel(
        "Feature"
    )

    plt.title(
        f"Top SHAP Features - {class_name}"
    )

    plt.tight_layout()

    plot_path = (
        per_class_dir /
        f"class_{class_id}_{safe_name}.png"
    )

    plt.savefig(
        plot_path,
        dpi=250,
        bbox_inches="tight"
    )

    plt.close()


print(
    f"Per-class results saved to:\n"
    f"{per_class_dir}"
)


# ============================================================
# SUMMARY JSON
# ============================================================

summary = {
    "model": str(MODEL_PATH),
    "dataset": str(DATASET_PATH),
    "shap_version": shap.__version__,
    "samples_analyzed": int(len(X_sample)),
    "features_analyzed": int(len(model_features)),
    "number_of_classes": int(len(id_to_label)),
    "calculation_time_seconds": float(elapsed),
    "top_features": [
        {
            "rank": int(row["Rank"]),
            "feature": row["Feature"],
            "importance_percentage": float(
                row["Importance_Percentage"]
            ),
        }
        for _, row in global_df.head(10).iterrows()
    ],
}


SUMMARY_JSON = (
    OUTPUT_DIR /
    "shap_analysis_summary.json"
)

with open(
    SUMMARY_JSON,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=4
    )


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("SHAP ANALYSIS COMPLETED SUCCESSFULLY")
print("=" * 70)

print(
    f"\nSamples analyzed : {len(X_sample):,}"
)

print(
    f"Features         : {len(model_features)}"
)

print(
    f"Classes          : {len(id_to_label)}"
)

print(
    f"SHAP version     : {shap.__version__}"
)

print(
    f"\nOutput directory:"
)

print(OUTPUT_DIR)

print("\nGenerated:")
print("  - global_shap_feature_importance.csv")
print("  - global_shap_importance.png")
print("  - shap_summary_plot.png")
print("  - per_class/")
print("  - shap_analysis_summary.json")

print("\nDone.")