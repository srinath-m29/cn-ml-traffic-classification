import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = PROJECT_ROOT / "models"

TEMPORAL_DIR = MODEL_DIR / "temporal_multiclass"

PREDICTIONS_FILE = (
    TEMPORAL_DIR / "friday_predictions.csv"
)

LABEL_MAPPING_FILE = (
    MODEL_DIR / "multiclass_label_mapping.json"
)

OUTPUT_DIR = (
    TEMPORAL_DIR / "error_analysis"
)


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


print("=" * 70)
print("MULTICLASS TEMPORAL ERROR ANALYSIS")
print("=" * 70)


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

print("\nChecking required files...")

if not PREDICTIONS_FILE.exists():
    raise FileNotFoundError(
        f"\nPrediction file not found:\n"
        f"{PREDICTIONS_FILE}"
    )

if not LABEL_MAPPING_FILE.exists():
    raise FileNotFoundError(
        f"\nLabel mapping file not found:\n"
        f"{LABEL_MAPPING_FILE}"
    )

print(
    f"Predictions : {PREDICTIONS_FILE}"
)

print(
    f"Mapping     : {LABEL_MAPPING_FILE}"
)


# ============================================================
# LOAD LABEL MAPPING
# ============================================================

print("\nLoading label mapping...")

with open(
    LABEL_MAPPING_FILE,
    "r",
    encoding="utf-8",
) as f:
    mapping = json.load(f)


# ============================================================
# HANDLE DIFFERENT MAPPING FORMATS
# ============================================================

if "id_to_label" in mapping:

    id_to_label_raw = mapping[
        "id_to_label"
    ]

elif "label_to_id" in mapping:

    id_to_label_raw = {
        str(value): key
        for key, value
        in mapping["label_to_id"].items()
    }

else:

    id_to_label_raw = mapping


id_to_label = {
    int(key): str(value)
    for key, value
    in id_to_label_raw.items()
}


print(
    f"Number of classes: "
    f"{len(id_to_label)}"
)


print("\nClass mapping:")

for class_id in sorted(id_to_label):

    print(
        f"{class_id:2d} -> "
        f"{id_to_label[class_id]}"
    )


# ============================================================
# LOAD FRIDAY PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("LOADING FRIDAY PREDICTIONS")
print("=" * 70)


df = pd.read_csv(
    PREDICTIONS_FILE
)


print(
    f"\nRows loaded: "
    f"{len(df):,}"
)


print("\nColumns found:")

for column in df.columns:

    print(
        f"  {column}"
    )


# ============================================================
# EXPECTED COLUMNS
# ============================================================

# Your prediction file currently contains:
#
# Actual_ID
# Actual_Label
# Predicted_ID
# Predicted_Label
#
# Therefore we directly use the label columns.


if "Actual_Label" not in df.columns:

    raise ValueError(
        "\nActual_Label column not found."
        "\nAvailable columns:"
        "\n"
        + "\n".join(df.columns)
    )


if "Predicted_Label" not in df.columns:

    raise ValueError(
        "\nPredicted_Label column not found."
        "\nAvailable columns:"
        "\n"
        + "\n".join(df.columns)
    )


print("\nUsing columns:")

print(
    "Actual    : Actual_Label"
)

print(
    "Predicted : Predicted_Label"
)


# ============================================================
# NORMALIZE LABELS
# ============================================================

df["Actual_Label"] = (
    df["Actual_Label"]
    .astype(str)
    .str.strip()
)


df["Predicted_Label"] = (
    df["Predicted_Label"]
    .astype(str)
    .str.strip()
)


# ============================================================
# BASIC INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("DATASET INFORMATION")
print("=" * 70)


actual_labels = df[
    "Actual_Label"
]


predicted_labels = df[
    "Predicted_Label"
]


print(
    f"\nTotal samples: "
    f"{len(df):,}"
)


# ============================================================
# OVERALL METRICS
# ============================================================

accuracy = accuracy_score(
    actual_labels,
    predicted_labels,
)


macro_precision = precision_score(
    actual_labels,
    predicted_labels,
    average="macro",
    zero_division=0,
)


macro_recall = recall_score(
    actual_labels,
    predicted_labels,
    average="macro",
    zero_division=0,
)


macro_f1 = f1_score(
    actual_labels,
    predicted_labels,
    average="macro",
    zero_division=0,
)


weighted_f1 = f1_score(
    actual_labels,
    predicted_labels,
    average="weighted",
    zero_division=0,
)


print("\n" + "=" * 70)
print("OVERALL TEMPORAL PERFORMANCE")
print("=" * 70)


print(
    f"\nAccuracy         : "
    f"{accuracy:.4f}"
)


print(
    f"Macro Precision  : "
    f"{macro_precision:.4f}"
)


print(
    f"Macro Recall     : "
    f"{macro_recall:.4f}"
)


print(
    f"Macro F1         : "
    f"{macro_f1:.4f}"
)


print(
    f"Weighted F1      : "
    f"{weighted_f1:.4f}"
)


# ============================================================
# ACTUAL CLASS DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("ACTUAL CLASS DISTRIBUTION")
print("=" * 70)


actual_counts = (
    actual_labels
    .value_counts()
    .rename_axis("Class")
    .reset_index(
        name="Actual_Count"
    )
)


print()

print(
    actual_counts.to_string(
        index=False
    )
)


actual_counts.to_csv(
    OUTPUT_DIR
    / "actual_class_distribution.csv",
    index=False,
)


# ============================================================
# PREDICTED CLASS DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("PREDICTED CLASS DISTRIBUTION")
print("=" * 70)


predicted_counts = (
    predicted_labels
    .value_counts()
    .rename_axis("Class")
    .reset_index(
        name="Predicted_Count"
    )
)


print()

print(
    predicted_counts.to_string(
        index=False
    )
)


predicted_counts.to_csv(
    OUTPUT_DIR
    / "predicted_class_distribution.csv",
    index=False,
)


# ============================================================
# CLASS DISTRIBUTION COMPARISON
# ============================================================

distribution = pd.merge(
    actual_counts,
    predicted_counts,
    on="Class",
    how="outer",
).fillna(0)


distribution[
    "Actual_Count"
] = distribution[
    "Actual_Count"
].astype(int)


distribution[
    "Predicted_Count"
] = distribution[
    "Predicted_Count"
].astype(int)


distribution[
    "Difference"
] = (
    distribution["Predicted_Count"]
    - distribution["Actual_Count"]
)


distribution[
    "Absolute_Difference"
] = (
    distribution["Difference"]
    .abs()
)


distribution = distribution.sort_values(
    "Actual_Count",
    ascending=False,
)


print("\n" + "=" * 70)
print("ACTUAL VS PREDICTED CLASS DISTRIBUTION")
print("=" * 70)


print()

print(
    distribution.to_string(
        index=False
    )
)


distribution.to_csv(
    OUTPUT_DIR
    / "class_distribution_comparison.csv",
    index=False,
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 70)
print("PER-CLASS PERFORMANCE")
print("=" * 70)


report_dict = classification_report(
    actual_labels,
    predicted_labels,
    output_dict=True,
    zero_division=0,
)


report_df = (
    pd.DataFrame(
        report_dict
    )
    .transpose()
)


print()

print(
    report_df.to_string()
)


report_df.to_csv(
    OUTPUT_DIR
    / "classification_report.csv"
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

all_classes = sorted(
    set(
        actual_labels.unique()
    )
    |
    set(
        predicted_labels.unique()
    )
)


cm = confusion_matrix(
    actual_labels,
    predicted_labels,
    labels=all_classes,
)


cm_df = pd.DataFrame(
    cm,
    index=all_classes,
    columns=all_classes,
)


cm_df.index.name = "Actual"

cm_df.columns.name = "Predicted"


cm_df.to_csv(
    OUTPUT_DIR
    / "confusion_matrix.csv"
)


# ============================================================
# TOP CONFUSION PAIRS
# ============================================================

print("\n" + "=" * 70)
print("TOP MISCLASSIFICATION PAIRS")
print("=" * 70)


confusion_pairs = []


for i, actual_class in enumerate(
    all_classes
):

    for j, predicted_class in enumerate(
        all_classes
    ):

        # Ignore correct predictions
        if i == j:
            continue

        count = cm[i, j]

        if count > 0:

            confusion_pairs.append(
                {
                    "Actual":
                        actual_class,

                    "Predicted":
                        predicted_class,

                    "Count":
                        int(count),
                }
            )


if confusion_pairs:

    confusion_pairs_df = (
        pd.DataFrame(
            confusion_pairs
        )
        .sort_values(
            "Count",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    print()

    print(
        confusion_pairs_df
        .head(30)
        .to_string(
            index=False
        )
    )

else:

    confusion_pairs_df = pd.DataFrame(
        columns=[
            "Actual",
            "Predicted",
            "Count",
        ]
    )

    print(
        "\nNo misclassifications found."
    )


confusion_pairs_df.to_csv(
    OUTPUT_DIR
    / "top_confusion_pairs.csv",
    index=False,
)


# ============================================================
# CORRECT / INCORRECT FLAG
# ============================================================

df["Correct"] = (
    df["Actual_Label"]
    ==
    df["Predicted_Label"]
)


# ============================================================
# MISCLASSIFIED SAMPLES
# ============================================================

misclassified = df[
    ~df["Correct"]
].copy()


correct_count = int(
    df["Correct"].sum()
)


incorrect_count = int(
    len(misclassified)
)


error_rate = (
    incorrect_count
    / len(df)
)


print("\n" + "=" * 70)
print("MISCLASSIFICATION ANALYSIS")
print("=" * 70)


print(
    f"\nCorrect predictions   : "
    f"{correct_count:,}"
)


print(
    f"Incorrect predictions : "
    f"{incorrect_count:,}"
)


print(
    f"Error rate            : "
    f"{error_rate:.4%}"
)


misclassified.to_csv(
    OUTPUT_DIR
    / "misclassified_samples.csv",
    index=False,
)


# ============================================================
# ERROR RATE BY ACTUAL CLASS
# ============================================================

actual_error = (
    df.groupby(
        "Actual_Label"
    )
    .agg(
        Total=(
            "Correct",
            "size",
        ),
        Correct=(
            "Correct",
            "sum",
        ),
    )
)


actual_error["Incorrect"] = (
    actual_error["Total"]
    -
    actual_error["Correct"]
)


actual_error["Error_Rate"] = (
    actual_error["Incorrect"]
    /
    actual_error["Total"]
)


actual_error["Recall"] = (
    actual_error["Correct"]
    /
    actual_error["Total"]
)


actual_error = (
    actual_error
    .sort_values(
        "Error_Rate",
        ascending=False,
    )
)


print("\n" + "=" * 70)
print("ERROR RATE BY ACTUAL CLASS")
print("=" * 70)


print()

print(
    actual_error.to_string()
)


actual_error.to_csv(
    OUTPUT_DIR
    / "error_rate_by_actual_class.csv"
)


# ============================================================
# MOST COMMON PREDICTION FOR EACH ACTUAL CLASS
# ============================================================

print("\n" + "=" * 70)
print(
    "MOST COMMON PREDICTION "
    "FOR EACH ACTUAL CLASS"
)
print("=" * 70)


most_common_rows = []


for actual_class in all_classes:

    subset = df[
        df["Actual_Label"]
        ==
        actual_class
    ]


    if subset.empty:
        continue


    prediction_counts = (
        subset[
            "Predicted_Label"
        ]
        .value_counts()
    )


    most_common_prediction = (
        prediction_counts.index[0]
    )


    count = int(
        prediction_counts.iloc[0]
    )


    total = len(subset)


    percentage = (
        count
        /
        total
        *
        100
    )


    most_common_rows.append(
        {
            "Actual_Class":
                actual_class,

            "Most_Common_Prediction":
                most_common_prediction,

            "Count":
                count,

            "Total":
                total,

            "Percentage":
                percentage,
        }
    )


most_common_df = pd.DataFrame(
    most_common_rows
)


print()

print(
    most_common_df.to_string(
        index=False
    )
)


most_common_df.to_csv(
    OUTPUT_DIR
    / "most_common_prediction_by_class.csv",
    index=False,
)


# ============================================================
# CLASS-SPECIFIC CONFUSION ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("CLASS-SPECIFIC ERROR SUMMARY")
print("=" * 70)


class_error_rows = []


for actual_class in all_classes:

    subset = df[
        df["Actual_Label"]
        ==
        actual_class
    ]


    total = len(subset)


    if total == 0:
        continue


    correct = int(
        subset["Correct"].sum()
    )


    incorrect = (
        total
        -
        correct
    )


    recall = (
        correct
        /
        total
    )


    # Find the most common WRONG prediction

    wrong_subset = subset[
        ~subset["Correct"]
    ]


    if not wrong_subset.empty:

        wrong_counts = (
            wrong_subset[
                "Predicted_Label"
            ]
            .value_counts()
        )


        most_common_wrong = (
            wrong_counts.index[0]
        )


        most_common_wrong_count = int(
            wrong_counts.iloc[0]
        )

    else:

        most_common_wrong = "None"

        most_common_wrong_count = 0


    class_error_rows.append(
        {
            "Class":
                actual_class,

            "Total":
                total,

            "Correct":
                correct,

            "Incorrect":
                incorrect,

            "Recall":
                recall,

            "Error_Rate":
                incorrect / total,

            "Most_Common_Wrong_Prediction":
                most_common_wrong,

            "Wrong_Prediction_Count":
                most_common_wrong_count,
        }
    )


class_error_df = pd.DataFrame(
    class_error_rows
)


class_error_df = (
    class_error_df
    .sort_values(
        "Error_Rate",
        ascending=False,
    )
)


print()

print(
    class_error_df.to_string(
        index=False
    )
)


class_error_df.to_csv(
    OUTPUT_DIR
    / "class_error_summary.csv",
    index=False,
)


# ============================================================
# CHECK WHETHER PROBABILITY COLUMNS EXIST
# ============================================================

probability_columns = [
    column
    for column in df.columns
    if (
        "prob" in column.lower()
        or
        "confidence" in column.lower()
    )
]


print("\n" + "=" * 70)
print("CONFIDENCE ANALYSIS")
print("=" * 70)


if probability_columns:

    print(
        "\nProbability/confidence "
        "columns found:"
    )


    for column in probability_columns:

        print(
            f"  {column}"
        )


        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )


        print(
            f"    Mean   : "
            f"{df[column].mean():.6f}"
        )


        print(
            f"    Median : "
            f"{df[column].median():.6f}"
        )


        print(
            f"    Min    : "
            f"{df[column].min():.6f}"
        )


        print(
            f"    Max    : "
            f"{df[column].max():.6f}"
        )


    confidence_summary = []


    for column in probability_columns:

        confidence_summary.append(
            {
                "Column":
                    column,

                "Mean":
                    df[column].mean(),

                "Median":
                    df[column].median(),

                "Min":
                    df[column].min(),

                "Max":
                    df[column].max(),
            }
        )


    confidence_df = pd.DataFrame(
        confidence_summary
    )


    confidence_df.to_csv(
        OUTPUT_DIR
        / "confidence_summary.csv",
        index=False,
    )


else:

    print(
        "\nNo probability/confidence "
        "columns found."
    )


# ============================================================
# ANALYZE ACTUAL FRIDAY CLASSES
# ============================================================

print("\n" + "=" * 70)
print("FRIDAY CLASS COVERAGE")
print("=" * 70)


expected_classes = [
    id_to_label[class_id]
    for class_id
    in sorted(id_to_label)
]


coverage_rows = []


for class_name in expected_classes:

    actual_count = int(
        (
            actual_labels
            ==
            class_name
        ).sum()
    )


    predicted_count = int(
        (
            predicted_labels
            ==
            class_name
        ).sum()
    )


    coverage_rows.append(
        {
            "Class":
                class_name,

            "Actual_Count":
                actual_count,

            "Predicted_Count":
                predicted_count,

            "Present_In_Friday":
                actual_count > 0,
        }
    )


coverage_df = pd.DataFrame(
    coverage_rows
)


print()

print(
    coverage_df.to_string(
        index=False
    )
)


coverage_df.to_csv(
    OUTPUT_DIR
    / "friday_class_coverage.csv",
    index=False,
)


# ============================================================
# SUMMARY JSON
# ============================================================

summary = {
    "total_samples":
        int(len(df)),

    "correct_predictions":
        correct_count,

    "incorrect_predictions":
        incorrect_count,

    "error_rate":
        float(error_rate),

    "accuracy":
        float(accuracy),

    "macro_precision":
        float(macro_precision),

    "macro_recall":
        float(macro_recall),

    "macro_f1":
        float(macro_f1),

    "weighted_f1":
        float(weighted_f1),

    "actual_classes":
        int(
            actual_labels.nunique()
        ),

    "predicted_classes":
        int(
            predicted_labels.nunique()
        ),

    "total_model_classes":
        int(
            len(id_to_label)
        ),

    "number_of_confusion_pairs":
        int(
            len(confusion_pairs_df)
        ),
}


with open(
    OUTPUT_DIR
    / "error_analysis_summary.json",
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        summary,
        f,
        indent=4,
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("ERROR ANALYSIS COMPLETED")
print("=" * 70)


print(
    f"\nAccuracy       : "
    f"{accuracy:.4f}"
)


print(
    f"Macro F1       : "
    f"{macro_f1:.4f}"
)


print(
    f"Weighted F1    : "
    f"{weighted_f1:.4f}"
)


print(
    f"Misclassified  : "
    f"{incorrect_count:,}"
)


print(
    "\nOutput directory:"
)


print(
    OUTPUT_DIR
)


print("\nGenerated files:")


for file in sorted(
    OUTPUT_DIR.iterdir()
):

    if file.is_file():

        print(
            f"  - {file.name}"
        )


print("\nDone.")