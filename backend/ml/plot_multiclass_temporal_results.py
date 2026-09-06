import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

INPUT_DIR = os.path.join(
    BASE_DIR,
    "models",
    "temporal_multiclass",
    "error_analysis"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "models",
    "temporal_multiclass",
    "plots"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# HELPER
# ============================================================

def save_plot(filename):
    path = os.path.join(OUTPUT_DIR, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ============================================================
# LOAD FILES
# ============================================================

print("=" * 70)
print("MULTICLASS TEMPORAL RESULT VISUALIZATION")
print("=" * 70)

print("\nLoading analysis results...")

report_path = os.path.join(
    INPUT_DIR,
    "classification_report.csv"
)

actual_dist_path = os.path.join(
    INPUT_DIR,
    "actual_class_distribution.csv"
)

predicted_dist_path = os.path.join(
    INPUT_DIR,
    "predicted_class_distribution.csv"
)

error_path = os.path.join(
    INPUT_DIR,
    "class_error_summary.csv"
)

confusion_path = os.path.join(
    INPUT_DIR,
    "confusion_matrix.csv"
)

coverage_path = os.path.join(
    INPUT_DIR,
    "friday_class_coverage.csv"
)


required_files = [
    report_path,
    actual_dist_path,
    predicted_dist_path,
    error_path,
    confusion_path,
    coverage_path
]

for path in required_files:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )

print("All required files found.")


# ============================================================
# 1. CLASSIFICATION REPORT
# ============================================================

print("\nLoading classification report...")

report = pd.read_csv(report_path)

print("Classification report loaded.")


# Handle possible unnamed/index columns

if "Unnamed: 0" in report.columns:
    report = report.rename(
        columns={"Unnamed: 0": "Class"}
    )

if "Class" not in report.columns:
    first_column = report.columns[0]
    report = report.rename(
        columns={first_column: "Class"}
    )


# Keep actual class rows only

metric_columns = [
    "precision",
    "recall",
    "f1-score"
]

for column in metric_columns:
    if column not in report.columns:
        raise ValueError(
            f"Missing column in classification report: {column}"
        )


# Remove aggregate rows

aggregate_names = [
    "accuracy",
    "macro avg",
    "weighted avg"
]

class_report = report[
    ~report["Class"].astype(str).isin(aggregate_names)
].copy()


# ============================================================
# 2. PER-CLASS F1 SCORE
# ============================================================

print("\nGenerating per-class F1 chart...")

class_report = class_report.sort_values(
    "f1-score",
    ascending=True
)

plt.figure(figsize=(11, 7))

plt.barh(
    class_report["Class"],
    class_report["f1-score"]
)

plt.xlabel("F1 Score")
plt.ylabel("Traffic Class")
plt.title(
    "Multiclass Temporal Evaluation - Per-Class F1 Score"
)

plt.xlim(0, 1.05)

for index, value in enumerate(class_report["f1-score"]):
    plt.text(
        value + 0.01,
        index,
        f"{value:.3f}",
        va="center",
        fontsize=9
    )

save_plot("per_class_f1.png")


# ============================================================
# 3. PER-CLASS RECALL
# ============================================================

print("\nGenerating per-class recall chart...")

class_report = class_report.sort_values(
    "recall",
    ascending=True
)

plt.figure(figsize=(11, 7))

plt.barh(
    class_report["Class"],
    class_report["recall"]
)

plt.xlabel("Recall")
plt.ylabel("Traffic Class")
plt.title(
    "Multiclass Temporal Evaluation - Per-Class Recall"
)

plt.xlim(0, 1.05)

for index, value in enumerate(class_report["recall"]):
    plt.text(
        value + 0.01,
        index,
        f"{value:.3f}",
        va="center",
        fontsize=9
    )

save_plot("per_class_recall.png")


# ============================================================
# 4. ACTUAL VS PREDICTED DISTRIBUTION
# ============================================================

print("\nLoading class distributions...")

actual = pd.read_csv(actual_dist_path)
predicted = pd.read_csv(predicted_dist_path)

if "Class" not in actual.columns:
    actual = actual.rename(
        columns={actual.columns[0]: "Class"}
    )

if "Class" not in predicted.columns:
    predicted = predicted.rename(
        columns={predicted.columns[0]: "Class"}
    )


actual_count_column = [
    column for column in actual.columns
    if "count" in column.lower()
]

predicted_count_column = [
    column for column in predicted.columns
    if "count" in column.lower()
]

if not actual_count_column:
    raise ValueError(
        "Could not identify actual count column."
    )

if not predicted_count_column:
    raise ValueError(
        "Could not identify predicted count column."
    )

actual_count_column = actual_count_column[0]
predicted_count_column = predicted_count_column[0]


distribution = pd.merge(
    actual[
        ["Class", actual_count_column]
    ],
    predicted[
        ["Class", predicted_count_column]
    ],
    on="Class",
    how="outer"
)

distribution = distribution.fillna(0)

distribution = distribution.sort_values(
    actual_count_column,
    ascending=False
)


# ============================================================
# 5. DISTRIBUTION PLOT
# ============================================================

print("\nGenerating actual vs predicted distribution chart...")

x = np.arange(len(distribution))
width = 0.38

plt.figure(figsize=(14, 7))

plt.bar(
    x - width / 2,
    distribution[actual_count_column],
    width,
    label="Actual"
)

plt.bar(
    x + width / 2,
    distribution[predicted_count_column],
    width,
    label="Predicted"
)

plt.xlabel("Traffic Class")
plt.ylabel("Number of Samples")
plt.title(
    "Friday Traffic - Actual vs Predicted Class Distribution"
)

plt.xticks(
    x,
    distribution["Class"],
    rotation=60,
    ha="right"
)

plt.legend()

save_plot("actual_vs_predicted_distribution.png")


# ============================================================
# 6. ERROR RATE BY CLASS
# ============================================================

print("\nLoading class error summary...")

error_df = pd.read_csv(error_path)

if "Class" not in error_df.columns:
    error_df = error_df.rename(
        columns={error_df.columns[0]: "Class"}
    )


required_error_columns = [
    "Total",
    "Incorrect",
    "Error_Rate"
]

for column in required_error_columns:
    if column not in error_df.columns:
        raise ValueError(
            f"Missing error-analysis column: {column}"
        )


error_df = error_df[
    error_df["Total"] > 0
].copy()

error_df = error_df.sort_values(
    "Error_Rate",
    ascending=True
)


# ============================================================
# 7. ERROR RATE PLOT
# ============================================================

print("\nGenerating error-rate chart...")

plt.figure(figsize=(11, 7))

plt.barh(
    error_df["Class"],
    error_df["Error_Rate"] * 100
)

plt.xlabel("Error Rate (%)")
plt.ylabel("Traffic Class")
plt.title(
    "Friday Traffic - Classification Error Rate by Class"
)

for index, value in enumerate(
    error_df["Error_Rate"] * 100
):
    plt.text(
        value + 0.05,
        index,
        f"{value:.3f}%",
        va="center",
        fontsize=9
    )

save_plot("error_rate_by_class.png")


# ============================================================
# 8. CONFUSION MATRIX
# ============================================================

print("\nLoading confusion matrix...")

confusion = pd.read_csv(
    confusion_path,
    index_col=0
)

print(
    f"Confusion matrix shape: {confusion.shape}"
)

# Convert values to numeric

confusion = confusion.apply(
    pd.to_numeric,
    errors="coerce"
).fillna(0)


# ============================================================
# 9. NORMALIZED CONFUSION MATRIX
# ============================================================

print("\nGenerating normalized confusion matrix...")

matrix = confusion.to_numpy(dtype=float)

row_sums = matrix.sum(axis=1, keepdims=True)

normalized = np.divide(
    matrix,
    row_sums,
    out=np.zeros_like(matrix),
    where=row_sums != 0
)

plt.figure(figsize=(12, 10))

plt.imshow(
    normalized,
    interpolation="nearest",
    aspect="auto"
)

plt.title(
    "Normalized Confusion Matrix - Temporal Multiclass Model"
)

plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")

plt.colorbar(
    label="Normalized Frequency"
)

plt.xticks(
    range(len(confusion.columns)),
    confusion.columns,
    rotation=90
)

plt.yticks(
    range(len(confusion.index)),
    confusion.index
)

# Add values only where meaningful

for i in range(normalized.shape[0]):
    for j in range(normalized.shape[1]):

        value = normalized[i, j]

        if value >= 0.01:

            plt.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=7
            )

save_plot("normalized_confusion_matrix.png")


# ============================================================
# 10. RAW CONFUSION MATRIX
# ============================================================

print("\nGenerating raw confusion matrix...")

plt.figure(figsize=(12, 10))

plt.imshow(
    matrix,
    interpolation="nearest",
    aspect="auto"
)

plt.title(
    "Confusion Matrix - Temporal Multiclass Model"
)

plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")

plt.colorbar(
    label="Number of Samples"
)

plt.xticks(
    range(len(confusion.columns)),
    confusion.columns,
    rotation=90
)

plt.yticks(
    range(len(confusion.index)),
    confusion.index
)

# Show values only for non-zero cells

for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):

        value = matrix[i, j]

        if value > 0:

            plt.text(
                j,
                i,
                f"{int(value)}",
                ha="center",
                va="center",
                fontsize=7
            )

save_plot("confusion_matrix.png")


# ============================================================
# 11. FRIDAY CLASS COVERAGE
# ============================================================

print("\nLoading Friday class coverage...")

coverage = pd.read_csv(coverage_path)

if "Class" not in coverage.columns:
    coverage = coverage.rename(
        columns={coverage.columns[0]: "Class"}
    )

if "Actual_Count" not in coverage.columns:
    raise ValueError(
        "Actual_Count column not found in coverage file."
    )

coverage = coverage.sort_values(
    "Actual_Count",
    ascending=False
)


# ============================================================
# 12. CLASS COVERAGE PLOT
# ============================================================

print("\nGenerating Friday class coverage chart...")

plt.figure(figsize=(12, 7))

plt.bar(
    coverage["Class"],
    coverage["Actual_Count"]
)

plt.xlabel("Traffic Class")
plt.ylabel("Actual Samples")
plt.title(
    "Traffic Class Coverage in Unseen Friday Dataset"
)

plt.xticks(
    rotation=60,
    ha="right"
)

save_plot("friday_class_coverage.png")


# ============================================================
# 13. IMPORTANT SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("VISUALIZATION SUMMARY")
print("=" * 70)

print(
    f"\nPlots saved to:\n{OUTPUT_DIR}"
)

print("\nGenerated files:")

files = [
    "per_class_f1.png",
    "per_class_recall.png",
    "actual_vs_predicted_distribution.png",
    "error_rate_by_class.png",
    "normalized_confusion_matrix.png",
    "confusion_matrix.png",
    "friday_class_coverage.png"
]

for filename in files:
    print(f"  - {filename}")

print("\nVisualization completed successfully.")