from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


DATASET_FILE = Path("dataset/processed/training_dataset.csv")
MODEL_FILE = Path("models/xgboost_binary_gpu.json")
OUTPUT_DIR = Path("models/evaluation")

RANDOM_STATE = 42


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


print("Loading dataset...")

df = pd.read_csv(DATASET_FILE)

df["Target"] = (df["Label"] != "BENIGN").astype(int)


# Reproduce the same working dataset used during training.
_, df = __import__("sklearn.model_selection").model_selection.train_test_split(
    df,
    train_size=500_000,
    stratify=df["Target"],
    random_state=RANDOM_STATE,
)


X = df.drop(columns=["Label", "Target"])
y = df["Target"]


# Reproduce the same train/test split.
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=RANDOM_STATE,
)


print(f"Test samples: {len(X_test):,}")


print("\nLoading XGBoost model...")

model = xgb.XGBClassifier()

model.load_model(MODEL_FILE)

print("Model loaded.")


print("\nGenerating predictions...")

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]


# ------------------------------------------------------------------
# METRICS
# ------------------------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print("\n" + "=" * 70)
print("MODEL EVALUATION")
print("=" * 70)

print(f"\nAccuracy : {accuracy:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["BENIGN", "ATTACK"],
        digits=4,
    )
)


# ------------------------------------------------------------------
# CONFUSION MATRIX
# ------------------------------------------------------------------

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

fig, ax = plt.subplots(figsize=(7, 6))

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["BENIGN", "ATTACK"],
)

display.plot(ax=ax)

ax.set_title("XGBoost Binary Traffic Classification")

plt.tight_layout()

confusion_file = OUTPUT_DIR / "confusion_matrix.png"

plt.savefig(
    confusion_file,
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print(f"\nSaved: {confusion_file}")


# ------------------------------------------------------------------
# ROC CURVE
# ------------------------------------------------------------------

fpr, tpr, _ = roc_curve(y_test, y_prob)

fig, ax = plt.subplots(figsize=(7, 6))

ax.plot(
    fpr,
    tpr,
    label=f"XGBoost (AUC = {roc_auc:.4f})",
)

ax.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier",
)

ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve")

ax.legend()

plt.tight_layout()

roc_file = OUTPUT_DIR / "roc_curve.png"

plt.savefig(
    roc_file,
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print(f"Saved: {roc_file}")


# ------------------------------------------------------------------
# PRECISION-RECALL CURVE
# ------------------------------------------------------------------

precision, recall, _ = precision_recall_curve(
    y_test,
    y_prob,
)

fig, ax = plt.subplots(figsize=(7, 6))

ax.plot(
    recall,
    precision,
    label="XGBoost",
)

ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curve")

ax.legend()

plt.tight_layout()

pr_file = OUTPUT_DIR / "precision_recall_curve.png"

plt.savefig(
    pr_file,
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print(f"Saved: {pr_file}")


print("\nEvaluation completed successfully.")