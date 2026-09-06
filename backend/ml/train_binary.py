from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


DATASET_FILE = Path("dataset/processed/training_dataset.csv")
MODEL_DIR = Path("models")
MODEL_FILE = MODEL_DIR / "random_forest_binary.joblib"

SAMPLE_SIZE = 500_000
RANDOM_STATE = 42


print(f"Loading dataset: {DATASET_FILE.resolve()}")

df = pd.read_csv(DATASET_FILE)

print(f"Full dataset shape: {df.shape}")


# Convert original traffic labels into binary classes.
df["Target"] = (df["Label"] != "BENIGN").astype(int)

print("\nBinary class distribution:")
print(df["Target"].value_counts())
print("\nBinary class percentages:")
print(df["Target"].value_counts(normalize=True).mul(100).round(2))


# Stratified sampling keeps the same class proportion.
if len(df) > SAMPLE_SIZE:
    _, df = train_test_split(
        df,
        train_size=SAMPLE_SIZE,
        stratify=df["Target"],
        random_state=RANDOM_STATE,
    )

print(f"\nWorking dataset shape: {df.shape}")


X = df.drop(columns=["Label", "Target"])
y = df["Target"]


print(f"Features used: {X.shape[1]}")


# Keep the class distribution consistent between training and testing.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=RANDOM_STATE,
)


print(f"\nTraining samples: {len(X_train):,}")
print(f"Testing samples : {len(X_test):,}")


print("\nTraining Random Forest...")

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    n_jobs=-1,
    random_state=RANDOM_STATE,
    class_weight="balanced",
)

model.fit(X_train, y_train)


print("\nTraining completed.")

print("\nMaking predictions...")

y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)

print(f"\nAccuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["BENIGN", "ATTACK"],
        digits=4,
    )
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


MODEL_DIR.mkdir(parents=True, exist_ok=True)

joblib.dump(model, MODEL_FILE)

print(f"\nModel saved to:")
print(MODEL_FILE.resolve())