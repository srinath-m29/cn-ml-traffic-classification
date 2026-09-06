from pathlib import Path
import time

import joblib
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


DATASET_FILE = Path("dataset/processed/training_dataset.csv")
MODEL_DIR = Path("models")
MODEL_FILE = MODEL_DIR / "xgboost_binary_gpu.json"

RANDOM_STATE = 42


print(f"Loading dataset: {DATASET_FILE.resolve()}")

df = pd.read_csv(DATASET_FILE)

print(f"Full dataset shape: {df.shape}")


# Binary classification:
# BENIGN = 0
# Any attack = 1
df["Target"] = (df["Label"] != "BENIGN").astype(int)

print("\nBinary class distribution:")
print(df["Target"].value_counts())

print("\nBinary class percentages:")
print(df["Target"].value_counts(normalize=True).mul(100).round(2))


# Match the CPU Random Forest experiment:
# retain approximately 80% of the original dataset.
_, df = train_test_split(
    df,
    train_size=500_000,
    stratify=df["Target"],
    random_state=RANDOM_STATE,
)

print(f"\nWorking dataset shape: {df.shape}")


X = df.drop(columns=["Label", "Target"])
y = df["Target"]

print(f"Features used: {X.shape[1]}")


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=RANDOM_STATE,
)


print(f"\nTraining samples: {len(X_train):,}")
print(f"Testing samples : {len(X_test):,}")


# Account for class imbalance.
negative = (y_train == 0).sum()
positive = (y_train == 1).sum()

scale_pos_weight = negative / positive

print(f"\nscale_pos_weight: {scale_pos_weight:.4f}")

print("\nTraining XGBoost on RTX 3050...")
print("CUDA device enabled.")

start_time = time.perf_counter()

model = XGBClassifier(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    device="cuda",
    scale_pos_weight=scale_pos_weight,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

model.fit(
    X_train,
    y_train,
    eval_set=[(X_test, y_test)],
    verbose=False,
)

training_time = time.perf_counter() - start_time

print(f"\nTraining completed in {training_time:.2f} seconds.")


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

model.save_model(MODEL_FILE)

print("\nModel saved to:")
print(MODEL_FILE.resolve())