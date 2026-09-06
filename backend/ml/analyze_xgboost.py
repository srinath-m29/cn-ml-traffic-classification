from pathlib import Path

import pandas as pd
import xgboost as xgb


MODEL_FILE = Path("models/xgboost_binary_gpu.json")
DATASET_FILE = Path("dataset/processed/training_dataset.csv")
OUTPUT_FILE = Path("models/xgboost_feature_importance.csv")


print(f"Loading model: {MODEL_FILE.resolve()}")

model = xgb.XGBClassifier()
model.load_model(MODEL_FILE)

print("Model loaded successfully.")


print("\nLoading feature names...")

df = pd.read_csv(DATASET_FILE, nrows=1)

feature_names = [
    column
    for column in df.columns
    if column not in ["Label", "Target"]
]


importance = model.feature_importances_

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance,
})

importance_df = importance_df.sort_values(
    "Importance",
    ascending=False,
).reset_index(drop=True)


print("\nTop 20 important network-flow features:")
print("-" * 70)

for index, row in importance_df.head(20).iterrows():
    print(
        f"{index + 1:02d}. "
        f"{row['Feature']:<40} "
        f"{row['Importance']:.6f}"
    )


importance_df.to_csv(
    OUTPUT_FILE,
    index=False,
)

print("\nFull feature importance saved to:")
print(OUTPUT_FILE.resolve())