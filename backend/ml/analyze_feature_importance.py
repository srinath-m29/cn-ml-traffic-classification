import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = ROOT / "models" / "xgboost_multiclass_gpu.json"
OUTPUT_DIR = ROOT / "models" / "feature_importance"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = OUTPUT_DIR / "feature_importance.csv"
PLOT_PATH = OUTPUT_DIR / "feature_importance.png"
TOP_PLOT_PATH = OUTPUT_DIR / "top_20_feature_importance.png"


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("MULTICLASS XGBOOST FEATURE IMPORTANCE ANALYSIS")
print("=" * 70)


# ============================================================
# CHECK MODEL
# ============================================================

print("\nChecking model...")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"\nModel not found:\n{MODEL_PATH}"
    )

print(f"Model found: {MODEL_PATH}")


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading XGBoost multiclass model...")

model = xgb.XGBClassifier()
model.load_model(str(MODEL_PATH))

print("Model loaded successfully.")


# ============================================================
# GET BOOSTER
# ============================================================

booster = model.get_booster()

model_features = booster.feature_names

if model_features is None:
    raise ValueError(
        "The XGBoost model does not contain feature names."
    )

print(f"\nFeatures in model: {len(model_features)}")


# ============================================================
# DISPLAY FEATURE ORDER
# ============================================================

print("\nModel feature order:")

for i, feature in enumerate(model_features, start=1):
    print(f"{i:02d}. {feature}")


# ============================================================
# GET FEATURE IMPORTANCE
# ============================================================

print("\nCalculating feature importance...")

# Gain is generally the most useful importance measure
# for interpreting tree-based models.

importance_gain = booster.get_score(importance_type="gain")

importance_weight = booster.get_score(importance_type="weight")
importance_cover = booster.get_score(importance_type="cover")


# ============================================================
# BUILD COMPLETE FEATURE TABLE
# ============================================================

records = []

for feature in model_features:

    gain = importance_gain.get(feature, 0.0)
    weight = importance_weight.get(feature, 0.0)
    cover = importance_cover.get(feature, 0.0)

    records.append(
        {
            "Feature": feature,
            "Gain": float(gain),
            "Weight": float(weight),
            "Cover": float(cover),
        }
    )


importance_df = pd.DataFrame(records)


# ============================================================
# NORMALIZED GAIN
# ============================================================

total_gain = importance_df["Gain"].sum()

if total_gain > 0:
    importance_df["Gain_Percentage"] = (
        importance_df["Gain"] / total_gain * 100
    )
else:
    importance_df["Gain_Percentage"] = 0.0


# ============================================================
# SORT BY GAIN
# ============================================================

importance_df = importance_df.sort_values(
    by="Gain",
    ascending=False
).reset_index(drop=True)

importance_df["Rank"] = np.arange(
    1,
    len(importance_df) + 1
)

importance_df = importance_df[
    [
        "Rank",
        "Feature",
        "Gain",
        "Gain_Percentage",
        "Weight",
        "Cover",
    ]
]


# ============================================================
# SAVE CSV
# ============================================================

importance_df.to_csv(
    CSV_PATH,
    index=False
)

print(f"\nFeature importance saved to:")
print(CSV_PATH)


# ============================================================
# DISPLAY TOP 20
# ============================================================

print("\n" + "=" * 70)
print("TOP 20 FEATURES BY GAIN")
print("=" * 70)

top_20 = importance_df.head(20)

for _, row in top_20.iterrows():

    print(
        f"{int(row['Rank']):02d}. "
        f"{row['Feature']:<35} "
        f"Gain = {row['Gain_Percentage']:8.4f}%"
    )


# ============================================================
# PLOT ALL FEATURES
# ============================================================

print("\nGenerating complete feature importance plot...")

plot_df = importance_df.sort_values(
    by="Gain_Percentage",
    ascending=True
)

plt.figure(figsize=(12, 16))

plt.barh(
    plot_df["Feature"],
    plot_df["Gain_Percentage"]
)

plt.xlabel("Importance (%)")
plt.ylabel("Feature")
plt.title("XGBoost Multiclass Feature Importance")

plt.tight_layout()

plt.savefig(
    PLOT_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"Saved:")
print(PLOT_PATH)


# ============================================================
# TOP 20 PLOT
# ============================================================

print("\nGenerating top-20 feature importance plot...")

top_plot_df = importance_df.head(20).sort_values(
    by="Gain_Percentage",
    ascending=True
)

plt.figure(figsize=(12, 10))

plt.barh(
    top_plot_df["Feature"],
    top_plot_df["Gain_Percentage"]
)

plt.xlabel("Importance (%)")
plt.ylabel("Feature")
plt.title("Top 20 XGBoost Features by Gain")

plt.tight_layout()

plt.savefig(
    TOP_PLOT_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"Saved:")
print(TOP_PLOT_PATH)


# ============================================================
# SUMMARY JSON
# ============================================================

summary = {
    "model": str(MODEL_PATH),
    "number_of_features": len(model_features),
    "importance_method": "gain",
    "top_10_features": [
        {
            "rank": int(row["Rank"]),
            "feature": row["Feature"],
            "gain_percentage": float(row["Gain_Percentage"]),
        }
        for _, row in importance_df.head(10).iterrows()
    ],
}

SUMMARY_PATH = OUTPUT_DIR / "feature_importance_summary.json"

with open(
    SUMMARY_PATH,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        summary,
        f,
        indent=4
    )

print(f"\nSummary saved to:")
print(SUMMARY_PATH)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("FEATURE IMPORTANCE ANALYSIS COMPLETED")
print("=" * 70)

print(f"\nFeatures analyzed : {len(model_features)}")
print(f"CSV               : {CSV_PATH}")
print(f"Full plot         : {PLOT_PATH}")
print(f"Top-20 plot       : {TOP_PLOT_PATH}")
print(f"Summary           : {SUMMARY_PATH}")

print("\nDone.")