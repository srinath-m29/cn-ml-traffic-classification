import numpy as np
import xgboost as xgb

print("XGBoost version:", xgb.__version__)
print("Testing CUDA GPU...")

X = np.random.rand(10000, 10)
y = np.random.randint(0, 2, 10000)

model = xgb.XGBClassifier(
    n_estimators=10,
    max_depth=4,
    learning_rate=0.1,
    tree_method="hist",
    device="cuda",
    eval_metric="logloss",
    random_state=42,
)

model.fit(X, y)

print("\nGPU training test completed successfully.")
print("Device:", model.get_booster().attributes().get("device", "CUDA"))