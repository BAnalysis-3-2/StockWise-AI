import pandas as pd
import lightgbm as lgb
import os

# === Paths ===
FEATURES_PATH = "data/processed/training_features.csv"
MODEL_PATH = "models/lgb/lgb_model.txt"
FORECAST_PATH = "reports/lgb_forecast.csv"

# === Load features ===
X = pd.read_csv(FEATURES_PATH)

# === Load model ===
model = lgb.Booster(model_file=MODEL_PATH)

# === Predict ===
y_pred = model.predict(X)

# === Save forecast ===
os.makedirs("reports", exist_ok=True)
pd.DataFrame({"forecast": y_pred}).to_csv(FORECAST_PATH, index=False)

print(f"✅ LightGBM forecast saved to: {FORECAST_PATH}")
