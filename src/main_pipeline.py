import os
import sys
import pandas as pd
import json
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
import joblib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.sku_forecaster import run_sku_forecast  # Intelligent forecaster

# === Step 0: Get user folder ===
if len(sys.argv) < 2:
    print("❌ Please provide a user folder path.")
    exit()

user_folder = sys.argv[1]
processed_path = os.path.join(user_folder, "data", "processed")
reports_path = os.path.join(user_folder, "reports")
os.makedirs(reports_path, exist_ok=True)

# === Step 1: Load processed data ===
try:
    training_data = pd.read_csv(os.path.join(processed_path, "training_data.csv"))
    training_features = pd.read_csv(os.path.join(processed_path, "training_features.csv"))
except Exception as e:
    print("❌ Failed to load processed data:", e)
    exit()

# === Step 2: Run per-SKU forecaster using global ARIMA model ===
print("📈 Running intelligent per-SKU forecaster...")
try:
    total_count, forecasted_count, skipped_count = run_sku_forecast(user_folder)
    forecast_path = os.path.join(reports_path, "sku_forecast.csv")
    print(f"✅ Forecasts saved to: {forecast_path}")
except Exception as e:
    print("❌ Forecasting failed:", e)
    total_count, forecasted_count, skipped_count = 0, 0, 0

# === Step 3: Save forecast summary ===
summary = {
    "total_skus": total_count,
    "forecasted": forecasted_count,
    "skipped": skipped_count,
    "timestamp": str(pd.Timestamp.now())
}
with open(os.path.join(reports_path, "forecast_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print("🧾 Forecast summary saved.")
