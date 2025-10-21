import pandas as pd
import os
import json
from statsmodels.tsa.arima.model import ARIMA
import warnings
import joblib


# === Paths ===
DATA_PATH = "data/processed/training_data.csv"
MODEL_DIR = "models/arima"
METADATA_PATH = os.path.join(MODEL_DIR, "model_info.json")

# === Load and sanitize data ===
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
series = df["demand_forecast"]

# === Fit ARIMA model ===
warnings.filterwarnings("ignore")
model = ARIMA(series, order=(2, 1, 2))  # You can tune this
model_fit = model.fit()

# === Save metadata ===
os.makedirs(MODEL_DIR, exist_ok=True)
# Save trained model
joblib.dump(model_fit, os.path.join(MODEL_DIR, "arima_model.pkl"))
metadata = {
    "trained_on": str(pd.Timestamp.today().date()),
    "model_type": "ARIMA",
    "order": model_fit.model_orders,
    "aic": round(model_fit.aic, 2),
    "bic": round(model_fit.bic, 2)
}
with open(METADATA_PATH, "w") as f:
    json.dump(metadata, f, indent=2)
print(f"✅ ARIMA model trained on global data")
print(f"📊 AIC: {metadata['aic']}, BIC: {metadata['bic']}")
print(f"📁 Metadata saved to: {METADATA_PATH}")
