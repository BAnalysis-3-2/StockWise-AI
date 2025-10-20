import pandas as pd
import xgboost as xgb
import os
import json
import joblib
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# === Paths ===
FEATURES_PATH = "data/processed/training_features.csv"
TARGET_PATH = "data/processed/training_data.csv"
MODEL_DIR = "models/xgb"
MODEL_FILE = "xgb_model.pkl"
METADATA_FILE = "model_info.json"

# === Ensure model directory exists ===
os.makedirs(MODEL_DIR, exist_ok=True)

# === Load and sanitize target ===
df_target = pd.read_csv(TARGET_PATH)
df_target.columns = df_target.columns.str.strip().str.lower().str.replace(" ", "_")
y = df_target["demand_forecast"]

# === Load features ===
X = pd.read_csv(FEATURES_PATH)

# === Align lengths ===
min_len = min(len(X), len(y))
X = X.iloc[:min_len].reset_index(drop=True)
y = y.iloc[:min_len].reset_index(drop=True)

# === Train/test split ===
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# === Train model ===
model = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1)
model.fit(X_train, y_train)

# === Evaluate ===
y_pred = model.predict(X_val)
mae = mean_absolute_error(y_val, y_pred)

# === Save model ===
model_path = os.path.join(MODEL_DIR, MODEL_FILE)

joblib.dump(model, model_path)

# === Save expected feature list ===
with open(os.path.join(MODEL_DIR, "features.txt"), "w") as f:
    f.write("\n".join(X.columns))


# === Save metadata ===
metadata = {
    "trained_on": str(pd.Timestamp.today().date()),
    "features": list(X.columns),
    "mae": round(mae, 2),
    "model_type": "XGBoost"
}
metadata_path = os.path.join(MODEL_DIR, METADATA_FILE)
with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=2)

# === Done ===
print(f"✅ XGBoost model trained and saved to: {model_path}")
print(f"📊 MAE on validation set: {mae:.2f}")
print(f"📁 Metadata saved to: {metadata_path}")
