import pandas as pd
import lightgbm as lgb
import os
import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# === Paths ===
FEATURES_PATH = "data/processed/training_features.csv"
TARGET_PATH = "data/processed/training_data.csv"
MODEL_DIR = "models/lgb"
MODEL_FILE = "lgb_model.pkl"
METADATA_FILE = "model_info.json"

# === Load and sanitize ===
X = pd.read_csv(FEATURES_PATH)
df_target = pd.read_csv(TARGET_PATH)
df_target.columns = df_target.columns.str.strip().str.lower().str.replace(" ", "_")
y = df_target["demand_forecast"]

# === Align lengths ===
min_len = min(len(X), len(y))
X = X.iloc[:min_len].reset_index(drop=True)
y = y.iloc[:min_len].reset_index(drop=True)

# === Split ===
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# === Train model ===
model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1)
model.fit(X_train, y_train)

# === Evaluate ===
y_pred = model.predict(X_val)
mae = mean_absolute_error(y_val, y_pred)

# === Save model and metadata ===
os.makedirs(MODEL_DIR, exist_ok=True)

model_path = os.path.join(MODEL_DIR, MODEL_FILE)
joblib.dump(model, model_path)

# === Save expected feature list ===
with open(os.path.join(MODEL_DIR, "features.txt"), "w") as f:
    f.write("\n".join(X.columns))

metadata = {
    "trained_on": str(pd.Timestamp.today().date()),
    "features": list(X.columns),
    "mae": round(mae, 2),
    "model_type": "LightGBM"
}
with open(os.path.join(MODEL_DIR, METADATA_FILE), "w") as f:
    json.dump(metadata, f, indent=2)

print(f"✅ LightGBM model trained and saved to: {MODEL_FILE}")
print(f"📊 MAE on validation set: {mae:.2f}")
