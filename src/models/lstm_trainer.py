import pandas as pd
import numpy as np
import os
import json
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# === Paths ===
DATA_PATH = "data/processed/training_data.csv"
MODEL_DIR = "models/lstm"
MODEL_FILE = "lstm_model.keras"
METADATA_FILE = "model_info.json"

# === Load and sanitize ===
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
series = df["demand_forecast"].values.reshape(-1, 1)

# === Normalize ===
scaler = MinMaxScaler()
series_scaled = scaler.fit_transform(series)

# === Create sequences ===
def create_sequences(data, window=10):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i+window])
        y.append(data[i+window])
    return np.array(X), np.array(y)

X, y = create_sequences(series_scaled)

# === Build model ===
model = Sequential([
    LSTM(64, input_shape=(X.shape[1], X.shape[2])),
    Dense(1)
])
model.compile(optimizer="adam", loss="mse")
model.fit(X, y, epochs=10, batch_size=32, verbose=0)

# === Save model and metadata ===
os.makedirs(MODEL_DIR, exist_ok=True)
model.save(os.path.join(MODEL_DIR, MODEL_FILE))
metadata = {
    "trained_on": str(pd.Timestamp.today().date()),
    "model_type": "LSTM",
    "input_window": X.shape[1],
    "scaler_min": float(scaler.data_min_[0]),
    "scaler_max": float(scaler.data_max_[0])
}
with open(os.path.join(MODEL_DIR, METADATA_FILE), "w") as f:
    json.dump(metadata, f, indent=2)

print(f"✅ LSTM model trained and saved to: {MODEL_FILE}")
