import pandas as pd
import numpy as np
import os
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import json
from datetime import timedelta

def forecast_sku(model, series, scaler, last_date, window_size=30):
    series_scaled = scaler.transform(series.values.reshape(-1, 1))
    if len(series_scaled) < window_size:
        return []

    input_seq = series_scaled[-window_size:].reshape(1, window_size, 1)
    forecasts = []

    for day_ahead in range(1, 8):
        forecast_scaled = model.predict(input_seq)
        forecast_value = scaler.inverse_transform(forecast_scaled)[0][0]
        forecast_date = last_date + timedelta(days=day_ahead)
        forecasts.append((forecast_date.strftime("%Y-%m-%d"), day_ahead, round(forecast_value, 2)))
        input_seq = np.append(input_seq[:, 1:, :], [[[forecast_scaled[0][0]]]], axis=1)

    return forecasts

def run_lstm_forecast(user_folder):
    # === Paths ===
    data_path = os.path.join(user_folder, "data", "processed", "training_data.csv")
    model_path = "models/lstm/lstm_model.keras"
    metadata_path = "models/lstm/model_info.json"
    forecast_path = os.path.join(user_folder, "reports", "lstm_forecast.csv")

    # === Load and sanitize ===
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    df["date"] = pd.to_datetime(df["date"])

    required_cols = {"store_id", "product_id", "demand_forecast", "price", "inventory"}
    if not required_cols.issubset(df.columns):
        print("❌ Missing required columns in training_data.csv")
        return

    if not os.path.exists(model_path):
        print("❌ LSTM model not found:", model_path)
        return

    if not os.path.exists(metadata_path):
        print("❌ Metadata file not found:", metadata_path)
        return

    with open(metadata_path, "r") as f:
        meta = json.load(f)

    scaler = MinMaxScaler()
    scaler.fit([[meta["scaler_min"]], [meta["scaler_max"]]])
    model = load_model(model_path)
    window_size = meta["input_window"]

    # === Create synthetic SKU column ===
    df["sku"] = df["store_id"].astype(str) + "_" + df["product_id"].astype(str)

    results = []

    for sku in df["sku"].unique():
        sku_df = df[df["sku"] == sku].sort_values("date")
        if len(sku_df) < window_size:
            print(f"⚠️ Skipping {sku}: not enough data ({len(sku_df)} rows)")
            continue

        last_date = sku_df["date"].max()
        demand_series = sku_df["demand_forecast"]
        latest_row = sku_df.iloc[-1]

        forecast = forecast_sku(model, demand_series, scaler, last_date, window_size)
        for date, day_ahead, pred in forecast:
            results.append({
                "sku": sku,
                "date": date,
                "day_ahead": day_ahead,
                "lstm_forecast": pred,
                "price": latest_row["price"],
                "inventory": latest_row["inventory"]
            })

    if not results:
        print("❌ No forecasts generated.")
        return

    os.makedirs(os.path.join(user_folder, "reports"), exist_ok=True)
    pd.DataFrame(results).to_csv(forecast_path, index=False)
    print(f"✅ 7-day per-SKU LSTM forecast saved to: {forecast_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        run_lstm_forecast(sys.argv[1])
    else:
        run_lstm_forecast(".")
