import pandas as pd
import numpy as np
import os
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from datetime import timedelta
import json
import joblib

def forecast_lstm(series, model, scaler, last_date, window_size=30):
    series_scaled = scaler.transform(series.values.reshape(-1, 1))
    if len(series_scaled) < window_size:
        return None

    input_seq = series_scaled[-window_size:].reshape(1, window_size, 1)
    forecasts = []

    for day_ahead in range(1, 8):
        forecast_scaled = model.predict(input_seq)
        forecast_value = scaler.inverse_transform(forecast_scaled)[0][0]
        forecast_date = last_date + timedelta(days=day_ahead)
        forecasts.append((forecast_date.strftime("%Y-%m-%d"), day_ahead, round(forecast_value, 2)))
        input_seq = np.append(input_seq[:, 1:, :], [[[forecast_scaled[0][0]]]], axis=1)

    return forecasts

def forecast_arima_global(model_fit, last_date):
    try:
        preds = model_fit.forecast(steps=7)
        forecasts = []
        for i, pred in enumerate(preds, start=1):
            forecast_date = last_date + timedelta(days=i)
            forecasts.append((forecast_date.strftime("%Y-%m-%d"), i, round(pred, 2)))
        return forecasts
    except Exception as e:
        print(f"⚠️ Global ARIMA forecast failed: {e}")
        return None

def run_sku_forecast(user_folder):
    # === Paths ===
    data_path = os.path.join(user_folder, "data", "processed", "training_data.csv")
    model_path = "models/lstm/lstm_model.keras"
    metadata_path = "models/lstm/model_info.json"
    arima_path = "models/arima/arima_model.pkl"
    forecast_path = os.path.join(user_folder, "reports", "sku_forecast.csv")

    # === Load and sanitize ===
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    df["date"] = pd.to_datetime(df["date"])

    required_cols = {"store_id", "product_id", "demand_forecast", "price", "inventory"}
    if not required_cols.issubset(df.columns):
        print("❌ Missing required columns in training_data.csv")
        return 0, 0, 0

    # === Load LSTM model and scaler ===
    if not os.path.exists(model_path) or not os.path.exists(metadata_path):
        print("❌ LSTM model or metadata missing — skipping LSTM")
        model = None
        scaler = None
        window_size = None
    else:
        model = load_model(model_path)
        with open(metadata_path, "r") as f:
            meta = json.load(f)
        scaler = MinMaxScaler()
        scaler.fit([[meta["scaler_min"]], [meta["scaler_max"]]])
        window_size = meta["input_window"]

    # === Load global ARIMA model ===
    if os.path.exists(arima_path):
        try:
            arima_model_fit = joblib.load(arima_path)
        except Exception as e:
            print(f"❌ Failed to load global ARIMA model: {e}")
            arima_model_fit = None
    else:
        print("❌ Global ARIMA model not found.")
        arima_model_fit = None

    df["sku"] = df["store_id"].astype(str) + "_" + df["product_id"].astype(str)
    results = []

    total_count = 0
    forecasted_count = 0
    skipped_count = 0

    for sku in df["sku"].unique():
        total_count += 1
        sku_df = df[df["sku"] == sku].sort_values("date")
        row_count = len(sku_df)
        last_date = sku_df["date"].max()
        demand_series = sku_df["demand_forecast"]
        latest_row = sku_df.iloc[-1]

        forecast = None
        source_model = None

        # Try LSTM
        if model and scaler and row_count >= window_size:
            forecast = forecast_lstm(demand_series, model, scaler, last_date, window_size)
            source_model = "LSTM"

        # Try global ARIMA
        if forecast is None and arima_model_fit and row_count >= 6:
            forecast = forecast_arima_global(arima_model_fit, last_date)
            source_model = "ARIMA"

        if forecast is None:
            print(f"⚠️ Skipping {sku}: not enough data ({row_count} rows)")
            skipped_count += 1
            continue

        forecasted_count += 1
        for date, day_ahead, pred in forecast:
            results.append({
                "sku": sku,
                "date": date,
                "day_ahead": day_ahead,
                "forecast": pred,
                "source_model": source_model,
                "price": latest_row["price"],
                "inventory": latest_row["inventory"]
            })

    if not results:
        print("❌ No forecasts generated.")
        return total_count, forecasted_count, skipped_count

    os.makedirs(os.path.join(user_folder, "reports"), exist_ok=True)
    pd.DataFrame(results).to_csv(forecast_path, index=False)
    print(f"✅ Forecasts saved to: {forecast_path}")

    return total_count, forecasted_count, skipped_count

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        run_sku_forecast(sys.argv[1])
    else:
        run_sku_forecast(".")
