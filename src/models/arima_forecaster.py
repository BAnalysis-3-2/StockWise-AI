import pandas as pd
import os
from statsmodels.tsa.arima.model import ARIMA
from datetime import timedelta

def forecast_arima(series, last_date):
    forecasts = []
    try:
        model = ARIMA(series, order=(5, 1, 0))
        model_fit = model.fit()
        preds = model_fit.forecast(steps=7)
        for i, pred in enumerate(preds, start=1):
            forecast_date = last_date + timedelta(days=i)
            forecasts.append((forecast_date.strftime("%Y-%m-%d"), i, round(pred, 2)))
    except Exception as e:
        print(f"⚠️ ARIMA failed: {e}")
    return forecasts

def run_arima_forecast(user_folder):
    # === Paths ===
    data_path = os.path.join(user_folder, "data", "processed", "training_data.csv")
    forecast_path = os.path.join(user_folder, "reports", "arima_forecast.csv")

    # === Load and sanitize ===
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    df["date"] = pd.to_datetime(df["date"])

    required_cols = {"store_id", "product_id", "demand_forecast", "price", "inventory"}
    if not required_cols.issubset(df.columns):
        print("❌ Missing required columns in training_data.csv")
        return

    # === Create synthetic SKU column ===
    df["sku"] = df["store_id"].astype(str) + "_" + df["product_id"].astype(str)

    results = []

    for sku in df["sku"].unique():
        sku_df = df[df["sku"] == sku].sort_values("date")
        if len(sku_df) < 8:
            print(f"⚠️ Skipping {sku}: not enough data ({len(sku_df)} rows)")
            continue

        last_date = sku_df["date"].max()
        demand_series = sku_df["demand_forecast"]
        latest_row = sku_df.iloc[-1]

        forecast = forecast_arima(demand_series, last_date)
        for date, day_ahead, pred in forecast:
            results.append({
                "sku": sku,
                "date": date,
                "day_ahead": day_ahead,
                "arima_forecast": pred,
                "price": latest_row["price"],
                "inventory": latest_row["inventory"]
            })

    if not results:
        print("❌ No forecasts generated.")
        return

    os.makedirs(os.path.join(user_folder, "reports"), exist_ok=True)
    pd.DataFrame(results).to_csv(forecast_path, index=False)
    print(f"✅ 7-day per-SKU ARIMA forecast saved to: {forecast_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        run_arima_forecast(sys.argv[1])
    else:
        run_arima_forecast(".")
