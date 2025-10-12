import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from tqdm import tqdm
import warnings
import joblib
import os
import sys

warnings.filterwarnings("ignore")

def forecast_arima(user_path, min_rows=30):
    print(f"📈 Training ARIMA per SKU for: {user_path}")

    train_path = os.path.join(user_path, "training_data.csv")
    output_path = os.path.join(user_path, "forecasts_arima_7d.csv")
    model_dir = os.path.join(user_path, "models", "arima")
    os.makedirs(model_dir, exist_ok=True)

    if not os.path.exists(train_path):
        print("❌ Training data not found. Please run load_and_clean first.")
        return

    df = pd.read_csv(train_path)
    df["date"] = pd.to_datetime(df["date"])
    df["sku"] = df["sku"].astype(str).str.strip()

    forecast_rows = []

    # 🔧 Match forecast horizon to LGB/XGB
    forecast_dates = pd.date_range(start="2023-12-26", periods=7)

    for sku in tqdm(df["sku"].unique()):
        sku_df = df[df["sku"] == sku].sort_values("date")

        if len(sku_df) < min_rows:
            continue

        ts = sku_df["units sold"]

        try:
            model = ARIMA(ts, order=(1, 1, 1))
            fitted = model.fit()

            joblib.dump(fitted, os.path.join(model_dir, f"arima_model_{sku}.pkl"))

            forecast = fitted.forecast(steps=7)

            forecast_df = pd.DataFrame({
                "sku": sku,
                "date": forecast_dates,
                "forecast": forecast.clip(lower=0)
            })

            forecast_rows.append(forecast_df)

        except Exception as e:
            print(f"⚠️ ARIMA failed for {sku}: {e}")
            continue

    if forecast_rows:
        result = pd.concat(forecast_rows)
        result.to_csv(output_path, index=False)
        print(f"✅ Saved ARIMA forecasts: {output_path}")
    else:
        print("⚠️ No SKUs met the minimum row threshold.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❌ Please provide user folder path as argument.")
    else:
        forecast_arima(sys.argv[1])
