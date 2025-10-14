import pandas as pd
import os
import sys

def combine_forecasts(user_path):
    print(f"🔧 Combining forecasts for: {user_path}")

    lgb_path = os.path.join(user_path, "forecasts_lgb_7d.csv")
    xgb_path = os.path.join(user_path, "forecasts_xgb_7d.csv")
    arima_path = os.path.join(user_path, "forecasts_arima_7d.csv")
    output_path = os.path.join(user_path, "ensemble_v2_3models_with_xgb.csv")

    # Load forecasts
    try:
        lgb = pd.read_csv(lgb_path)
        xgb = pd.read_csv(xgb_path)
        arima = pd.read_csv(arima_path)
    except Exception as e:
        print(f"❌ Failed to load forecast files: {e}")
        return

    # Clean up formats
    for df in [lgb, xgb, arima]:
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df["sku"] = df["sku"].astype(str).str.strip()

    # Merge forecasts
    merged = lgb.merge(xgb, on=["sku", "date"], suffixes=("_lgb", "_xgb"))
    merged = merged.merge(arima, on=["sku", "date"])
    merged.rename(columns={"forecast": "forecast_arima"}, inplace=True)

    # Apply fixed weights
    merged["ensemble_forecast"] = (
        0.4 * merged["forecast_lgb"] +
        0.4 * merged["forecast_xgb"] +
        0.2 * merged["forecast_arima"]
    )

    # Save ensemble output
    merged[["sku", "date", "ensemble_forecast"]].to_csv(output_path, index=False)
    print(f"✅ Ensemble forecast saved: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❌ Please provide user folder path as argument.")
    else:
        combine_forecasts(sys.argv[1])
