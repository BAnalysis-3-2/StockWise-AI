import pandas as pd
import os
import sys

def evaluate_and_fallback(user_path, threshold=50):
    print(f"🔧 Evaluating forecasts and applying fallback for: {user_path}")

    # Define paths
    ensemble_path = os.path.join(user_path, "ensemble_v2_3models_with_xgb.csv")
    actuals_path = os.path.join(user_path, "training_data.csv")
    arima_path = os.path.join(user_path, "forecasts_arima_7d.csv")
    mae_path = os.path.join(user_path, "mae_per_sku.csv")
    final_path = os.path.join(user_path, "final_forecast_with_fallback.csv")

    # Load data
    try:
        ensemble = pd.read_csv(ensemble_path)
        actuals = pd.read_csv(actuals_path)
        arima = pd.read_csv(arima_path)
    except Exception as e:
        print(f"❌ Failed to load required files: {e}")
        return

    # Clean formats
    for df in [ensemble, actuals, arima]:
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df["sku"] = df["sku"].astype(str).str.strip()

    # Merge actuals
    merged = ensemble.merge(actuals[["sku", "date", "units sold"]], on=["sku", "date"])
    merged["error"] = abs(merged["ensemble_forecast"] - merged["units sold"])

    # Compute MAE per SKU
    mae_per_sku = merged.groupby("sku")["error"].mean().reset_index()
    mae_per_sku.columns = ["sku", "mae"]
    mae_per_sku.to_csv(mae_path, index=False)
    print(f"✅ Saved MAE per SKU: {mae_path}")

    # Identify weak SKUs
    weak_skus = mae_per_sku[mae_per_sku["mae"] > threshold]["sku"].values

    # Merge ARIMA forecasts
    merged = merged.merge(arima, on=["sku", "date"])
    merged.rename(columns={"forecast": "forecast_arima"}, inplace=True)

    # Apply fallback logic
    merged["final_forecast"] = merged.apply(
        lambda row: row["forecast_arima"] if row["sku"] in weak_skus else row["ensemble_forecast"],
        axis=1
    )

    # Save final forecast
    merged[["sku", "date", "final_forecast"]].to_csv(final_path, index=False)
    print(f"✅ Saved final forecast with fallback: {final_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❌ Please provide user folder path as argument.")
    else:
        evaluate_and_fallback(sys.argv[1])
