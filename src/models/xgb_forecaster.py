import os
import pandas as pd
import xgboost as xgb

def run_xgb_forecast(user_folder):
    # === Paths ===
    features_path = os.path.join(user_folder, "data", "processed", "training_features.csv")
    model_path = "models/xgb/xgb_model.json"  # Central pre-trained model
    forecast_path = os.path.join(user_folder, "reports", "xgb_forecast.csv")

    # === Load features ===
    df = pd.read_csv(features_path)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # === Select features for prediction ===
    X = df[[
        "price", "discount", "inventory", "holiday",
        "lag_1", "lag_7", "rolling_mean_7", "rolling_std_7",
        "day_of_week", "month", "year"
    ]]

    # === Load pre-trained model ===
    model = xgb.XGBRegressor()
    model.load_model(model_path)

    # === Predict and save ===
    forecast = model.predict(X)
    pd.DataFrame({"xgb_forecast": forecast}).to_csv(forecast_path, index=False)

    print(f"✅ XGB forecast saved to: {forecast_path}")
