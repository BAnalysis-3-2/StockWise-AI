import pandas as pd
import xgboost as xgb
import joblib
from tqdm import tqdm
import os
import sys

def forecast_xgb(user_path):
    train_path = os.path.join(user_path, "training_features.csv")
    output_path = os.path.join(user_path, "forecasts_xgb_7d.csv")
    model_dir = os.path.join(user_path, "models")
    os.makedirs(model_dir, exist_ok=True)

    print(f"🚀 Training XGBoost per-SKU for: {user_path}")

    if not os.path.exists(train_path):
        print("❌ Feature data not found. Please run build_features first.")
        return

    df = pd.read_csv(train_path)
    df['date'] = pd.to_datetime(df['date'])

    forecast_rows = []

    for sku in tqdm(df['sku'].unique()):
        sku_df = df[df['sku'] == sku].sort_values('date')

        if len(sku_df) < 90:
            continue

        train_df = sku_df.iloc[:-7]
        test_df = sku_df.iloc[-7:]

        features = [
            'lag_1', 'lag_7', 'lag_14',
            'rolling_mean_3', 'rolling_mean_7', 'rolling_std_7',
            'day_of_week', 'month',
            'category', 'region', 'weather condition', 'seasonality',
            'price', 'price_change_7d',
            'discount', 'discount_flag',
            'inventory_level', 'inventory_ratio',
            'holiday_flag'
        ]

        X_train = train_df[features]
        y_train = train_df['units sold']
        X_test = test_df[features]

        model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
        model.fit(X_train, y_train)

        joblib.dump(model, os.path.join(model_dir, f"xgb_model_{sku}.pkl"))

        preds = model.predict(X_test)
        test_df = test_df.copy()
        test_df['forecast'] = preds.clip(0)
        forecast_rows.append(test_df[['sku', 'date', 'forecast']])

    if forecast_rows:
        result = pd.concat(forecast_rows)
        result.to_csv(output_path, index=False)
        print(f"✅ Forecasts saved: {output_path}")
    else:
        print("⚠️ No SKUs met the minimum row threshold.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❌ Please provide user folder path as argument.")
    else:
        forecast_xgb(sys.argv[1])
