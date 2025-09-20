"""
src/models/xgb_trainer.py

Train and predict XGBoost (sklearn API) using the same feature pipeline as LightGBM trainer.
Generates a forecast CSV file containing predicted demand (units sold) for each SKU over the next 7 days.
"""
import os
import joblib
import argparse
from typing import Dict, List
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

from src.models.lgb_trainer import prepare_features_from_processed, split_by_date
from src.data.feature_engineering import get_default_feature_list


def train_xgb_from_processed(processed_dir: str,
                             model_out: str,
                             artifacts_out: str,
                             val_days: int = 28,
                             test_days: int = 28,
                             xgb_params: Dict = None,
                             num_round: int = 1000,
                             early_stopping_rounds: int = 50,
                             verbose_eval: int = 50) -> Dict:
    print("Preparing features from processed CSVs (XGBoost)...")
    df_full, artifacts = prepare_features_from_processed(processed_dir)
    train, val, test = split_by_date(df_full, val_days=val_days, test_days=test_days)

    # determine features
    default_feats = get_default_feature_list()
    FEATURES = [f for f in default_feats if f in df_full.columns]
    print("Using features:", FEATURES)

    train = train.dropna(subset=FEATURES)
    val = val.dropna(subset=FEATURES)

    X_train = train[FEATURES].values
    y_train = train['units_sold'].values
    X_val = val[FEATURES].values
    y_val = val['units_sold'].values

    if xgb_params is None:
        xgb_params = {
            'n_estimators': num_round,
            'learning_rate': 0.05,
            'max_depth': 8,
            'n_jobs': -1,
            'objective': 'reg:squarederror',
            'verbosity': 1,
            'eval_metric': 'mae'
        }

    model = XGBRegressor(**xgb_params)
    print("Starting XGBoost training...")

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)]
    )

    # evaluate on validation set
    preds_val = model.predict(X_val)
    val_mae = mean_absolute_error(y_val, preds_val)
    print("Validation MAE (XGB):", val_mae)

    # save model and artifacts
    os.makedirs(os.path.dirname(model_out) or '.', exist_ok=True)
    os.makedirs(os.path.dirname(artifacts_out) or '.', exist_ok=True)
    joblib.dump(model, model_out)
    artifacts['features'] = FEATURES
    joblib.dump(artifacts, artifacts_out)
    print(f"Saved XGBoost model to {model_out}")
    print(f"Saved artifacts to {artifacts_out}")

    return {'model_path': model_out, 'artifacts_path': artifacts_out, 'val_mae': float(val_mae)}



def recursive_forecast_for_sku_xgb(model, df_hist: pd.DataFrame, sku: str, last_date: pd.Timestamp, horizon: int, features: List[str]) -> pd.DataFrame:
    hist = df_hist[df_hist['sku'] == sku].sort_values('date').copy()
    out_rows = []
    for h in range(horizon):
        d = last_date + pd.Timedelta(days=h+1)
        row = {'sku': sku, 'date': d}
        # date features
        row['dow'] = d.dayofweek
        row['month'] = d.month
        row['day'] = d.day
        row['week'] = d.isocalendar()[1]
        # static last known values
        for c in ['price','discount','competitor_price','inventory','category_enc','region_enc','weather_enc','season_enc','is_holiday']:
            if c in hist.columns:
                row[c] = hist[c].iloc[-1]
        # lag features
        for lag in (1,7,14):
            ld = d - pd.Timedelta(days=lag)
            val = hist.loc[hist['date'] == ld, 'units_sold']
            row[f'lag_{lag}'] = float(val.values[0]) if len(val) else 0.0
        # rolling window features
        for w in (7,28):
            window_start = d - pd.Timedelta(days=w)
            vals = hist[(hist['date'] >= window_start) & (hist['date'] < d)]['units_sold']
            row[f'roll_mean_{w}'] = float(vals.mean()) if len(vals) else 0.0
            row[f'roll_std_{w}'] = float(vals.std()) if len(vals) else 0.0
        # days since last sale
        if (hist['units_sold'] > 0).any():
            row['days_since_sale'] = (d - hist[hist['units_sold'] > 0]['date'].max()).days
        else:
            row['days_since_sale'] = 999
        # make prediction
        X = pd.DataFrame([row])[features].fillna(0)
        yhat = model.predict(X.values)[0]
        yhat = float(max(0.0, yhat))
        out_rows.append({'sku': sku, 'date': d, 'predicted_units_sold': yhat})
        # append forecast for next iteration
        new_row = dict(row)
        new_row['units_sold'] = yhat
        hist = pd.concat([hist, pd.DataFrame([new_row])], ignore_index=True)
    return pd.DataFrame(out_rows)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate 7-day XGBoost forecasts for all SKUs.")
    parser.add_argument("--processed_dir", type=str, required=True, help="Path to processed data directory")
    parser.add_argument("--model_out", type=str, default="models/xgb_model.pkl", help="Path to save/load trained model")
    parser.add_argument("--artifacts_out", type=str, default="artifacts/artifacts_xgb.pkl", help="Path to save/load artifacts")
    parser.add_argument("--forecast_days", type=int, default=7, help="Forecast horizon in days")
    parser.add_argument("--output_csv", type=str, default="forecasts_xgb_7d.csv", help="Output CSV for forecasts")
    parser.add_argument("--train", action="store_true", help="Train model before forecasting (default: only forecast)")
    parser.add_argument("--val_days", type=int, default=28, help="Validation days")
    parser.add_argument("--test_days", type=int, default=28, help="Test days")
    args = parser.parse_args()

    # Train if requested or if model file does not exist
    if args.train or not os.path.exists(args.model_out) or not os.path.exists(args.artifacts_out):
        print("Training XGBoost model...")
        train_xgb_from_processed(
            processed_dir=args.processed_dir,
            model_out=args.model_out,
            artifacts_out=args.artifacts_out,
            val_days=args.val_days,
            test_days=args.test_days
        )

    # Load model and artifacts
    model = joblib.load(args.model_out)
    artifacts = joblib.load(args.artifacts_out)
    features = artifacts['features']

    # Load processed data
    df_full, _ = prepare_features_from_processed(args.processed_dir)
    last_date = df_full['date'].max()
    skus = df_full['sku'].unique()

    # Forecast for all SKUs
    forecasts = []
    for sku in tqdm(skus, desc="Forecasting SKUs"):
        fc = recursive_forecast_for_sku_xgb(model, df_full, sku, last_date, args.forecast_days, features)
        forecasts.append(fc)
    forecasts_df = pd.concat(forecasts, ignore_index=True)
    # Format columns as required
    forecasts_df.rename(columns={'sku': 'SKU', 'date': 'Date', 'predicted_units_sold': 'PredictedUnitsSold'}, inplace=True)
    forecasts_df['Date'] = forecasts_df['Date'].dt.strftime('%Y-%m-%d')
    forecasts_df.to_csv(args.output_csv, index=False)
    print(f"Saved forecasts to {args.output_csv}")