"""
src/models/xgb_train_log.py

Train XGBoost on log1p(units_sold) and produce recursive multi-day forecasts (expm1 invert).

Usage examples:

Train only:
  python -m src.models.xgb_train_log --processed_dir data/processed --model_out models/xgb_model_log.pkl --artifacts_out artifacts/artifacts_xgb_log.pkl --train --val_days 28 --test_days 28

Forecast only (model must exist):
  python -m src.models.xgb_train_log --processed_dir data/processed --model_out models/xgb_model_log.pkl --artifacts_out artifacts/artifacts_xgb_log.pkl --forecast_days 7 --output_csv forecasts_xgb_log_7d.csv

Train + forecast:
  python -m src.models.xgb_train_log --processed_dir data/processed --model_out models/xgb_model_log.pkl --artifacts_out artifacts/artifacts_xgb_log.pkl --train --forecast_days 7 --output_csv forecasts_xgb_log_7d.csv
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

def train_xgb_log(processed_dir: str,
                  model_out: str,
                  artifacts_out: str,
                  val_days: int = 28,
                  test_days: int = 28,
                  xgb_params: Dict = None,
                  num_round: int = 1000,
                  early_stopping_rounds: int = 50,
                  verbose_eval: int = 50) -> Dict:
    print("Preparing features for XGBoost (log1p target)...")
    df_full, artifacts = prepare_features_from_processed(processed_dir)
    train, val, test = split_by_date(df_full, val_days=val_days, test_days=test_days)

    FEATURES = [f for f in get_default_feature_list() if f in df_full.columns]
    print("Using features:", FEATURES)

    # drop rows missing required features (warmup)
    train = train.dropna(subset=FEATURES).copy()
    val = val.dropna(subset=FEATURES).copy()

    # log1p target
    y_train = np.log1p(train['units_sold'].values)
    y_val = np.log1p(val['units_sold'].values)

    X_train = train[FEATURES].values
    X_val = val[FEATURES].values

    if xgb_params is None:
        xgb_params = {
            'n_estimators': num_round,
            'learning_rate': 0.05,
            'max_depth': 8,
            'n_jobs': -1,
            'objective': 'reg:squarederror',
            'verbosity': 1,
        }

    model = XGBRegressor(**xgb_params)
    print("Training XGBoost (log1p target)...")
    # Try to use early stopping if supported; otherwise fall back to simple fit
    try:
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='rmse',
            early_stopping_rounds=early_stopping_rounds,
            verbose=verbose_eval
        )
    except TypeError:
        print("Warning: XGB fit did not accept eval/early_stopping kwargs on this xgboost version. Falling back to model.fit(X, y) without early stopping.")
        model.fit(X_train, y_train)

    # evaluate (invert predictions)
    preds_val_log = model.predict(X_val)
    preds_val = np.expm1(preds_val_log)
    val_mae = mean_absolute_error(val['units_sold'].values, preds_val)
    print("Validation MAE (XGB log1p -> expm1):", val_mae)

    # save model & artifacts (include features list)
    os.makedirs(os.path.dirname(model_out) or '.', exist_ok=True)
    os.makedirs(os.path.dirname(artifacts_out) or '.', exist_ok=True)
    joblib.dump(model, model_out)
    artifacts['features'] = FEATURES
    # ensure encoders exist in artifacts if present from prepare_features...
    joblib.dump(artifacts, artifacts_out)
    print(f"Saved XGBoost model to {model_out}")
    print(f"Saved artifacts to {artifacts_out}")

    return {'model_path': model_out, 'artifacts_path': artifacts_out, 'val_mae': float(val_mae)}

def recursive_forecast_xgb_log(model, df_hist: pd.DataFrame, sku: str, last_date: pd.Timestamp, horizon: int, features: List[str]) -> pd.DataFrame:
    """
    Recursive multi-day forecast for one SKU using XGB model trained on log1p.
    Predictions are inverted with expm1 before being appended to history.
    """
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
        # static fields (last known)
        for c in ['price','discount','competitor_price','inventory','category_enc','region_enc','weather_enc','season_enc','is_holiday']:
            if c in hist.columns:
                row[c] = hist[c].iloc[-1]
        # lags
        for lag in (1,7,14):
            ld = d - pd.Timedelta(days=lag)
            val = hist.loc[hist['date'] == ld, 'units_sold']
            row[f'lag_{lag}'] = float(val.values[0]) if len(val) else 0.0
        # rolling features
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
        # predict (model outputs log1p)
        X = pd.DataFrame([row])[features].fillna(0)
        y_log = model.predict(X.values)[0]
        yhat = float(max(0.0, np.expm1(y_log)))
        out_rows.append({'sku': sku, 'date': d, 'pred': yhat})
        # append for recursive use
        new_row = row.copy()
        new_row['units_sold'] = yhat
        new_row['date'] = d
        hist = pd.concat([hist, pd.DataFrame([new_row])], ignore_index=True)
    return pd.DataFrame(out_rows)

def main():
    parser = argparse.ArgumentParser(description="XGBoost log1p trainer + recursive forecast")
    parser.add_argument('--processed_dir', default='data/processed', help='Path to processed data')
    parser.add_argument('--model_out', default='models/xgb_model_log.pkl', help='Path to save model')
    parser.add_argument('--artifacts_out', default='artifacts/artifacts_xgb_log.pkl', help='Path to save artifacts')
    parser.add_argument('--train', action='store_true', help='Train model before forecasting')
    parser.add_argument('--forecast_days', type=int, default=7, help='Forecast horizon (days)')
    parser.add_argument('--output_csv', default='forecasts_xgb_log_7d.csv', help='Output CSV for forecasts')
    parser.add_argument('--val_days', type=int, default=28)
    parser.add_argument('--test_days', type=int, default=28)
    parser.add_argument('--num_round', type=int, default=1000)
    parser.add_argument('--early_stopping_rounds', type=int, default=50)
    parser.add_argument('--verbose', type=int, default=50)
    args = parser.parse_args()

    if args.train or not os.path.exists(args.model_out) or not os.path.exists(args.artifacts_out):
        print("Training XGBoost (log1p)...")
        train_xgb_log(
            processed_dir=args.processed_dir,
            model_out=args.model_out,
            artifacts_out=args.artifacts_out,
            val_days=args.val_days,
            test_days=args.test_days,
            num_round=args.num_round,
            early_stopping_rounds=args.early_stopping_rounds,
            verbose_eval=args.verbose
        )

    # load model & artifacts
    model = joblib.load(args.model_out)
    artifacts = joblib.load(args.artifacts_out)
    features = artifacts.get('features')
    if features is None:
        raise RuntimeError("Artifacts missing 'features' list; ensure artifacts file is correct.")

    # build full history and forecast for all SKUs
    df_full, _ = prepare_features_from_processed(args.processed_dir)
    last_date = df_full['date'].max()
    skus = df_full['sku'].unique()
    out_list = []
    print("Generating recursive forecasts for all SKUs...")
    for sku in tqdm(skus, desc="Forecasting SKUs"):
        fc = recursive_forecast_xgb_log(model, df_full, sku, last_date, args.forecast_days, features)
        out_list.append(fc)
    out_df = pd.concat(out_list, ignore_index=True)
    # ensure date column and save
    out_df['date'] = pd.to_datetime(out_df['date'])
    # ensure columns: sku, date, pred
    out_df = out_df[['sku','date','pred']].copy()
    out_df.to_csv(args.output_csv, index=False)
    print("Saved forecasts to", args.output_csv)

if __name__ == '__main__':
    main()
