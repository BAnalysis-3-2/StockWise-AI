"""
src/models/lgb_predict_log.py

Produce recursive LGB forecasts when model was trained on log1p(units_sold).
Usage:
  python -m src.models.lgb_predict_log --processed_dir data/processed \
    --model models/lgb_model_log.pkl --artifacts artifacts/artifacts_lgb_log.pkl \
    --forecast_days 7 --output_csv forecasts_lgb_log_7d.csv
"""
import os
import joblib
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm

from src.models.lgb_trainer import prepare_features_from_processed
# (we reuse feature generation from the lgb_trainer pipeline)

def recursive_forecast_lgb_log(model, df_hist: pd.DataFrame, sku: str, last_date: pd.Timestamp, horizon: int, features):
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
        # handle LightGBM object (LightGBM booster via joblib)
        try:
            # lgb.Booster via joblib -> use model.predict with num_iteration if available
            if hasattr(model, 'predict') and 'best_iteration' in dir(model):
                # some LightGBM wrappers have best_iteration attribute; try to use it
                try:
                    y_log = model.predict(X, num_iteration=model.best_iteration)[0]
                except Exception:
                    y_log = model.predict(X)[0]
            else:
                y_log = model.predict(X)[0]
        except Exception:
            # fallback
            y_log = model.predict(X)[0]
        yhat = float(max(0.0, np.expm1(y_log)))
        out_rows.append({'sku': sku, 'date': d, 'pred': yhat})
        # append forecast for next iterations
        new_row = row.copy()
        new_row['units_sold'] = yhat
        new_row['date'] = d
        hist = pd.concat([hist, pd.DataFrame([new_row])], ignore_index=True)
    return pd.DataFrame(out_rows)

def main(processed_dir, model_path, artifacts_path, forecast_days, output_csv):
    print("Loading model and artifacts...")
    model = joblib.load(model_path)
    artifacts = joblib.load(artifacts_path)
    features = artifacts.get('features')
    if features is None:
        raise RuntimeError("Artifacts missing 'features' list.")

    print("Preparing full history...")
    df_full, _ = prepare_features_from_processed(processed_dir)
    last_date = df_full['date'].max()
    skus = df_full['sku'].unique()

    out_list = []
    print("Generating LGB(log1p) recursive forecasts for all SKUs...")
    for sku in tqdm(skus, desc="Forecasting SKUs"):
        fc = recursive_forecast_lgb_log(model, df_full, sku, last_date, forecast_days, features)
        out_list.append(fc)
    out_df = pd.concat(out_list, ignore_index=True)
    out_df['date'] = pd.to_datetime(out_df['date'])
    out_df = out_df[['sku','date','pred']].copy()
    out_df.to_csv(output_csv, index=False)
    print("Saved LGB forecasts to", output_csv)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--processed_dir', default='data/processed')
    parser.add_argument('--model', dest='model_path', default='models/lgb_model_log.pkl')
    parser.add_argument('--artifacts', dest='artifacts_path', default='artifacts/artifacts_lgb_log.pkl')
    parser.add_argument('--forecast_days', type=int, default=7)
    parser.add_argument('--output_csv', default='forecasts_lgb_log_7d.csv')
    args = parser.parse_args()
    main(args.processed_dir, args.model_path, args.artifacts_path, args.forecast_days, args.output_csv)
