"""
src/models/evaluate_ensemble.py

Evaluate ensemble forecasts vs LightGBM and ARIMA on the test slice.

Usage (from project root):
  python -m src.models.evaluate_ensemble \
    --ensemble ensemble_forecasts_7d.csv \
    --lgb reports/test_with_preds.csv \
    --arima forecasts_arima_7d.csv \
    --processed_dir data/processed \
    --out_dir reports

Outputs:
- reports/ensemble_test_with_preds.csv (merged test rows with ensemble, lgb, arima preds)
- reports/ensemble_per_sku.csv (per-SKU aggregated errors)
- reports/ensemble_overall_metrics.csv (overall MAE/RMSE/SMAPE)
"""
import os
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def smape(y_true, y_pred):
    denom = (np.abs(y_true) + np.abs(y_pred))
    denom[denom == 0] = 1e-9
    return 100.0 * np.mean(2.0 * np.abs(y_pred - y_true) / denom)

def load_ensemble(path):
    df = pd.read_csv(path, parse_dates=['date'])
    # allowed ensemble file may have columns sku,date,pred or sku,date,ensemble_pred
    if 'pred' in df.columns:
        df = df[['sku','date','pred']].rename(columns={'pred':'ensemble_pred'})
    elif 'ensemble_pred' in df.columns:
        df = df[['sku','date','ensemble_pred']]
    else:
        raise ValueError("Ensemble CSV must contain 'pred' or 'ensemble_pred' column")
    return df

def evaluate(ensemble_path, lgb_test_path, arima_path, processed_dir, out_dir='reports'):
    os.makedirs(out_dir, exist_ok=True)
    # load ensemble, lgb test preds (reports/test_with_preds.csv), arima forecasts
    ens = load_ensemble(ensemble_path)
    lgb = pd.read_csv(lgb_test_path, parse_dates=['date'])
    # lgb file should contain 'sku','date','pred' (pred = lgb prediction) and 'units_sold' actuals
    if 'pred' in lgb.columns:
        lgb = lgb[['sku','date','pred','units_sold']].rename(columns={'pred':'lgb_pred','units_sold':'actual'})
    elif 'pred' not in lgb.columns and 'units_sold' in lgb.columns and 'pred' not in lgb.columns:
        raise ValueError("LGB test file must contain columns 'pred' and 'units_sold'")

    arima = pd.read_csv(arima_path, parse_dates=['date'])
    if 'pred' in arima.columns:
        arima = arima[['sku','date','pred']].rename(columns={'pred':'arima_pred'})
    elif 'arima_pred' in arima.columns:
        arima = arima[['sku','date','arima_pred']]
    else:
        raise ValueError("ARIMA forecast file must contain column 'pred'")

    # Merge ensemble with LGB test (which contains actuals)
    merged = lgb.merge(ens, on=['sku','date'], how='left')
    merged = merged.merge(arima, on=['sku','date'], how='left')

    # Fill missing predictions with 0
    merged['ensemble_pred'] = merged['ensemble_pred'].fillna(0.0)
    merged['arima_pred'] = merged['arima_pred'].fillna(0.0)
    merged['lgb_pred'] = merged['lgb_pred'].fillna(0.0)

    # Compute overall metrics
    actual = merged['actual'].values
    ens_pred = merged['ensemble_pred'].values
    lgb_pred = merged['lgb_pred'].values
    arima_pred = merged['arima_pred'].values

    ens_mae = mean_absolute_error(actual, ens_pred)
    ens_rmse = float(np.sqrt(mean_squared_error(actual, ens_pred)))
    ens_smape = smape(actual, ens_pred)

    lgb_mae = mean_absolute_error(actual, lgb_pred)
    lgb_rmse = float(np.sqrt(mean_squared_error(actual, lgb_pred)))
    lgb_smape = smape(actual, lgb_pred)

    arima_mae = mean_absolute_error(actual, arima_pred)
    arima_rmse = float(np.sqrt(mean_squared_error(actual, arima_pred)))
    arima_smape = smape(actual, arima_pred)

    print("Overall metrics (test slice):")
    print(f"Ensemble MAE: {ens_mae:.4f}  RMSE: {ens_rmse:.4f}  SMAPE: {ens_smape:.4f}%")
    print(f"LGB      MAE: {lgb_mae:.4f}  RMSE: {lgb_rmse:.4f}  SMAPE: {lgb_smape:.4f}%")
    print(f"ARIMA    MAE: {arima_mae:.4f}  RMSE: {arima_rmse:.4f}  SMAPE: {arima_smape:.4f}%")

    # Save merged test preds
    merged.to_csv(os.path.join(out_dir, 'ensemble_test_with_preds.csv'), index=False)
    print("Saved reports/ensemble_test_with_preds.csv")

    # Per SKU aggregates
    merged['abs_ens'] = np.abs(merged['actual'] - merged['ensemble_pred'])
    merged['abs_lgb'] = np.abs(merged['actual'] - merged['lgb_pred'])
    merged['abs_arima'] = np.abs(merged['actual'] - merged['arima_pred'])

    sku_agg = merged.groupby('sku').agg(
        n=('actual','size'),
        actual_sum=('actual','sum'),
        ens_sum=('ensemble_pred','sum'),
        lgb_sum=('lgb_pred','sum'),
        arima_sum=('arima_pred','sum'),
        ens_mae=('abs_ens','mean'),
        lgb_mae=('abs_lgb','mean'),
        arima_mae=('abs_arima','mean')
    ).reset_index().sort_values('actual_sum', ascending=False)

    sku_agg.to_csv(os.path.join(out_dir, 'ensemble_per_sku.csv'), index=False)
    print("Saved reports/ensemble_per_sku.csv")

    # Save overall metrics
    overall = {
        'ensemble_mae': ens_mae, 'ensemble_rmse': ens_rmse, 'ensemble_smape': ens_smape,
        'lgb_mae': lgb_mae, 'lgb_rmse': lgb_rmse, 'lgb_smape': lgb_smape,
        'arima_mae': arima_mae, 'arima_rmse': arima_rmse, 'arima_smape': arima_smape
    }
    pd.Series(overall).to_csv(os.path.join(out_dir,'ensemble_overall_metrics.csv'))
    print("Saved reports/ensemble_overall_metrics.csv")

    return overall

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--ensemble', required=True)
    p.add_argument('--lgb', required=True, help='LGB test preds file (reports/test_with_preds.csv)')
    p.add_argument('--arima', required=True)
    p.add_argument('--processed_dir', required=True)
    p.add_argument('--out_dir', default='reports')
    args = p.parse_args()
    evaluate(args.ensemble, args.lgb, args.arima, args.processed_dir, out_dir=args.out_dir)
