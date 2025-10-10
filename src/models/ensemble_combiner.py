"""
src/models/ensemble_combiner.py

Combine forecasts from multiple models (LightGBM and ARIMA) using per-SKU dynamic weights
derived from model performance (MAE). Falls back to global weights if per-SKU metrics missing.

Usage (from project root):
  python -m src.models.ensemble_combiner \
    --lgb forecasts_7d.csv \
    --arima forecasts_arima_7d.csv \
    --per_sku reports/arima_vs_lgb_per_sku.csv \
    --summary reports/arima_vs_lgb_summary.csv \
    --out ensemble_forecasts_7d.csv

Notes:
- Input forecast CSVs must contain columns: sku, date, pred
- per_sku CSV should contain columns: sku, arima_mae, lgb_mae (produced earlier)
- summary CSV should contain aggregated MAE for fallback (e.g., arima_vs_lgb_summary.csv)
"""
import os
import argparse
import pandas as pd
import numpy as np

EPS = 1e-6  # smoothing to avoid divide-by-zero

def load_forecasts(path):
    df = pd.read_csv(path, parse_dates=['date'])
    # normalize column names
    df = df.rename(columns={c: c.strip() for c in df.columns})
    # ensure required columns
    if not set(['sku','date','pred']).issubset(set(df.columns)):
        raise ValueError(f"Forecast file {path} must contain columns 'sku','date','pred'")
    return df[['sku','date','pred']].copy()

def load_per_sku_metrics(path):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    # expected columns: sku, arima_mae, lgb_mae
    expected = set(['sku','arima_mae','lgb_mae'])
    if not expected.issubset(set(df.columns)):
        # attempt to infer names
        # look for columns containing 'arima' and 'lgb'
        cols = df.columns.tolist()
        return df
    return df

def load_summary(path):
    if not os.path.exists(path):
        return None
    s = pd.read_csv(path, index_col=0, header=None).iloc[:,0].to_dict()
    # returns a mapping of keys->values (strings from file)
    return s

def compute_weights_per_sku(per_sku_df, summary_dict=None, eps=EPS):
    """
    Returns DataFrame with columns sku, w_lgb, w_arima
    Uses inverse MAE weighting: inv = 1/(mae+eps); weight = inv / (inv_lgb + inv_arima)
    If per_sku_df is None, fallback to global weights from summary_dict.
    """
    if per_sku_df is not None and 'sku' in per_sku_df.columns and 'arima_mae' in per_sku_df.columns and 'lgb_mae' in per_sku_df.columns:
        df = per_sku_df.copy()
        df['arima_mae'] = df['arima_mae'].astype(float).fillna(np.inf)
        df['lgb_mae'] = df['lgb_mae'].astype(float).fillna(np.inf)
        df['inv_arima'] = 1.0 / (df['arima_mae'] + eps)
        df['inv_lgb'] = 1.0 / (df['lgb_mae'] + eps)
        df['sum_inv'] = df['inv_arima'] + df['inv_lgb']
        df['w_arima'] = df['inv_arima'] / df['sum_inv']
        df['w_lgb'] = df['inv_lgb'] / df['sum_inv']
        return df[['sku','w_lgb','w_arima']]
    # fallback: use summary_dict to create global weights
    if summary_dict is not None:
        # look for keys containing 'arima_mae' and 'lgb_mae' or similar
        arima_mae = None
        lgb_mae = None
        for k,v in summary_dict.items():
            kn = str(k).lower()
            if 'arima_mae' in kn or 'arima' in kn and 'mae' in kn:
                arima_mae = float(v)
            if 'lgb_mae' in kn or ('lgb' in kn and 'mae' in kn) or ('lgb' in kn and 'mae' not in kn):
                lgb_mae = float(v)
        # if not found, try positional keys
        if arima_mae is None or lgb_mae is None:
            # parse common keys
            arima_mae = float(summary_dict.get('arima_mae', summary_dict.get('arima_mae', np.nan)))
            lgb_mae = float(summary_dict.get('lgb_mae', summary_dict.get('lgb_mae', np.nan)))
        if np.isnan(arima_mae) or np.isnan(lgb_mae):
            # fallback equal weights
            w_lgb = 0.5; w_arima = 0.5
        else:
            inv_arima = 1.0/(arima_mae + eps)
            inv_lgb = 1.0/(lgb_mae + eps)
            s = inv_arima + inv_lgb
            w_arima = inv_arima/s
            w_lgb = inv_lgb/s
        # return DataFrame with a single row and marker sku='__GLOBAL__'
        return pd.DataFrame([{'sku':'__GLOBAL__','w_lgb':w_lgb,'w_arima':w_arima}])
    # final fallback equal weights
    return pd.DataFrame([{'sku':'__GLOBAL__','w_lgb':0.5,'w_arima':0.5}])

def combine_forecasts(lgb_path, arima_path, per_sku_path=None, summary_path=None, out_path='ensemble_forecasts.csv', eps=EPS):
    lgb = load_forecasts(lgb_path)
    arima = load_forecasts(arima_path)
    # ensure date dtype matches
    lgb['date'] = pd.to_datetime(lgb['date']).dt.normalize()
    arima['date'] = pd.to_datetime(arima['date']).dt.normalize()

    # merge LGB and ARIMA forecasts on sku+date
    merged = pd.merge(lgb, arima, on=['sku','date'], how='outer', suffixes=('_lgb','_arima'))

    # load per-sku metrics and summary
    per_sku = load_per_sku_metrics(per_sku_path) if per_sku_path else None
    summary = load_summary(summary_path) if summary_path else None

    weights_df = compute_weights_per_sku(per_sku, summary_dict=summary, eps=eps)
    # build a mapping sku -> weights
    weight_map = {}
    if 'sku' in weights_df.columns and 'w_lgb' in weights_df.columns and 'w_arima' in weights_df.columns:
        for _, row in weights_df.iterrows():
            weight_map[row['sku']] = (float(row['w_lgb']), float(row['w_arima']))

    # apply weights per row
    def get_weights(sku):
        if sku in weight_map:
            return weight_map[sku]
        # fallback to global if present
        if '__GLOBAL__' in weight_map:
            return weight_map['__GLOBAL__']
        # else equal
        return (0.5, 0.5)

    w_lgbs = []
    w_arimas = []
    for sku in merged['sku'].values:
        wg = get_weights(sku)
        w_lgbs.append(wg[0])
        w_arimas.append(wg[1])
    merged['w_lgb'] = w_lgbs
    merged['w_arima'] = w_arimas

    # fill missing predictions with 0 (or consider better fallback)
    merged['pred_lgb'] = merged['pred_lgb'].fillna(0.0)
    merged['pred_arima'] = merged['pred_arima'].fillna(0.0)

    # compute ensemble
    merged['ensemble_pred'] = merged['w_lgb'] * merged['pred_lgb'] + merged['w_arima'] * merged['pred_arima']

    # save
    out = merged[['sku','date','ensemble_pred','pred_lgb','pred_arima','w_lgb','w_arima']].copy()
    out = out.rename(columns={'ensemble_pred':'pred'})
    out.to_csv(out_path, index=False)
    print(f"Saved ensemble forecasts to {out_path}")
    return out_path

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--lgb', required=True, help='Path to LightGBM forecasts CSV (sku,date,pred)')
    p.add_argument('--arima', required=True, help='Path to ARIMA forecasts CSV (sku,date,pred)')
    p.add_argument('--per_sku', default='reports/arima_vs_lgb_per_sku.csv', help='Per-SKU metrics CSV from ARIMA vs LGB comparison')
    p.add_argument('--summary', default='reports/arima_vs_lgb_summary.csv', help='Summary CSV (global MAEs) fallback')
    p.add_argument('--out', default='ensemble_forecasts_7d.csv', help='Path to save ensemble forecasts')
    args = p.parse_args()
    combine_forecasts(args.lgb, args.arima, per_sku_path=args.per_sku, summary_path=args.summary, out_path=args.out)
