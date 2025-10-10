# src/models/evaluate_xgb_per_sku.py
import os, joblib, pandas as pd, numpy as np
from sklearn.metrics import mean_absolute_error
from src.models.lgb_trainer import prepare_features_from_processed, split_by_date

def main(processed_dir='data/processed', xgb_model='models/xgb_model.pkl', artifacts='artifacts/artifacts_xgb.pkl', out='reports/per_sku_xgb_mae.csv'):
    print("Loading XGB model and artifacts...")
    model = joblib.load(xgb_model)
    art = joblib.load(artifacts)
    feats = art['features']
    print("Preparing features (same pipeline as training)...")
    df_full, _ = prepare_features_from_processed(processed_dir)
    train, val, test = split_by_date(df_full)
    # restrict to rows where features are available
    test = test.dropna(subset=feats)
    rows = []
    print("Computing per-SKU XGB MAE on test slice...")
    for sku, g in test.groupby('sku'):
        X = g[feats].values
        y = g['units_sold'].values
        try:
            preds = model.predict(X)
            mae = float(mean_absolute_error(y, preds))
        except Exception as e:
            print(f"Warning: SKU {sku} failed: {e}")
            mae = float('nan')
        rows.append({'sku': sku, 'xgb_mae': mae})
    out_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    out_df.to_csv(out, index=False)
    print("Saved", out)

if __name__ == '__main__':
    main()


