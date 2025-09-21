"""
Evaluate trained LightGBM model using the same feature pipeline as training.
This corrected evaluator loads the saved artifacts file to get the feature list.
"""
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.models.lgb_trainer import prepare_features_from_processed, split_by_date

def smape(y_true, y_pred):
    denom = (np.abs(y_true) + np.abs(y_pred))
    denom[denom == 0] = 1e-9
    return 100.0 * np.mean(2.0 * np.abs(y_pred - y_true) / denom)

def wmape(y_true, y_pred, weights=None):
    if weights is None:
        denom = np.sum(np.abs(y_true))
        if denom == 0:
            return np.nan
        return 100.0 * np.sum(np.abs(y_pred - y_true)) / denom * 100.0
    denom = np.sum(weights)
    if denom == 0:
        return np.nan
    return 100.0 * np.sum(weights * np.abs(y_pred - y_true)) / denom

def evaluate(model_path, artifacts_path, processed_dir, out_dir='reports', val_days=28, test_days=28):
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not os.path.exists(artifacts_path):
        raise FileNotFoundError(f"Artifacts not found: {artifacts_path}")

    # Load saved artifacts (contains 'features' list and encoders)
    saved_artifacts = joblib.load(artifacts_path)
    features = saved_artifacts.get('features')
    if features is None:
        raise ValueError("Saved artifacts missing 'features' list. Ensure you passed the correct artifacts file.")

    # Recreate full dataframe with engineered features (same as training)
    print("Preparing features from processed CSVs (same pipeline as training)...")
    df_full, _ = prepare_features_from_processed(processed_dir)
    print("Splitting by date...")
    train, val, test = split_by_date(df_full, val_days=val_days, test_days=test_days)

    # drop rows in test without full feature set (warmup NaNs)
    test = test.dropna(subset=features)
    if test.empty:
        raise RuntimeError("Test set is empty after dropping rows without required features. Check processed data / lookback windows.")

    # load model
    model = joblib.load(model_path)

    X_test = test[features]
    y_test = test['units_sold'].values
    preds = model.predict(X_test, num_iteration=model.best_iteration)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    s_map = smape(y_test, preds)
    w_map = wmape(y_test, preds)

    print("Overall metrics on test set:")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"SMAPE: {s_map:.4f}%")
    print(f"WMAPE: {w_map:.4f}%")

    # attach preds and save test slice with preds
    test_out = test.copy()
    test_out['pred'] = preds
    test_out.to_csv(os.path.join(out_dir, 'test_with_preds.csv'), index=False)
    print(f"Saved test_with_preds.csv ({len(test_out)} rows)")

    # per-SKU analysis (top 20 by actual volume)
    sku_agg = test_out.groupby('sku').agg(
        n_rows=('units_sold', 'size'),
        actual_sum=('units_sold', 'sum'),
        pred_sum=('pred', 'sum'),
        mae=('units_sold', lambda x: np.mean(np.abs(x - test_out.loc[x.index, 'pred'])))
    )
    sku_agg = sku_agg.sort_values('actual_sum', ascending=False)
    sku_agg.head(20).to_csv(os.path.join(out_dir, 'top20_sku_errors.csv'))
    print("Saved top20_sku_errors.csv with per-SKU aggregates.")

    # feature importance
    try:
        fmap = model.feature_importance(importance_type='gain')
        fnames = features
        fi = pd.DataFrame({'feature': fnames, 'gain': fmap})
        fi = fi.sort_values('gain', ascending=False)
        fi.to_csv(os.path.join(out_dir, 'feature_importance.csv'), index=False)
        print("Saved feature_importance.csv")
        print(fi.head(20).to_string(index=False))
    except Exception as e:
        print("Could not extract feature importance:", e)

    # save overall metrics
    metrics = {'MAE': mae, 'RMSE': rmse, 'SMAPE': s_map, 'WMAPE': w_map}
    pd.Series(metrics).to_csv(os.path.join(out_dir, 'overall_metrics.csv'))
    print("Saved overall_metrics.csv")
    return metrics

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='models/lgb_model.pkl')
    p.add_argument('--artifacts', default='artifacts/artifacts.pkl')
    p.add_argument('--processed_dir', default='data/processed')
    p.add_argument('--out_dir', default='reports')
    p.add_argument('--val_days', type=int, default=28)
    p.add_argument('--test_days', type=int, default=28)
    args = p.parse_args()
    evaluate(args.model, args.artifacts, args.processed_dir, args.out_dir, val_days=args.val_days, test_days=args.test_days)
