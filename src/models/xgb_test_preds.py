# src/models/xgb_test_preds.py
import joblib, os, pandas as pd, numpy as np
from src.models.lgb_trainer import prepare_features_from_processed, split_by_date

def main(processed_dir='data/processed', model_path='models/xgb_model_log.pkl', artifacts_path='artifacts/artifacts_xgb_log.pkl', out='reports/xgb_test_preds.csv'):
    print("Loading model and artifacts...")
    m = joblib.load(model_path)
    art = joblib.load(artifacts_path)
    feats = art['features']
    print("Preparing features & test slice...")
    df_full, _ = prepare_features_from_processed(processed_dir)
    train,val,test = split_by_date(df_full)
    test = test.copy()
    # drop rows without features
    test = test.dropna(subset=feats)
    print("Predicting on test slice for", len(test), "rows...")
    X = test[feats]
    # model outputs log1p -> invert
    preds_log = m.predict(X.values)
    preds = np.expm1(preds_log)
    out_df = test[['sku','date']].copy()
    out_df['pred'] = preds
    out_dir = os.path.dirname(out) or '.'
    os.makedirs(out_dir, exist_ok=True)
    out_df.to_csv(out, index=False)
    print("Saved XGB test preds to", out)

if __name__ == '__main__':
    main()
