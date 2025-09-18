# src/models/lgb_train_log.py
import os, joblib
import pandas as pd, numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
from src.models.lgb_trainer import prepare_features_from_processed, split_by_date, get_default_feature_list
from tqdm import tqdm

def train_lgb_log(processed_dir='data/processed', model_out='models/lgb_model_log.pkl', artifacts_out='artifacts/artifacts_lgb_log.pkl', val_days=28, test_days=28):
    print("Preparing features...")
    df_full, artifacts = prepare_features_from_processed(processed_dir)
    train, val, test = split_by_date(df_full, val_days=val_days, test_days=test_days)
    FEATURES = [f for f in get_default_feature_list() if f in df_full.columns]
    print("Features:", FEATURES)
    # log1p target
    train = train.dropna(subset=FEATURES).copy()
    val = val.dropna(subset=FEATURES).copy()
    y_train = np.log1p(train['units_sold'].values)
    y_val = np.log1p(val['units_sold'].values)
    dtrain = lgb.Dataset(train[FEATURES], label=y_train)
    dval = lgb.Dataset(val[FEATURES], label=y_val, reference=dtrain)
    params = {'objective':'regression','metric':'rmse','learning_rate':0.05,'num_leaves':64,'verbosity':-1}
    print("Training LGB on log1p(target)...")
    model = lgb.train(params, dtrain, valid_sets=[dval], num_boost_round=2000, callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(100)])
    # Save
    os.makedirs(os.path.dirname(model_out) or '.', exist_ok=True)
    joblib.dump(model, model_out)
    artifacts['features'] = FEATURES
    joblib.dump(artifacts, artifacts_out)
    # evaluate (invert)
    preds_val = np.expm1(model.predict(val[FEATURES], num_iteration=model.best_iteration))
    val_mae = mean_absolute_error(val['units_sold'], preds_val)
    print("LGB(log1p) validation MAE (after expm1):", val_mae)
    return model_out, artifacts_out, val_mae

def recursive_forecast_lgb_log(model, df_all, sku, last_date, horizon, features):
    # similar to existing recursive_forecast but apply expm1 to predictions
    hist = df_all[df_all['sku']==sku].sort_values('date').copy()
    out=[]
    for h in range(horizon):
        d = last_date + pd.Timedelta(days=h+1)
        row = {'date':d}
        # build same date/static/lag/roll features as before (reuse code from lgb_trainer)
        # For brevity, we will call the existing helper in lgb_trainer if available, else implement inline.
        # Here we implement inline minimal fields assuming artifacts contain price etc.
        row['dow']=d.dayofweek; row['month']=d.month; row['day']=d.day; row['week']=d.isocalendar()[1]
        for c in ['price','discount','competitor_price','inventory','category_enc','region_enc','weather_enc','season_enc','is_holiday']:
            if c in hist.columns:
                row[c]=hist[c].iloc[-1]
        for lag in (1,7,14):
            ld = d - pd.Timedelta(days=lag)
            val = hist.loc[hist['date']==ld,'units_sold']
            row[f'lag_{lag}'] = float(val.values[0]) if len(val) else 0.0
        for w in (7,28):
            s = d - pd.Timedelta(days=w)
            vals = hist[(hist['date']>=s)&(hist['date']<d)]['units_sold']
            row[f'roll_mean_{w}'] = float(vals.mean()) if len(vals) else 0.0
            row[f'roll_std_{w}'] = float(vals.std()) if len(vals) else 0.0
        if (hist['units_sold']>0).any():
            row['days_since_sale'] = (d - hist[hist['units_sold']>0]['date'].max()).days
        else:
            row['days_since_sale']=999
        X = pd.DataFrame([row])[features].fillna(0)
        ylog = model.predict(X, num_iteration=model.best_iteration)[0]
        yhat = float(max(0.0, np.expm1(ylog)))
        out.append({'sku':sku,'date':d,'pred':yhat})
        new_row = row.copy(); new_row['units_sold']=yhat; new_row['date']=d
        hist = pd.concat([hist, pd.DataFrame([new_row])], ignore_index=True)
    return pd.DataFrame(out)

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--processed_dir', default='data/processed')
    p.add_argument('--model_out', default='models/lgb_model_log.pkl')
    p.add_argument('--artifacts_out', default='artifacts/artifacts_lgb_log.pkl')
    p.add_argument('--val_days', type=int, default=28)
    p.add_argument('--test_days', type=int, default=28)
    args = p.parse_args()
    train_xgb = train_lgb_log(args.processed_dir, args.model_out, args.artifacts_out, val_days=args.val_days, test_days=args.test_days)
    print("Done:", train_xgb)
