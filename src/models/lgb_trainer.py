"""
src/models/lgb_trainer.py

LightGBM training & prediction pipeline.
- Saves model and artifacts via joblib.

Expectations:
- Processed CSVs (training_data.csv, validation_data.csv, test_data.csv) are in processed_dir.
- Feature engineering file src/data/feature_engineering.py exists and provides:
    create_lag_roll_features, label_encode_columns, get_default_feature_list
"""
import os
import joblib
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from tqdm import tqdm
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error

from src.data.feature_engineering import (
    create_lag_roll_features,
    label_encode_columns,
    get_default_feature_list,
)


# Helpers
def _safe_label_transform(le, series: pd.Series) -> pd.Series:
    """
    Map series values to label encoder integers safely.
    Unknown values map to -1.
    """
    if le is None:
        return pd.Series(-1, index=series.index, dtype=int)
    classes = list(le.classes_)
    mapping = {v: i for i, v in enumerate(classes)}
    return series.map(mapping).fillna(-1).astype(int)


# Feature preparation

def prepare_features_from_processed(
    processed_dir: str,
    lags: List[int] = [1, 7, 14],
    windows: List[int] = [7, 28],
) -> Tuple[pd.DataFrame, Dict]:
    """
    Load processed train/val/test CSVs, concatenate, compute lag & rolling features,
    label-encode categorical columns, and return df_full and artifacts dictionary.
    """
    train_path = os.path.join(processed_dir, "training_data.csv")
    val_path = os.path.join(processed_dir, "validation_data.csv")
    test_path = os.path.join(processed_dir, "test_data.csv")

    for p in (train_path, val_path, test_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Expected processed file: {p} not found. Run prepare first.")

    dfs = [
        pd.read_csv(train_path, parse_dates=["date"]),
        pd.read_csv(val_path, parse_dates=["date"]),
        pd.read_csv(test_path, parse_dates=["date"]),
    ]
    df_full = pd.concat(dfs, ignore_index=True).sort_values(["sku", "date"]).reset_index(drop=True)

    # create lags and rolling features
    df_full = create_lag_roll_features(df_full, group_col="sku", target_col="units_sold", lags=lags, windows=windows)

    # label encode some categorical columns
    cat_cols = [c for c in ["category", "region", "weather", "season"] if c in df_full.columns]
    df_full, encoders = label_encode_columns(df_full, cat_cols)

    # ensure is_holiday exists
    if "is_holiday" not in df_full.columns:
        df_full["is_holiday"] = 0

    artifacts = {"encoders": encoders, "last_date": df_full["date"].max()}
    return df_full, artifacts

def split_by_date(df_full: pd.DataFrame, val_days: int = 28, test_days: int = 28) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    max_date = df_full["date"].max()
    test_start = max_date - pd.Timedelta(days=test_days - 1)
    val_start = test_start - pd.Timedelta(days=val_days)
    train = df_full[df_full["date"] < val_start].copy()
    val = df_full[(df_full["date"] >= val_start) & (df_full["date"] < test_start)].copy()
    test = df_full[df_full["date"] >= test_start].copy()
    return train, val, test


# Training

def train_lightgbm_from_processed(
    processed_dir: str,
    model_out: str,
    artifacts_out: str,
    val_days: int = 28,
    test_days: int = 28,
    lgb_params: Optional[Dict] = None,
    num_boost_round: int = 2000,
    early_stopping_rounds: int = 50,
    verbose_eval: int = 100,
) -> Dict:
    """
    Train a global LightGBM model from processed CSVs and save model + artifacts.

    Returns a dict with model_path, artifacts_path, and val_mae.
    """
    print("Preparing features from processed CSVs...")
    df_full, artifacts = prepare_features_from_processed(processed_dir)
    print("Splitting into train/val/test by date...")
    train, val, test = split_by_date(df_full, val_days=val_days, test_days=test_days)

    # determine features
    default_feats = get_default_feature_list()
    FEATURES = [f for f in default_feats if f in df_full.columns]
    print("Using features:", FEATURES)

    # drop rows with NaNs in chosen features (warm-up period where lags are NaN)
    train = train.dropna(subset=FEATURES)
    val = val.dropna(subset=FEATURES)

    print(f"Train size: {len(train)}  Val size: {len(val)}")

    if lgb_params is None:
        lgb_params = {
            "objective": "poisson",
            "metric": "mae",
            "learning_rate": 0.05,
            "num_leaves": 64,
            "verbosity": -1,
        }

    dtrain = lgb.Dataset(train[FEATURES], label=train["units_sold"])
    dval = lgb.Dataset(val[FEATURES], label=val["units_sold"], reference=dtrain)

    print("Training LightGBM...")

    # Use callbacks for logging and early stopping for compatibility across LightGBM versions
    callbacks = [lgb.log_evaluation(verbose_eval), lgb.early_stopping(stopping_rounds=early_stopping_rounds)]
    model = lgb.train(
        lgb_params,
        dtrain,
        valid_sets=[dval],
        num_boost_round=num_boost_round,
        callbacks=callbacks,
    )

    # evaluate on val
    preds_val = model.predict(val[FEATURES], num_iteration=model.best_iteration)
    val_mae = mean_absolute_error(val["units_sold"], preds_val)
    print("Validation MAE:", val_mae)

    # persist model and artifacts
    os.makedirs(os.path.dirname(model_out) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(artifacts_out) or ".", exist_ok=True)
    joblib.dump(model, model_out)

    artifacts["features"] = FEATURES
    # Note: encoders (LabelEncoders) are not picklable across Python versions; joblib should handle it here.
    joblib.dump(artifacts, artifacts_out)

    print(f"Saved model to {model_out}")
    print(f"Saved artifacts to {artifacts_out}")

    return {"model_path": model_out, "artifacts_path": artifacts_out, "val_mae": float(val_mae)}


# Recursive forecasting

def recursive_forecast_for_sku(model, df_hist: pd.DataFrame, sku: str, last_date: pd.Timestamp, horizon: int, features: List[str]) -> pd.DataFrame:
    """
    Recursively forecast horizon days for a single SKU using the provided model and history.
    The function appends predicted days to the working history so subsequent lags use predicted values.
    """
    hist = df_hist[df_hist["sku"] == sku].sort_values("date").copy()
    out_rows = []
    for h in range(horizon):
        d = last_date + pd.Timedelta(days=h + 1)
        row = {"sku": sku, "date": d}
        # date features
        row["dow"] = d.dayofweek
        row["month"] = d.month
        row["day"] = d.day
        row["week"] = d.isocalendar()[1]
        # static fields: pick last available
        for c in ["price", "discount", "competitor_price", "inventory", "category_enc", "region_enc", "weather_enc", "season_enc", "is_holiday"]:
            if c in hist.columns:
                row[c] = hist[c].iloc[-1]
        # lags
        for lag in (1, 7, 14):
            ld = d - pd.Timedelta(days=lag)
            val = hist.loc[hist["date"] == ld, "units_sold"]
            row[f"lag_{lag}"] = float(val.values[0]) if len(val) else 0.0
        # rolling windows
        for w in (7, 28):
            window_start = d - pd.Timedelta(days=w)
            vals = hist[(hist["date"] >= window_start) & (hist["date"] < d)]["units_sold"]
            row[f"roll_mean_{w}"] = float(vals.mean()) if len(vals) else 0.0
            row[f"roll_std_{w}"] = float(vals.std()) if len(vals) else 0.0
        # days since last sale
        if (hist["units_sold"] > 0).any():
            row["days_since_sale"] = (d - hist[hist["units_sold"] > 0]["date"].max()).days
        else:
            row["days_since_sale"] = 999
        X = pd.DataFrame([row])[features].fillna(0)
        yhat = model.predict(X, num_iteration=model.best_iteration)[0]
        yhat = float(max(0.0, yhat))
        out_rows.append({"sku": sku, "date": d, "pred": yhat})
        # append predicted day to hist for next iteration
        new_row = row.copy()
        new_row["units_sold"] = yhat
        hist = pd.concat([hist, pd.DataFrame([new_row])], ignore_index=True)
    return pd.DataFrame(out_rows)

def bulk_forecast(model_path: str, artifacts_path: str, processed_dir: str, horizon: int = 7, out_csv: str = "forecasts.csv") -> str:
    """
    Run recursive forecasts for all SKUs and save to out_csv.
    """
    if not os.path.exists(model_path) or not os.path.exists(artifacts_path):
        raise FileNotFoundError("Model or artifacts not found.")

    model = joblib.load(model_path)
    artifacts = joblib.load(artifacts_path)
    FEATURES = artifacts.get("features", None)
    if FEATURES is None:
        raise ValueError("Artifacts missing 'features' list.")

    # load full processed dataset to get history
    train = pd.read_csv(os.path.join(processed_dir, "training_data.csv"), parse_dates=["date"])
    val = pd.read_csv(os.path.join(processed_dir, "validation_data.csv"), parse_dates=["date"])
    test = pd.read_csv(os.path.join(processed_dir, "test_data.csv"), parse_dates=["date"])
    df_all = pd.concat([train, val, test], ignore_index=True).sort_values(["sku", "date"]).reset_index(drop=True)

    # apply encoders if needed (safely)
    encoders = artifacts.get("encoders", {})
    for c, le in encoders.items():
        enc_col = c + "_enc"
        if c in df_all.columns and enc_col not in df_all.columns:
            df_all[c] = df_all[c].fillna("NA").astype(str)
            df_all[enc_col] = _safe_label_transform(le, df_all[c])

    last_date = df_all["date"].max()
    skus = df_all["sku"].unique()
    out_list = []
    print(f"Running bulk recursive forecast for {len(skus)} SKUs for horizon={horizon} days...")
    for sku in tqdm(skus):
        try:
            fc = recursive_forecast_for_sku(model, df_all, sku, last_date, horizon, FEATURES)
            out_list.append(fc)
        except Exception as e:
            # don't fail the whole loop if one SKU errors; log and continue
            print(f"Warning: forecasting failed for SKU={sku}: {e}")

    if not out_list:
        raise RuntimeError("No forecasts produced.")

    out_df = pd.concat(out_list, ignore_index=True)
    out_df.to_csv(out_csv, index=False)
    print(f"Forecasts saved to {out_csv}")
    return out_csv
