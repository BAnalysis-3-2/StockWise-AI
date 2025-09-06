# file: main.py
import argparse
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

from src.preprocess import prepare_data
from src.features import build_feature_matrix
from src.config import DATE_COL, TARGET
from src.models.arima_model import fit_forecast_arima
from src.models.xgb_model import train_xgb, forecast_xgb
from src.models.lstm_model import fit_forecast_lstm
from src.models.ensemble import blend
from src.models.inventory import compute_inventory_plan
from src.models.evaluate import evaluate_forecasts

# ---------------------------
# Utilities
# ---------------------------


def ensure_dirs(*paths: str):
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)

# ---------------------------
# Pipeline
# ---------------------------

def run_pipeline(
    inv_path: str,
    tx_path: str,
    out_forecast: str,
    out_reorder: str,
    horizon: int = 14,
    service_level: float = 0.95,
    default_lead_days: float = 7.0
):
    print("Preparing data ...")
    merged = prepare_data(
        inventory_file=Path(inv_path).name,
        transactions_file=Path(tx_path).name
    )

    print("Building features ...")
    merged = build_feature_matrix(merged)

    # Drop rows with NaN from lag features before training
    train_df = merged.dropna(subset=[TARGET] + [c for c in merged.columns if c.startswith("lag_")])

    print("Forecasting with ARIMA, XGB, and LSTM ...")
    fc_rows = []

    for pkey, g in train_df.groupby("Product_Item"):
        g = g.sort_values(DATE_COL)
        y = g[TARGET]

        # --- 1. ARIMA ---
        arima_pred = fit_forecast_arima(y, horizon=horizon)

        # --- 2. LSTM ---
        lstm_pred = fit_forecast_lstm(y, horizon=horizon)

        # --- 3. XGB ---
        X_train = g.dropna(subset=[TARGET])
        y_train = X_train[TARGET]
        model_xgb = train_xgb(X_train, y_train)

        last_date = g[DATE_COL].max()
        future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")
        X_future = pd.DataFrame({DATE_COL: future_dates, "Product_Item": pkey})
        X_future = build_feature_matrix(pd.concat([g, X_future], ignore_index=True)).tail(horizon)
        xgb_pred = forecast_xgb(model_xgb, X_future)

        # --- 4. Blend using recent performance ---
        pred_dict = {
            "arima": arima_pred,
            "lstm": lstm_pred,
            "xgb": xgb_pred
        }
        y_recent = g[TARGET].values
        blended_pred, weights = blend(pred_dict, y_recent, window=min(28, len(y_recent)))

        w_arima = round(weights.get("arima", 0.0), 3)
        w_lstm = round(weights.get("lstm", 0.0), 3)
        w_xgb = round(weights.get("xgb", 0.0), 3)

        # --- 5. Append blended forecast rows ---
        meta = g.iloc[-1]
        for d, yhat in zip(future_dates, blended_pred):
            fc_rows.append({
                "date": d,
                "Product_ID": meta.get("Product_ID", np.nan),
                "Product_Item": pkey,
                "Product_Name": meta.get("Product_Name", np.nan),
                "forecast": float(max(0.0, yhat)),
                "hist_mean": float(y.mean()),
                "hist_std": float(y.std()),
                "Stock_Quantity": float(meta.get("Stock_Quantity", 0.0)),
                "Reorder_Level": float(meta.get("Reorder_Level", 0.0)),
                "Reorder_Quantity": float(meta.get("Reorder_Quantity", 0.0)),
                "Unit_Price": float(meta.get("Unit_Price", np.nan)),
                "w_arima": w_arima,
                "w_lstm": w_lstm,
                "w_xgb": w_xgb
            })

    # --- 6. Save forecasts ---
    fc_df = pd.DataFrame(fc_rows)
    ensure_dirs(Path(out_forecast).parent)
    fc_df.to_csv(out_forecast, index=False)
    print(f"Saved forecast to {out_forecast} ({len(fc_df)} rows)")

    # Suppose actual_df contains actual demand for the forecast period
    # # and fc_df is your forecast DataFrame
    eval_df = evaluate_forecasts(
    df=actual_df.merge(fc_df, on=["date", "Product_Item"]),
    actual_col="actual_demand",
    forecast_col="forecast",
    group_col="Product_Item"
    )

    print(eval_df)

    # --- 7. Compute and save reorder plan ---
    print("Computing reorder plan ...")
    plan_df = compute_inventory_plan(fc_df, service_level=service_level, default_lead=default_lead_days)
    ensure_dirs(Path(out_reorder).parent)
    plan_df.to_csv(out_reorder, index=False)
    print(f"Saved reorder plan to {out_reorder} ({len(plan_df)} rows)")

# ---------------------------
# CLI
# ---------------------------

def parse_args():
    p = argparse.ArgumentParser(description="AI Inventory Forecasting & Reorder Pipeline")
    p.add_argument("--inventory", default="data/raw/Inventory_and_Sales_Dataset.csv",
                   help="Path to inventory master CSV")
    p.add_argument("--transactions", default="data/raw/Retail_Transaction_Dataset.csv",
                   help="Path to transactions CSV")
    p.add_argument("--horizon", type=int, default=14, help="Forecast horizon (days)")
    p.add_argument("--service_level", type=float, default=0.95, help="Service level for safety stock (0-1)")
    p.add_argument("--lead_time_days", type=float, default=7.0, help="Default lead time in days")
    p.add_argument("--out_forecast", default="outputs/forecasts/forecast.csv",
                   help="Output path for forecast CSV")
    p.add_argument("--out_reorder", default="outputs/reports/reorder_plan.csv",
                   help="Output path for reorder plan CSV")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        inv_path=args.inventory,
        tx_path=args.transactions,
        out_forecast=args.out_forecast,
        out_reorder=args.out_reorder,
        horizon=args.horizon,
        service_level=args.service_level,
        default_lead_days=args.lead_time_days
    )
