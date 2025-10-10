"""
src/models/arima_trainer.py

Per-SKU ARIMA forecasting utilities and bulk forecast script.

Notes:
- Uses statsmodels.tsa.arima.model.ARIMA with a simple default order (1,0,1).
- Fits each SKU separately. Skips SKUs that fail to fit.
- For production or better performance consider pmdarima.auto_arima, seasonal orders, or model selection.
"""
import os
import warnings
from typing import List, Optional
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from tqdm import tqdm

warnings.filterwarnings("ignore")

def _fit_arima_and_forecast(series: pd.Series, order=(1,0,1), horizon: int = 7):
    """
    Fit ARIMA on given pd.Series (indexed by date) and forecast 'horizon' points.
    Returns numpy array of length 'horizon' (float forecasts).
    """
    # require at least some non-NA points
    if series.dropna().shape[0] < 10:
        # not enough history; return zeros
        return np.zeros(horizon)
    try:
        m = ARIMA(series, order=order).fit(method_kwargs={"warn_convergence": False})
        fc = m.forecast(steps=horizon)
        return np.array(fc).astype(float)
    except Exception as e:
        # fitting failed, fallback to naive last-value or zeros
        last = series.dropna().iloc[-1] if series.dropna().shape[0] > 0 else 0.0
        return np.array([float(last) for _ in range(horizon)])

def bulk_arima_forecast(processed_dir: str,
                        horizon: int = 7,
                        out_csv: str = "forecasts_arima.csv",
                        sku_list: Optional[List[str]] = None,
                        order=(1,0,1)) -> str:
    """
    Load processed CSVs (train/val/test) to build history and run ARIMA forecasts for each SKU.
    - If sku_list is provided, only forecast those SKUs.
    - Saves CSV with columns: sku,date,pred
    """
    # load full history
    train = pd.read_csv(os.path.join(processed_dir, "training_data.csv"), parse_dates=["date"])
    val   = pd.read_csv(os.path.join(processed_dir, "validation_data.csv"), parse_dates=["date"])
    test  = pd.read_csv(os.path.join(processed_dir, "test_data.csv"), parse_dates=["date"])
    df_all = pd.concat([train, val, test], ignore_index=True).sort_values(["sku","date"]).reset_index(drop=True)

    # restrict to requested SKUs (optional)
    skus = sku_list if sku_list is not None else df_all["sku"].unique().tolist()

    last_date = df_all["date"].max()
    results = []
    print(f"Running ARIMA forecasts for {len(skus)} SKUs, horizon={horizon}...")
    for sku in tqdm(skus):
        hist = df_all[df_all["sku"] == sku].sort_values("date")
        # use units_sold series indexed by date
        series = pd.Series(hist["units_sold"].values, index=hist["date"])
        preds = _fit_arima_and_forecast(series, order=order, horizon=horizon)
        for h in range(horizon):
            results.append({
                "sku": sku,
                "date": (last_date + pd.Timedelta(days=h+1)).date().isoformat(),
                "pred": float(preds[h])
            })
    out_df = pd.DataFrame(results)
    out_df.to_csv(out_csv, index=False)
    print(f"Saved ARIMA forecasts to {out_csv}")
    return out_csv
