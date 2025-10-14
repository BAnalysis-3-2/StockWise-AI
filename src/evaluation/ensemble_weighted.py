import pandas as pd

def blend_forecasts():
    print("🔁 Blending LightGBM and XGBoost")

    lgb = pd.read_csv("reports/forecasts_lgb_7d.csv")
    xgb = pd.read_csv("reports/forecasts_xgb_7d.csv")

    blended = lgb[['sku', 'date']].copy()
    blended['forecast'] = 0.49 * lgb['forecast'] + 0.51 * xgb['forecast']

    blended.to_csv("reports/forecast_weighted.csv", index=False)
    print("✅ Saved: reports/forecast_weighted.csv")

if __name__ == "__main__":
    blend_forecasts()
