import pandas as pd
from sklearn.metrics import mean_absolute_error

def smape(y_true, y_pred):
    return 100 * (abs(y_pred - y_true) / (abs(y_pred) + abs(y_true))).mean()

def evaluate_model(name, forecast_path, actuals):
    df = pd.read_csv(forecast_path)
    df = df.merge(actuals, on=['sku', 'date'], how='left')
    df = df.dropna(subset=['units_sold', 'forecast'])

    if len(df) == 0:
        print(f" Skipping {name}: no valid rows for evaluation")
        return {
            'Model': name,
            'MAE': 'N/A',
            'RMSE': 'N/A',
            'SMAPE (%)': 'N/A'
        }

    mae = mean_absolute_error(df['units_sold'], df['forecast'])
    rmse = ((df['forecast'] - df['units_sold']) ** 2).mean() ** 0.5
    smape_score = smape(df['units_sold'], df['forecast'])

    return {
        'Model': name,
        'MAE': round(mae, 2),
        'RMSE': round(rmse, 2),
        'SMAPE (%)': round(smape_score, 2)
    }


def main():
    print(" Comparing model performance")

    actuals = pd.read_csv("data/processed/training_features.csv")
    actuals = actuals[['sku', 'date', 'units sold']]
    actuals.rename(columns={'units sold': 'units_sold'}, inplace=True)

    models = [
        ("ARIMA", "reports/forecasts_arima_7d.csv"),
        ("LightGBM", "reports/forecasts_lgb_7d.csv"),
        ("XGBoost", "reports/forecasts_xgb_7d.csv"),
        ("Weighted Blend", "reports/forecast_weighted.csv"),
        ("Fallback Ensemble", "reports/forecast_fallback.csv")
    ]

    results = [evaluate_model(name, path, actuals) for name, path in models]
    df_results = pd.DataFrame(results)
    print("\n Model Comparison:\n")
    print(df_results.to_string(index=False))

if __name__ == "__main__":
    main()
