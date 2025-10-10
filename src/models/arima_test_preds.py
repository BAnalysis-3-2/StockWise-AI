# src/models/arima_test_preds.py
import os, pandas as pd
from src.models.arima_trainer import _fit_arima_and_forecast
from src.models.lgb_trainer import prepare_features_from_processed, split_by_date

def main(processed_dir='data/processed', out='reports/arima_test_preds.csv', order=(1,0,1)):
    df_full, _ = prepare_features_from_processed(processed_dir)
    train,val,test = split_by_date(df_full)
    # We will do one-step ARIMA prediction for each test date per SKU (same as earlier backtest)
    df_all = pd.concat([train, val], ignore_index=True).sort_values(['sku','date']).reset_index(drop=True)
    results=[]
    last_date = test['date'].max()
    print("Running one-step ARIMA predictions for test slice...")
    for sku, grp in test.groupby('sku'):
        # for each test row date, fit on history < date and predict 1-step for that date
        for _, row in grp.iterrows():
            date = row['date']
            hist = df_all[df_all['sku']==sku]
            series = hist[ hist['date']<date ]['units_sold']
            if series.dropna().shape[0] < 5:
                pred = float(series.dropna().iloc[-1]) if series.dropna().shape[0]>0 else 0.0
            else:
                try:
                    pred = float(_fit_arima_and_forecast(series, order=order, horizon=1)[0])
                except:
                    pred = float(series.dropna().iloc[-1]) if series.dropna().shape[0]>0 else 0.0
            results.append({'sku': sku, 'date': date, 'pred': pred})
    out_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    out_df.to_csv(out, index=False)
    print("Saved ARIMA test preds to", out)

if __name__ == '__main__':
    main()
