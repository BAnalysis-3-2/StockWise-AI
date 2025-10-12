import pandas as pd
import os
import sys

def build_features(user_path):
    input_path = os.path.join(user_path, "training_data.csv")
    output_path = os.path.join(user_path, "training_features.csv")

    print(f"🔧 Building features from: {input_path}")

    if not os.path.exists(input_path):
        print("❌ Training data not found. Please run load_and_clean first.")
        return

    df = pd.read_csv(input_path)
    df['date'] = pd.to_datetime(df['date'])

    # ✅ Filter SKUs with enough history and variance
    valid_skus = []
    for sku, group in df.groupby('sku'):
        if len(group) >= 90 and group['units sold'].std() >= 1:
            valid_skus.append(sku)
    df = df[df['sku'].isin(valid_skus)]

    # ✅ Sort for time series modeling
    df = df.sort_values(['sku', 'date'])

    # ✅ Lag features
    df['lag_1'] = df.groupby('sku')['units sold'].shift(1)
    df['lag_7'] = df.groupby('sku')['units sold'].shift(7)
    df['lag_14'] = df.groupby('sku')['units sold'].shift(14)

    # ✅ Rolling averages and volatility
    df['rolling_mean_3'] = df.groupby('sku')['units sold'].transform(lambda x: x.shift(1).rolling(3).mean())
    df['rolling_mean_7'] = df.groupby('sku')['units sold'].transform(lambda x: x.shift(1).rolling(7).mean())
    df['rolling_std_7'] = df.groupby('sku')['units sold'].transform(lambda x: x.shift(1).rolling(7).std())

    # ✅ Temporal features
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month

    # ✅ External features (raw)
    for col in ['price', 'discount', 'inventory_level', 'holiday_flag']:
        if col not in df.columns:
            df[col] = 0  # placeholder if missing

    # ✅ External features (engineered)
    df['price_change_7d'] = df.groupby('sku')['price'].transform(lambda x: x.pct_change(periods=7).fillna(0))
    df['discount_flag'] = (df['discount'] > 0).astype(int)
    df['inventory_ratio'] = df['inventory_level'] / (df['rolling_mean_7'] + 1)

    # ✅ Encode categorical features
    for col in ['category', 'region', 'weather condition', 'seasonality']:
        if col in df.columns:
            df[col] = df[col].astype('category').cat.codes

    # ✅ Drop rows with missing lag values
    df = df.dropna(subset=[
        'lag_1', 'lag_7', 'lag_14',
        'rolling_mean_3', 'rolling_mean_7', 'rolling_std_7'
    ])

    df.to_csv(output_path, index=False)
    print(f"✅ Saved feature-rich data: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❌ Please provide user folder path as argument.")
    else:
        build_features(sys.argv[1])
