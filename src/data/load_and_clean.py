import pandas as pd
import os
import sys

def clean_data(user_path):
    raw_path = os.path.join(user_path, "retail_store_inventory.csv")
    output_path = os.path.join(user_path, "training_data.csv")

    print(f"🚀 Cleaning: {raw_path}")

    if not os.path.exists(raw_path):
        print("❌ No raw data found. Please upload a CSV first.")
        return

    df = pd.read_csv(raw_path)

    # ✅ Create SKU identifier
    df['sku'] = df['store id'].astype(str) + "_" + df['product id'].astype(str)

    # ✅ Drop rows with missing critical fields
    df = df.dropna(subset=['sku', 'date', 'units sold'])

    # ✅ Convert date to datetime
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])

    # ✅ Sort for time series modeling
    df = df.sort_values(['sku', 'date'])

    # ✅ Clip negative sales
    df['units sold'] = df['units sold'].clip(lower=0)

    # ✅ Fill missing categorical features (if present)
    for col in ['category', 'region']:
        if col in df.columns:
            df[col] = df[col].fillna('unknown')

    # ✅ Save cleaned output
    df.to_csv(output_path, index=False)
    print(f"✅ Saved cleaned data: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❌ Please provide user folder path as argument.")
    else:
        clean_data(sys.argv[1])
