import pandas as pd
import os
import sys

# === Step 0: Get user folder from CLI ===
if len(sys.argv) < 2:
    print("❌ Please provide a user folder path.")
    exit()

user_folder = sys.argv[1]
raw_path = os.path.join(user_folder, "data", "raw")
processed_path = os.path.join(user_folder, "data", "processed")
os.makedirs(processed_path, exist_ok=True)

raw_file = os.path.join(raw_path, "retail_store_inventory.csv")
training_data_path = os.path.join(processed_path, "training_data.csv")
features_path = os.path.join(processed_path, "training_features.csv")

# === Step 1: Load raw data ===
if not os.path.exists(raw_file):
    print("❌ Raw file not found:", raw_file)
    exit()

df = pd.read_csv(raw_file)
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# === Step 2: Normalize known column variants ===
column_map = {
    "inventory_level": "inventory",
    "units_sold": "units_sold",
    "units_ordered": "units_ordered",
    "demand_forecast": "demand_forecast",
    "holiday/promotion": "holiday",
    "store_id": "store_id",
    "product_id": "product_id"
}
df.rename(columns={k: v for k, v in column_map.items() if k in df.columns}, inplace=True)

# === Step 3: Validate required columns ===
required = ["date", "price", "discount", "inventory", "product_id", "demand_forecast"]
missing = [col for col in required if col not in df.columns]
if missing:
    print("❌ Missing required columns:", missing)
    exit()

# === Step 4: Construct SKU ===
if "store_id" in df.columns and "product_id" in df.columns:
    df["sku"] = df["store_id"].astype(str).str.strip().str.upper() + "_" + df["product_id"].astype(str).str.strip().str.upper()
elif "product_id" in df.columns:
    df["sku"] = df["product_id"].astype(str).str.strip().str.upper()
else:
    print("❌ Cannot construct SKU — missing 'product_id' or both 'store_id' and 'product_id'")
    exit()

# === Step 5: Clean and format ===
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["sku", "date"]).reset_index(drop=True)

# === Step 6: Save training_data.csv ===
df.to_csv(training_data_path, index=False)
print(f"✅ Saved cleaned training data to: {training_data_path}")

# === Step 7: Feature engineering ===
features = []
for sku in df["sku"].unique():
    sku_df = df[df["sku"] == sku].copy()

    # Lag features
    sku_df["lag_1"] = sku_df["demand_forecast"].shift(1)
    sku_df["lag_7"] = sku_df["demand_forecast"].shift(7)
    sku_df["lag_14"] = sku_df["demand_forecast"].shift(14)

    # Rolling stats
    sku_df["rolling_mean_3"] = sku_df["demand_forecast"].rolling(3).mean()
    sku_df["rolling_mean_7"] = sku_df["demand_forecast"].rolling(7).mean()
    sku_df["rolling_std_7"] = sku_df["demand_forecast"].rolling(7).std()

    # Temporal features
    sku_df["day_of_week"] = sku_df["date"].dt.dayofweek
    sku_df["month"] = sku_df["date"].dt.month
    sku_df["year"] = sku_df["date"].dt.year

    # Price change
    sku_df["price_change_7d"] = sku_df["price"].pct_change(periods=7)

    # Discount flag
    sku_df["discount_flag"] = (sku_df["discount"] > 0).astype(int)

    # Inventory features
    sku_df["inventory_level"] = sku_df["inventory"]
    sku_df["inventory_ratio"] = sku_df["inventory"] / (sku_df["rolling_mean_7"] + 1)

    # Holiday flag
    sku_df["holiday_flag"] = sku_df["holiday"]

    features.append(sku_df)

df_feat = pd.concat(features).reset_index(drop=True)

# === Step 8: Drop rows missing essential features only ===
required_features = ["lag_1", "rolling_mean_3", "rolling_mean_7"]
df_feat = df_feat.dropna(subset=required_features)

# === Step 9: Save training_features.csv ===
df_feat.to_csv(features_path, index=False)
print(f"✅ Saved training features to: {features_path}")
print(f"📊 Final training features shape: {df_feat.shape}")
print("📄 Sample SKUs:", df_feat["sku"].unique()[:5])
