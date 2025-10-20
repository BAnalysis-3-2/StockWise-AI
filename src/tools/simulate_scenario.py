import argparse
import pandas as pd
import joblib
import os

def simulate_forecast(sku, price, discount, inventory, holiday_flag, user_folder):
    # === Normalize input SKU ===
    sku = sku.strip().upper()

    # === Load training features ===
    features_path = os.path.join(user_folder, "data", "processed", "training_features.csv")
    if not os.path.exists(features_path):
        print("❌ Feature file not found. Run the pipeline first.")
        return

    df = pd.read_csv(features_path)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    print(f"✅ Loaded training_features.csv with {len(df)} rows")
    print("📄 Columns:", df.columns.tolist())

    # === Reconstruct SKU column if missing ===
    if "sku" not in df.columns:
        if "store_id" in df.columns and "product_id" in df.columns:
            df["sku"] = df["store_id"].astype(str).str.strip().str.upper() + "_" + df["product_id"].astype(str).str.strip().str.upper()
            print("🔧 Reconstructed SKU column from store_id and product_id")
        else:
            print("❌ SKU column missing and cannot be constructed. Available columns:", df.columns.tolist())
            return
    else:
        df["sku"] = df["sku"].astype(str).str.strip().str.upper()

    # === Check if SKU exists ===
    if sku not in df["sku"].values:
        print(f"❌ SKU '{sku}' not found in training data.")
        print("✅ Available SKUs:", df["sku"].unique().tolist())
        return

    # === Prepare simulation input ===
    template = df[df["sku"] == sku].iloc[-1:].copy()
    template["price"] = price
    template["price_change_7d"] = 0
    template["discount"] = discount
    template["discount_flag"] = int(discount > 0)
    template["inventory_level"] = inventory
    template["inventory_ratio"] = inventory / (template["rolling_mean_7"].values[0] + 1)
    template["holiday_flag"] = holiday_flag

    # === Define feature columns ===
    feature_columns = [
        "lag_1", "lag_7", "lag_14",
        "rolling_mean_3", "rolling_mean_7", "rolling_std_7",
        "day_of_week", "month",
        "category", "region", "weather_condition", "seasonality",
        "price", "price_change_7d",
        "discount", "discount_flag",
        "inventory_level", "inventory_ratio",
        "holiday_flag"
    ]

    # === Encode categorical features ===
    available_features = [col for col in feature_columns if col in template.columns]
    missing_features = [col for col in feature_columns if col not in template.columns]
    if missing_features:
        print(f"⚠️ Skipping missing features: {missing_features}")

    object_cols = template[available_features].select_dtypes(include="object").columns.tolist()
    if object_cols:
        print(f"🔤 Encoding categorical features: {object_cols}")
        template = pd.get_dummies(template, columns=object_cols, drop_first=True)

    # === Align features with model expectations ===
    forecasts = {}

    try:
        model_lgb = joblib.load("models/lgb/lgb_model.pkl")
        with open("models/lgb/features.txt") as f:
            lgb_features = f.read().splitlines()
        lgb_input = template.reindex(columns=lgb_features, fill_value=0)
        forecasts["LGB"] = model_lgb.predict(lgb_input)[0]
    except Exception as e:
        forecasts["LGB"] = f"⚠️ Error: {e}"

    try:
        model_xgb = joblib.load("models/xgb/xgb_model.pkl")
        with open("models/xgb/features.txt") as f:
            xgb_features = f.read().splitlines()
        xgb_input = template.reindex(columns=xgb_features, fill_value=0)
        forecasts["XGB"] = model_xgb.predict(xgb_input)[0]
    except Exception as e:
        forecasts["XGB"] = f"⚠️ Error: {e}"

    # === Output results ===
    print(f"\n🧪 Simulation Results for {sku}")
    print(f"Price: R{price} | Discount: {discount*100:.0f}% | Inventory: {inventory} | Holiday: {'Yes' if holiday_flag else 'No'}\n")

    for model_name in ["LGB", "XGB"]:
        result = forecasts[model_name]
        if isinstance(result, (int, float)):
            print(f"- {model_name}: {result:.2f} units")
        else:
            print(f"- {model_name}: {result}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate demand forecast for a SKU")
    parser.add_argument("--sku", type=str, required=True)
    parser.add_argument("--price", type=float, required=True)
    parser.add_argument("--discount", type=float, default=0.0)
    parser.add_argument("--inventory", type=int, default=100)
    parser.add_argument("--holiday", type=int, choices=[0, 1], default=0)
    parser.add_argument("--user_folder", type=str, required=True)

    args = parser.parse_args()
    simulate_forecast(args.sku, args.price, args.discount, args.inventory, args.holiday, args.user_folder)
