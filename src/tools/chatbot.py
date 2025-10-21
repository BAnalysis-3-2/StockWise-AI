import pandas as pd
import os
import re
import json
import subprocess

# Load user-specific data
def load_forecast(user_folder):
    forecast_path = os.path.join(user_folder, "reports", "sku_forecast.csv")
    
    if not os.path.exists(forecast_path):
        
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(forecast_path)
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
        return df
    except Exception as e:
        print(f"❌ Failed to load forecast file: {e}")
        return pd.DataFrame()

def load_mae(user_folder):
    try:
        df = pd.read_csv(os.path.join(user_folder, "mae_per_sku.csv"))
        df["sku"] = df["sku"].astype(str).str.strip()
        return df
    except:
        return pd.DataFrame()

# Validate and create user folder
def get_user_folder():
    print("👋 Welcome! Choose user type")
    print("1. New user")
    print("2. Returning user")
    choice = input(" Your choice (1 or 2): ").strip()

    if choice not in ["1", "2"]:
        print("❌ Invalid choice. Please enter 1 or 2.")
        return None

    user_id = input(" Enter your user ID (e.g. Thabo_42120): ").strip()
    if not re.match(r"^[a-zA-Z]+_\d{5}$", user_id):
        print("❌ Invalid ID format. Use: Name_12345")
        return None

    user_path = os.path.join("users", user_id)

    if choice == "1":
        if os.path.exists(user_path):
            print(f" User already exists. Switching to returning mode.")
        else:
            os.makedirs(user_path)
            print(f"📁 Created folder: {user_path}")
    elif choice == "2":
        if not os.path.exists(user_path):
            print(f"❌ No folder found for {user_id}. Please register as new.")
            return None
        else:
            print(f"📂 Welcome back! Using folder: {user_path}")

    return user_path

# Forecast functions
def view_forecast(sku, forecast_df, user_folder):
    sku_forecast = forecast_df[forecast_df["sku"] == sku]
    if sku_forecast.empty:
        return f"❌ No forecast found for {sku}"

    model_used = sku_forecast["source_model"].iloc[0]
    model_info = f"\n Forecast generated using: {model_used}"

    return sku_forecast[["date", "forecast"]].to_string(index=False) + model_info

def compare_skus(sku1, sku2, forecast_df):
    df1 = forecast_df[forecast_df["sku"] == sku1][["date", "forecast"]].rename(columns={"forecast": f"forecast_{sku1}"})
    df2 = forecast_df[forecast_df["sku"] == sku2][["date", "forecast"]].rename(columns={"forecast": f"forecast_{sku2}"})
    if df1.empty or df2.empty:
        return "❌ One or both SKUs not found"
    merged = df1.merge(df2, on="date")
    return merged.to_string(index=False)

def top_skus(forecast_df, n=5):
    avg_forecast = forecast_df.groupby("sku")["forecast"].mean().sort_values(ascending=False)
    return avg_forecast.head(n).to_string()

def extract_skus(text):
    return re.findall(r"(?:S\d{3}_)?P\d{4}", text.upper())

def handle_nlp_query(query, forecast_df, mae_df, user_folder):
    query = query.lower()
    skus = extract_skus(query)

    if "forecast" in query and skus:
        return view_forecast(skus[0], forecast_df, user_folder)
    elif "compare" in query and len(skus) >= 2:
        return compare_skus(skus[0], skus[1], forecast_df)
    elif "top" in query or "best" in query:
        return top_skus(forecast_df)
    elif "breakdown" in query or "model usage" in query:
        return model_breakdown(forecast_df)
    elif "how much" in query and skus:
        return view_forecast(skus[0], forecast_df, user_folder)
    elif "simulate" in query or "what if" in query:
        sku = skus[0] if skus else None
        price_match = re.search(r"(?:r)?(\d{2,4}(?:\.\d{1,2})?)", query)
        discount_match = re.search(r"(\d{1,2})\s*%|\s*discount\s*(\d{1,2})", query)
        inventory_match = re.search(r"(?:inventory|stock)\s*(\d{2,4})", query)
        holiday_flag = int("holiday" in query)

        if not sku or not price_match:
            return "❌ Please specify both SKU and price for simulation."

        price = float(price_match.group(1))
        discount = float(discount_match.group(1) or discount_match.group(2) or 0) / 100 if discount_match else 0.0
        inventory = int(inventory_match.group(1)) if inventory_match else 100

        cmd = (
            f"python src/tools/simulate_scenario.py "
            f"--sku {sku} --price {price} --discount {discount} "
            f"--inventory {inventory} --holiday {holiday_flag}"
        )
        os.system(cmd)
        return ""

    else:
        return " Sorry, I couldn't understand that. Try asking about a forecast, comparison, or model explanation."

def model_breakdown(forecast_df):
    if forecast_df.empty or "source_model" not in forecast_df.columns:
        return "❌ No forecast data available."

    counts = forecast_df.groupby("source_model")["sku"].nunique()
    total = counts.sum()

    breakdown = "\n📊 Model Type Breakdown:\n"
    for model, count in counts.items():
        pct = round((count / total) * 100, 2)
        breakdown += f" - {model}: {count} SKUs ({pct}%)\n"

    return breakdown


def run_forecast(user_folder):
    print("\n🚀 Running forecast for:", user_folder)
    pipeline_path = os.path.join("src/main_pipeline.py")
    result = subprocess.run(["python", pipeline_path, user_folder])

    # === Confirm success ===
    print("\n Forecast completed.")
    print("📁 Results saved to:")
    print(f"   - Forecasts: {user_folder}/reports/sku_forecast.csv")
    print(f"   - Summary:  {user_folder}/reports/forecast_summary.json\n")
    input(" Press Enter to return to the main menu...")



# Main chatbot loop
def main():
    user_folder = get_user_folder()
    if not user_folder:
        return

    forecast_df = load_forecast(user_folder)
    mae_df = load_mae(user_folder)

    print("\n Hello, Welcome to DemandBot — Your Forecast Assistant")

    new_user_mode = forecast_df.empty

    while True:
        print("\nChoose an option:")

        if new_user_mode:
            print("1. Upload your own sales CSV and run forecast")
            print("2. Exit")
        else:
            print("1. View forecast for a SKU")
            print("2. Compare forecasts between SKUs")
            print("3. Show top SKUs by forecast")
            print("4. Ask a custom question (NLP)")
            print("5. Upload your own sales CSV and run forecast")
            print("6. Simulate a demand scenario")
            print("7. Exit")

        choice = input(" Your choice: ").strip()

        if new_user_mode:
            if choice == "1":
                path = input("📁 Enter path to your CSV file: ").strip()
                if os.path.exists(path):
                    os.makedirs(user_folder, exist_ok=True)
                    os.makedirs(os.path.join(user_folder, "data", "raw"), exist_ok=True)
                    os.makedirs(os.path.join(user_folder, "data", "processed"), exist_ok=True)
                    os.makedirs(os.path.join(user_folder, "reports"), exist_ok=True)
                    os.makedirs(os.path.join(user_folder, "models", "xgb"), exist_ok=True)

                    dest = os.path.join(user_folder, "data", "raw", "retail_store_inventory.csv")
                    os.replace(path, dest)
                    print(f" File saved to: {dest}")
                    
                    exit_code = os.system(f"python src/utils/preprocess_data.py {user_folder}")
                    if exit_code != 0:
                        print("❌ Preprocessing failed. Please check your file format.")
                        return

                    run_forecast(user_folder)
                    forecast_df = load_forecast(user_folder)
                    mae_df = load_mae(user_folder)

                    if not forecast_df.empty:
                        print("✅ Forecast ready. You can now ask questions.")
                        new_user_mode = False
                else:
                    print("❌ File not found. Please check the path.")
            elif choice == "2":
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Try again.")
        else:
            if choice == "1":
                sku = input("Enter SKU: ").strip().upper()
                print(view_forecast(sku, forecast_df, user_folder))
            elif choice == "2":
                sku1 = input("Enter first SKU: ").strip().upper()
                sku2 = input("Enter second SKU: ").strip().upper()
                print(compare_skus(sku1, sku2, forecast_df))
            elif choice == "3":
                print(top_skus(forecast_df))
            elif choice == "4":
                query = input(" Ask your question: ")
                print(handle_nlp_query(query, forecast_df, mae_df, user_folder))
            elif choice == "5":
                path = input("📁 Enter path to your CSV file: ").strip()
                if os.path.exists(path):
                    os.makedirs(user_folder, exist_ok=True)
                    os.makedirs(os.path.join(user_folder, "data", "raw"), exist_ok=True)

                    dest = os.path.join(user_folder, "data", "raw", "retail_store_inventory.csv")
                    os.replace(path, dest)
                    print(f" File saved to: {dest}")

                    run_forecast(user_folder)
                    forecast_df = load_forecast(user_folder)
                    mae_df = load_mae(user_folder)

                    if not forecast_df.empty:
                        print("✅ Forecast updated.")
                    else:
                        print("⚠️ Forecast pipeline may have failed. Please check your data or logs.")
                else:
                    print("❌ File not found. Please check the path.")
            elif choice == "6":
                print(" Simulate a demand scenario")
                sku = input("Enter SKU (e.g. S001_P0001): ").strip().upper()
                price = input("Enter price (e.g. 99.99): ").strip()
                discount = input("Enter discount (e.g. 0.15 for 15%): ").strip()
                inventory = input("Enter inventory level (e.g. 120): ").strip()
                holiday = input("Is it a holiday? (1 for Yes, 0 for No): ").strip()

                try:
                    price = float(price)
                    discount = float(discount)
                    inventory = int(inventory)
                    holiday = int(holiday)

                    cmd = (
                        f"python src/tools/simulate_scenario.py "
                        f"--sku {sku} --price {price} --discount {discount} "
                        f"--inventory {inventory} --holiday {holiday} " 
                        f"--user_folder {user_folder}"
                    )

                    os.system(cmd)
                except Exception as e:
                    print("❌ Invalid input. Please enter correct values.")
            elif choice == "7":
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Try again.")

print("🚀 Starting chatbot...")
if __name__ == "__main__":
    main()
