import pandas as pd
import os

# === Paths ===
CLEANED_PATH = "data/processed/training_data.csv"
FEATURES_PATH = "data/processed/training_features.csv"

# === Load cleaned data ===
df = pd.read_csv(CLEANED_PATH)

# === Feature engineering ===
df["day_of_week"] = pd.to_datetime(df["date"]).dt.dayofweek
df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

# Lag features
df["lag_1"] = df["demand_forecast"].shift(1)
df["rolling_mean_3"] = df["demand_forecast"].rolling(window=3).mean()

# Drop rows with NaNs from lag/rolling
df = df.dropna()

# === Select features ===
feature_cols = [
    "inventory_level", "units_sold", "units_ordered", "price", "discount",
    "competitor_pricing", "day_of_week", "is_weekend", "lag_1", "rolling_mean_3"
]
X = df[feature_cols]

# === Save features ===
X.to_csv(FEATURES_PATH, index=False)
print(f"✅ Training features saved to: {FEATURES_PATH}")
