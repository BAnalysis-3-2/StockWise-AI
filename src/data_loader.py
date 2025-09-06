# src/data_loader.py
import pandas as pd
import numpy as np
from .config import RAW_DATA_DIR
from .utils import normalize_name, clean_money

def load_inventory(filename="Inventory_and_Sales_Dataset.csv"):
    path = RAW_DATA_DIR / filename
    df = pd.read_csv(path)
    if "Unit_Price" in df.columns:
        df["Unit_Price"] = df["Unit_Price"].apply(clean_money)
    if "Product_Name" in df.columns:
        df["norm_name"] = df["Product_Name"].apply(normalize_name)
    return df

def load_transactions(filename="Retail_Transaction_Dataset.csv"):
    path = RAW_DATA_DIR / filename
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", infer_datetime_format=True)
    return df
