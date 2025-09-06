"""
preprocess.py
-------------
Data loading and preprocessing pipeline for the AI Inventory Forecast project.
Location-independent, uses data_loader for cleaning, robust to column name differences,
includes file-existence checks, and auto-saves processed output.
"""

from pathlib import Path
import pandas as pd
import sys

# Import the cleaned loaders from data_loader.py
from .data_loader import load_inventory as dl_inventory, load_transactions as dl_transactions

# -------------------------------------------------------------------
# Resolve project root and data paths
# -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)  # ensure folder exists

# -------------------------------------------------------------------
# Utility: check file existence
# -------------------------------------------------------------------
def _check_file_exists(file_path: Path):
    if not file_path.exists():
        print(f"❌ ERROR: Required file not found: {file_path}")
        print("💡 Make sure the file exists in data/raw/ or update the filename in your call to prepare_data().")
        sys.exit(1)

# -------------------------------------------------------------------
# Transformation helpers
# -------------------------------------------------------------------
def _standardize_product_column(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """
    Detect and rename the product column to 'Product_Item'.
    """
    possible_names = [
        "Product_Item", "Product_Name", "Product", "Item", "StockCode",
        "Description", "ProductName", "Product Code", "SKU", "Product_ID", "ItemCode"
    ]
    for col in possible_names:
        if col in df.columns:
            if col != "Product_Item":
                df = df.rename(columns={col: "Product_Item"})
            return df
    print(f"❌ ERROR: No product column found in {source} file. Checked: {possible_names}")
    print(f"📋 Columns found: {list(df.columns)}")
    sys.exit(1)


def explode_transactions(tx_df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure transactions have one row per product.
    If products are stored as comma-separated strings, split into multiple rows.
    """
    if tx_df["Product_Item"].dtype == object and tx_df["Product_Item"].str.contains(",").any():
        tx_df = tx_df.assign(Product_Item=tx_df["Product_Item"].str.split(","))
        tx_df = tx_df.explode("Product_Item")
        tx_df["Product_Item"] = tx_df["Product_Item"].str.strip()
    return tx_df


def merge_with_inventory(tx_df: pd.DataFrame, inv_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge exploded transactions with inventory master data on 'Product_Item'.
    """
    return pd.merge(tx_df, inv_df, how="left", on="Product_Item")

# -------------------------------------------------------------------
# Main pipeline
# -------------------------------------------------------------------
def prepare_data(
    inventory_file: str = "Inventory_and_Sales_Dataset.csv",
    transactions_file: str = "Retail_Transaction_Dataset.csv",
    save_output: bool = True
) -> pd.DataFrame:
    """
    Full preprocessing pipeline:
    1. Load inventory and transactions via data_loader (with cleaning)
    2. Standardize product column names
    3. Explode transactions to one row per product
    4. Merge with inventory master
    5. Standardize/compute demand_qty
    6. Clean and sort
    7. Optionally save processed dataset to data/processed/
    """
    # 1. Load cleaned data from data_loader
    inv_df = dl_inventory(inventory_file)
    tx_df = dl_transactions(transactions_file)

    # 2. Standardize product column names
    inv_df = _standardize_product_column(inv_df, "inventory")
    tx_df = _standardize_product_column(tx_df, "transactions")

    # 3. Explode transactions
    tx_df = explode_transactions(tx_df)

    # 4. Merge
    merged = merge_with_inventory(tx_df, inv_df)

    # 5. Standardize/compute demand_qty
    if "total_items" in merged.columns:
        merged = merged.rename(columns={"total_items": "demand_qty"})
    elif "Quantity" in merged.columns:
        merged = merged.rename(columns={"Quantity": "demand_qty"})
    elif "Total_Customer_Cost" in merged.columns and "Unit_Price" in merged.columns:
        merged["demand_qty"] = merged["Total_Customer_Cost"] / merged["Unit_Price"]
    else:
        merged["demand_qty"] = 1  # fallback

    # 6. Standardize date column
    if "Date" in merged.columns:
        merged = merged.rename(columns={"Date": "date"})
    merged["date"] = pd.to_datetime(merged["date"], errors="coerce")

    # Drop rows with no date or demand
    merged = merged.dropna(subset=["date", "demand_qty"])

    # Sort for consistency
    merged = merged.sort_values(["Product_Item", "date"]).reset_index(drop=True)

    # 7. Save output if requested
    if save_output:
        output_path = DATA_PROCESSED / "daily_demand_by_product.csv"
        merged.to_csv(output_path, index=False)
        print(f"💾 Processed dataset saved to: {output_path}")

    return merged
