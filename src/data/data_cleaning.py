"""
Cleaning and reindexing utilities:
- normalize column names to a standard set
- create 'sku' key (store_product)
- ensure continuous daily index per sku between min and max dates
- save processed splits (train/val/test)
"""
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict

DEFAULT_COL_MAP = {
    'Date':'date', 'Store ID':'store_id', 'Product ID':'product_id',
    'Category':'category', 'Region':'region',
    'Inventory Level':'inventory', 'Units Sold':'units_sold',
    'Units Ordered':'units_ordered', 'Demand Forecast':'demand_forecast',
    'Price':'price', 'Discount':'discount', 'Weather Condition':'weather',
    'Holiday/Promotion':'is_holiday', 'Competitor Pricing':'competitor_price',
    'Seasonality':'season'
}

def normalize_columns(df: pd.DataFrame, col_map: Dict = None) -> pd.DataFrame:
    if col_map is None:
        col_map = DEFAULT_COL_MAP
    # rename if key exists in df
    rename_map = {k:v for k,v in col_map.items() if k in df.columns}
    df = df.rename(columns=rename_map)
    return df

def create_sku_key(df: pd.DataFrame) -> pd.DataFrame:
    if 'store_id' in df.columns and 'product_id' in df.columns:
        df['sku'] = df['store_id'].astype(str) + '_' + df['product_id'].astype(str)
    elif 'product_id' in df.columns:
        df['sku'] = df['product_id'].astype(str)
    else:
        raise ValueError("CSV must contain at least 'Product ID' or both 'Store ID' and 'Product ID'.")
    return df

def coerce_numeric(df: pd.DataFrame, numeric_cols: List[str] = None) -> pd.DataFrame:
    if numeric_cols is None:
        numeric_cols = ['inventory','units_sold','units_ordered','price','discount','competitor_price']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
    if 'is_holiday' in df.columns:
        # convert to int flag
        df['is_holiday'] = df['is_holiday'].fillna(0).astype(int)
    return df

def reindex_by_sku(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure continuous daily rows for each sku between global min and max date.
    Missing units_sold are filled with 0 (assumption: no sales recorded => 0).
    Static columns (store_id, product_id, category, region, price, etc.) are forward/backward filled per sku.
    """
    df = df.copy()
    df = df.sort_values(['sku','date'])
    skus = df['sku'].unique()
    min_date = df['date'].min()
    max_date = df['date'].max()
    full_idx = pd.MultiIndex.from_product([skus, pd.date_range(min_date, max_date, freq='D')], names=['sku','date'])
    df = df.set_index(['sku','date']).reindex(full_idx).reset_index()
    static_cols = ['store_id','product_id','category','region','price','discount','competitor_price','season','weather']
    for c in static_cols:
        if c in df.columns:
            df[c] = df.groupby('sku')[c].ffill().bfill()
    if 'units_sold' in df.columns:
        df['units_sold'] = df['units_sold'].fillna(0)
    else:
        df['units_sold'] = 0
    # keep date as datetime
    df['date'] = pd.to_datetime(df['date'])
    return df

def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['dow'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['week'] = df['date'].dt.isocalendar().week.astype(int)
    return df

def split_time_series(df: pd.DataFrame, val_days: int = 28, test_days: int = 28) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split into train / val / test by date.
    - test: last test_days (most recent)
    - val: preceding val_days
    - train: all earlier data
    """
    max_date = df['date'].max()
    test_start = max_date - pd.Timedelta(days=test_days-1)
    val_start = test_start - pd.Timedelta(days=val_days)
    train = df[df['date'] < val_start].copy()
    val = df[(df['date'] >= val_start) & (df['date'] < test_start)].copy()
    test = df[df['date'] >= test_start].copy()
    return train, val, test

def save_processed_splits(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, out_dir: str):
    train.to_csv(f"{out_dir}/training_data.csv", index=False)
    val.to_csv(f"{out_dir}/validation_data.csv", index=False)
    test.to_csv(f"{out_dir}/test_data.csv", index=False)
