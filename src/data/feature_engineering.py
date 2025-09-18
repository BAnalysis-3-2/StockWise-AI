"""
Feature engineering utilities:
- create lag and rolling features grouped by SKU
- label-encode selected categorical columns and return encoders
- return final feature list for model training
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from typing import List, Tuple, Dict

def create_lag_roll_features(df: pd.DataFrame,
                             group_col: str = 'sku',
                             target_col: str = 'units_sold',
                             lags: List[int] = [1,7,14],
                             windows: List[int] = [7,28]) -> pd.DataFrame:
    df = df.sort_values([group_col, 'date']).copy()
    # create lags
    for lag in lags:
        df[f'lag_{lag}'] = df.groupby(group_col)[target_col].shift(lag)
    # rolling stats (shifted by 1 to avoid leaking current day's sales)
    for w in windows:
        df[f'roll_mean_{w}'] = df.groupby(group_col)[target_col].shift(1).rolling(window=w).mean().reset_index(level=0, drop=True)
        df[f'roll_std_{w}']  = df.groupby(group_col)[target_col].shift(1).rolling(window=w).std().reset_index(level=0, drop=True)
    # days since last sale: number of days since last positive sale
    def days_since_last(x):
        last = -999
        out = []
        for i, v in enumerate(x.values):
            if v > 0:
                last = i
                out.append(0)
            else:
                out.append(i - last if last >= 0 else 999)
        return pd.Series(out, index=x.index)
    df['days_since_sale'] = df.groupby(group_col)[target_col].apply(days_since_last).reset_index(level=0, drop=True)
    return df

def label_encode_columns(df: pd.DataFrame, cols: List[str]) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    encoders = {}
    for c in cols:
        if c in df.columns:
            df[c] = df[c].fillna('NA').astype(str)
            le = LabelEncoder()
            df[c + '_enc'] = le.fit_transform(df[c])
            encoders[c] = le
    return df, encoders

def get_default_feature_list() -> List[str]:
    """
    Default features used by LightGBM baseline.
    You can extend this list later.
    """
    feats = [
        'dow','month','day','week',
        'price','discount','competitor_price','inventory',
        'lag_1','lag_7','lag_14',
        'roll_mean_7','roll_std_7','roll_mean_28','roll_std_28',
        'days_since_sale',
        'category_enc','region_enc','weather_enc','season_enc',
        'is_holiday'
    ]
    return feats
