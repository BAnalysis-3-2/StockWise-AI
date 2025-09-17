"""
Robust data loader for retail_store_inventory.csv / TSV files.

- auto-detects delimiter (comma, tab, semicolon) using engine='python' and sep=None
- strips BOM and whitespace from column names
- auto-detects a date-like column if needed and parses it
"""
import pandas as pd
from typing import Optional

def detect_date_column(columns):
    candidates = [c for c in columns if c.strip().lower() in ('date','date_time','datetime','timestamp','time')]
    if candidates:
        return candidates[0]
    candidates = [c for c in columns if 'date' in c.lower() or 'time' in c.lower()]
    if candidates:
        return candidates[0]
    return None

def load_sales_csv(path: str, date_col: Optional[str] = "Date") -> pd.DataFrame:
    """
    Load CSV/TSV file and return DataFrame with column 'date' parsed as datetime.

    Args:
        path: file path to the data file
        date_col: preferred date column name (default "Date")

    Returns:
        pd.DataFrame with a column renamed to 'date'
    """
    # read only header first to inspect names, using Python engine auto-detect
    try:
        tmp = pd.read_csv(path, nrows=0, sep=None, engine='python')
    except Exception as e:
        # fallback: try tab separator explicitly
        tmp = pd.read_csv(path, nrows=0, sep='\t', engine='python')

    # clean column names
    cleaned_cols = [str(c).strip().replace('\ufeff', '') for c in tmp.columns]
    tmp.columns = cleaned_cols

    # decide which column to parse as date
    chosen = None
    if date_col and date_col in cleaned_cols:
        chosen = date_col
    else:
        # case-insensitive match
        lower_map = {c.lower(): c for c in cleaned_cols}
        if date_col and date_col.lower() in lower_map:
            chosen = lower_map[date_col.lower()]

    if chosen is None:
        detected = detect_date_column(cleaned_cols)
        if detected:
            chosen = detected

    if chosen is None:
        raise ValueError(
            "Could not detect a date column in the file. "
            "Found columns: " + str(cleaned_cols) +
            ". Please ensure there is a date column or pass its name to load_sales_csv."
        )

    # now read full file parsing the chosen date column, allowing auto delimiter detection
    try:
        df = pd.read_csv(path, parse_dates=[chosen], sep=None, engine='python')
    except Exception:
        # fallback to tab if auto-detect fails
        df = pd.read_csv(path, parse_dates=[chosen], sep='\t', engine='python')

    # clean column names again
    df.columns = [str(c).strip().replace('\ufeff','') for c in df.columns]

    # rename chosen date column to 'date'
    if chosen in df.columns:
        df = df.rename(columns={chosen: 'date'})
    else:
        lowers = {c.lower(): c for c in df.columns}
        if chosen.lower() in lowers:
            df = df.rename(columns={lowers[chosen.lower()]: 'date'})

    return df
