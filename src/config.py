# src/config.py

from pathlib import Path

# Data columns
DATE_COL = "date"
PRODUCT_KEY = "Product_Item"
TARGET = "demand_qty"

# Default parameters
DEFAULT_HORIZON = 14
DEFAULT_WINDOW = 7
DEFAULT_LEAD_DAYS = 7.0

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # one level up from src/
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FORECASTS_DIR = OUTPUTS_DIR / "forecasts"
REPORTS_DIR = OUTPUTS_DIR / "reports"
