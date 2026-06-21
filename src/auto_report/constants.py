"""Shared constants for the auto report pipeline."""

KEY_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]
OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]

DEFAULT_LEAKAGE_COLUMNS = [
    "day_high",
    "day_low",
    "day_close",
    "day_volume",
    "day_pivot",
]

DEFAULT_CLASS_ORDER = ["No - None", "No - Sideway", "Yes - Buy", "Yes - Sell"]
MISSING_MARKERS = {"", "none", "null", "nan", "na", "n/a"}
