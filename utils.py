# ═══════════════════════════════════════════════════════════════
# FILE: utils.py
# ═══════════════════════════════════════════════════════════════
"""
Logging, date helpers, and Breeze format utilities.
Deep-studied against Breeze SDK source.
"""

import logging
import sys
import calendar
from datetime import datetime, timedelta
from config import Config


def setup_logging() -> logging.Logger:
    logger = logging.getLogger("OptionsTerminal")
    if logger.handlers:
        return logger
    logger.setLevel(getattr(logging, Config.LOG_LEVEL.upper(), logging.INFO))
    fmt = logging.Formatter(
        "[%(asctime)s.%(msecs)03d] %(levelname)-5s │ %(name)s │ %(message)s",
        datefmt="%H:%M:%S",
    )
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


LOG = setup_logging()


def breeze_expiry_format(dt: datetime) -> str:
    """
    Breeze SDK expects expiry_date in ISO format:
      '2024-01-25T06:00:00.000Z'
    
    CRITICAL: Breeze uses this exact format for:
      - subscribe_feeds(expiry_date=...)
      - get_option_chain_quotes(expiry_date=...)
      - place_order(expiry_date=...)
    
    The T06:00:00.000Z suffix is mandatory for NFO.
    """
    return dt.strftime("%Y-%m-%dT06:00:00.000Z")


def breeze_strike_format(strike: float) -> str:
    """
    Breeze SDK expects strike_price as a string.
    For subscribe_feeds: "23500"
    For place_order: "23500"
    Always integer, no decimals.
    """
    return str(int(strike))


def next_weekly_expiry(stock_code: str = "NIFTY") -> datetime:
    """
    Return the next Thursday (NSE weekly expiry).
    If today IS Thursday and before 15:30, return today.
    """
    today = datetime.now()
    days_ahead = 3 - today.weekday()  # Thursday = 3
    if days_ahead < 0:
        days_ahead += 7
    elif days_ahead == 0 and today.hour >= 16:
        days_ahead = 7
    return (today + timedelta(days=days_ahead)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )


def next_monthly_expiry() -> datetime:
    """Return the last Thursday of the current month."""
    today = datetime.now()
    last_day = calendar.monthrange(today.year, today.month)[1]
    dt = datetime(today.year, today.month, last_day)
    while dt.weekday() != 3:
        dt -= timedelta(days=1)
    if dt.date() < today.date():
        if today.month == 12:
            nm = datetime(today.year + 1, 1, 1)
        else:
            nm = datetime(today.year, today.month + 1, 1)
        last_day2 = calendar.monthrange(nm.year, nm.month)[1]
        dt = datetime(nm.year, nm.month, last_day2)
        while dt.weekday() != 3:
            dt -= timedelta(days=1)
    return dt


def atm_strike(spot_price: float, gap: int) -> float:
    return round(spot_price / gap) * gap


def is_market_hours() -> bool:
    now = datetime.now()
    market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    return market_open <= now <= market_close


def safe_float(val, default: float = 0.0) -> float:
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def safe_int(val, default: int = 0) -> int:
    if val is None:
        return default
    try:
        return int(float(val))
    except (TypeError, ValueError):
        return default


def parse_breeze_datetime(dt_str: str) -> datetime:
    """Parse various Breeze datetime formats."""
    if not dt_str:
        return datetime.now()
    for fmt in [
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%S",
        "%d-%b-%Y %H:%M:%S",
        "%Y-%m-%d",
        "%d-%b-%Y",
    ]:
        try:
            return datetime.strptime(dt_str.strip(), fmt)
        except ValueError:
            continue
    return datetime.now()
