# ═══════════════════════════════════════════════════════════════
# FILE: utils.py
# ═══════════════════════════════════════════════════════════════
"""
Logging bootstrap and date/time helpers.
"""

import logging
import sys
from datetime import datetime, timedelta
from typing import Optional
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


def breeze_date(dt: datetime) -> str:
    """Format a datetime into Breeze ISO-style: '2024-01-25T06:00:00.000Z'"""
    return dt.strftime("%Y-%m-%dT06:00:00.000Z")


def next_weekly_expiry(stock_code: str = "NIFTY") -> datetime:
    """Return the next Thursday (NIFTY weekly expiry) from today."""
    today = datetime.now()
    days_ahead = 3 - today.weekday()  # Thursday = 3
    if days_ahead < 0 or (days_ahead == 0 and today.hour >= 15):
        days_ahead += 7
    return today + timedelta(days=days_ahead)


def next_monthly_expiry() -> datetime:
    """Return the last Thursday of the current month."""
    today = datetime.now()
    import calendar
    last_day = calendar.monthrange(today.year, today.month)[1]
    dt = datetime(today.year, today.month, last_day)
    while dt.weekday() != 3:
        dt -= timedelta(days=1)
    if dt.date() < today.date():
        if today.month == 12:
            next_month = datetime(today.year + 1, 1, 1)
        else:
            next_month = datetime(today.year, today.month + 1, 1)
        last_day2 = calendar.monthrange(next_month.year, next_month.month)[1]
        dt = datetime(next_month.year, next_month.month, last_day2)
        while dt.weekday() != 3:
            dt -= timedelta(days=1)
    return dt


def atm_strike(spot_price: float, gap: int) -> float:
    """Round to the nearest option strike gap."""
    return round(spot_price / gap) * gap


def is_market_hours() -> bool:
    """Check if current time is within NSE market hours (9:15 – 15:30 IST)."""
    now = datetime.now()
    market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    return market_open <= now <= market_close


def safe_float(val, default: float = 0.0) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def safe_int(val, default: int = 0) -> int:
    try:
        return int(val)
    except (TypeError, ValueError):
        return default
