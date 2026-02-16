# ═══════════════════════════════════════════════════════════════
# FILE: utils.py
# ═══════════════════════════════════════════════════════════════
"""
Logging, expiry calculations, Breeze format helpers.
NIFTY = Tuesday, BANKNIFTY = Wednesday, SENSEX = Thursday.
"""

import logging
import sys
import calendar
from datetime import datetime, timedelta
from app_config import Config, INSTRUMENTS


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


def breeze_expiry(dt: datetime) -> str:
    """Breeze ISO format: '2025-02-18T06:00:00.000Z'"""
    return dt.strftime("%Y-%m-%dT06:00:00.000Z")


def breeze_strike(strike: float) -> str:
    return str(int(strike))


def next_weekly_expiry(instrument_key: str) -> datetime:
    """
    Return next expiry date for the given instrument.
    Uses the per-instrument expiry weekday from INSTRUMENTS config.
    """
    inst = Config.instrument(instrument_key)
    target_weekday = inst["expiry_weekday"]
    today = datetime.now()
    days_ahead = target_weekday - today.weekday()
    if days_ahead < 0:
        days_ahead += 7
    elif days_ahead == 0 and today.hour >= 16:
        days_ahead = 7
    return (today + timedelta(days=days_ahead)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )


def next_monthly_expiry(instrument_key: str) -> datetime:
    """Last occurrence of the instrument's expiry weekday in the current month."""
    inst = Config.instrument(instrument_key)
    target_weekday = inst["expiry_weekday"]
    today = datetime.now()
    last_day = calendar.monthrange(today.year, today.month)[1]
    dt = datetime(today.year, today.month, last_day)
    while dt.weekday() != target_weekday:
        dt -= timedelta(days=1)
    if dt.date() < today.date():
        if today.month == 12:
            nm = datetime(today.year + 1, 1, 1)
        else:
            nm = datetime(today.year, today.month + 1, 1)
        last_day2 = calendar.monthrange(nm.year, nm.month)[1]
        dt = datetime(nm.year, nm.month, last_day2)
        while dt.weekday() != target_weekday:
            dt -= timedelta(days=1)
    return dt


def atm_strike(spot: float, gap: int) -> float:
    return round(spot / gap) * gap


def is_market_hours() -> bool:
    now = datetime.now()
    return now.replace(hour=9, minute=15, second=0) <= now <= now.replace(hour=15, minute=30, second=0)


def is_auto_exit_time() -> bool:
    if not Config.AUTO_EXIT_ENABLED:
        return False
    now = datetime.now()
    exit_time = now.replace(
        hour=Config.AUTO_EXIT_HOUR,
        minute=Config.AUTO_EXIT_MINUTE,
        second=0, microsecond=0
    )
    return now >= exit_time and now.hour < 16


def is_expiry_day(instrument_key: str) -> bool:
    exp = next_weekly_expiry(instrument_key)
    return exp.date() == datetime.now().date()


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
