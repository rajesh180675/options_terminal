"""
config.py — Centralised configuration loaded from .env and sensible defaults.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class InstrumentSpec:
    stock_code: str
    exchange_code: str
    lot_size: int
    tick_size: float
    strike_gap: int


INSTRUMENTS: dict[str, InstrumentSpec] = {
    "NIFTY": InstrumentSpec("NIFTY", "NFO", 65, 0.05, 50),
    "BANKNIFTY": InstrumentSpec("CNXBAN", "NFO", 15, 0.05, 100),
    "FINNIFTY": InstrumentSpec("NIFFIN", "NFO", 25, 0.05, 50),
    "SENSEX": InstrumentSpec("BSESEN", "BFO", 20, 0.05, 100),
}


@dataclass
class AppConfig:
    # --- Breeze credentials ---
    api_key: str = os.getenv("BREEZE_API_KEY", "")
    api_secret: str = os.getenv("BREEZE_API_SECRET", "")
    session_token: str = os.getenv("BREEZE_SESSION_TOKEN", "")
    totp_secret: str = os.getenv("BREEZE_TOTP_SECRET", "")

    # --- Runtime ---
    trading_mode: str = os.getenv("TRADING_MODE", "mock")  # "mock" or "live"
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    db_path: str = os.getenv("DB_PATH", "trading.db")

    # --- Greeks ---
    risk_free_rate: float = float(os.getenv("RISK_FREE_RATE", "0.07"))

    # --- Risk ---
    default_sl_multiplier: float = float(os.getenv("DEFAULT_SL_MULTIPLIER", "2.0"))
    max_loss_per_strategy: float = float(os.getenv("MAX_LOSS_PER_STRATEGY", "50000"))
    global_max_loss: float = float(os.getenv("GLOBAL_MAX_LOSS", "200000"))

    # --- Execution ---
    chase_timeout_seconds: float = float(os.getenv("CHASE_TIMEOUT_SECONDS", "3"))
    chase_max_retries: int = int(os.getenv("CHASE_MAX_RETRIES", "5"))

    # --- Rate limiting ---
    api_rate_limit: int = 100
    api_rate_period: int = 60  # seconds

    # --- Session cache ---
    session_cache_path: str = ".session_cache"

    def get_instrument(self, name: str) -> InstrumentSpec:
        name_upper = name.upper().replace(" ", "")
        if name_upper in INSTRUMENTS:
            return INSTRUMENTS[name_upper]
        raise ValueError(f"Unknown instrument: {name}. Available: {list(INSTRUMENTS)}")


CFG = AppConfig()
