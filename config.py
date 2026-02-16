# ═══════════════════════════════════════════════════════════════
# FILE: config.py
# ═══════════════════════════════════════════════════════════════
"""
Central configuration loaded from environment variables.
Every tunable constant lives here.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")


class Config:
    # ── Breeze credentials ──────────────────────────────────
    API_KEY: str = os.getenv("BREEZE_API_KEY", "")
    API_SECRET: str = os.getenv("BREEZE_API_SECRET", "")
    SESSION_TOKEN: str = os.getenv("BREEZE_SESSION_TOKEN", "")
    TRADING_MODE: str = os.getenv("TRADING_MODE", "mock")  # 'mock' | 'live'

    # ── Market parameters ───────────────────────────────────
    RISK_FREE_RATE: float = float(os.getenv("RISK_FREE_RATE", "0.07"))
    DEFAULT_STOCK: str = os.getenv("DEFAULT_STOCK", "NIFTY")

    LOT_SIZES: dict = {
        "NIFTY": int(os.getenv("NIFTY_LOT_SIZE", "65")),
        "CNXBAN": int(os.getenv("BANKNIFTY_LOT_SIZE", "15")),
        "BANKNIFTY": int(os.getenv("BANKNIFTY_LOT_SIZE", "15")),
    }

    STRIKE_GAPS: dict = {
        "NIFTY": 50,
        "CNXBAN": 100,
        "BANKNIFTY": 100,
    }

    # ── Risk limits ─────────────────────────────────────────
    MAX_LOSS_PER_STRATEGY: float = float(os.getenv("MAX_LOSS_PER_STRATEGY", "-5000"))
    GLOBAL_MAX_LOSS: float = float(os.getenv("GLOBAL_MAX_LOSS", "-15000"))
    SL_PERCENTAGE: float = float(os.getenv("SL_PERCENTAGE", "50"))

    # ── Execution ───────────────────────────────────────────
    CHASE_TIMEOUT: float = float(os.getenv("CHASE_TIMEOUT_SECONDS", "3"))
    CHASE_MAX_RETRIES: int = int(os.getenv("CHASE_MAX_RETRIES", "5"))

    # ── Rate limiting ───────────────────────────────────────
    API_RATE_LIMIT: int = int(os.getenv("API_RATE_LIMIT", "90"))

    # ── Persistence ─────────────────────────────────────────
    DB_PATH: str = os.getenv("DB_PATH", "trading.db")
    SESSION_CACHE_FILE: str = ".breeze_session_cache"

    # ── Monitoring ──────────────────────────────────────────
    MONITOR_INTERVAL: float = float(os.getenv("MONITOR_INTERVAL", "0.5"))
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # ── Latency circuit breaker ─────────────────────────────
    MAX_API_LATENCY_MS: int = int(os.getenv("MAX_API_LATENCY_MS", "2000"))
    MAX_SLIPPAGE_PCT: float = float(os.getenv("MAX_SLIPPAGE_PCT", "5.0"))

    @classmethod
    def lot_size(cls, stock_code: str) -> int:
        return cls.LOT_SIZES.get(stock_code.upper(), 25)

    @classmethod
    def strike_gap(cls, stock_code: str) -> int:
        return cls.STRIKE_GAPS.get(stock_code.upper(), 50)
