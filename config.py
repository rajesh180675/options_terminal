# ═══════════════════════════════════════════════════════════════
# FILE: config.py
# ═══════════════════════════════════════════════════════════════
"""
Unified configuration loader.
Priority: st.secrets (Streamlit Cloud) → .env file → OS environment → defaults.
This fixes the #1 critical bug where TRADING_MODE never resolved to 'live'.
"""

import os
import sys
from pathlib import Path

# ── Load .env FIRST (before any other import) ───────────────
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).parent / ".env"
    if _env_path.exists():
        load_dotenv(_env_path, override=True)
except ImportError:
    pass


def _get(key: str, default: str = "") -> str:
    """
    Three-tier config resolution:
      1. Streamlit secrets (st.secrets) — for cloud deployment
      2. Environment variable (loaded from .env or OS)
      3. Hardcoded default
    """
    # Tier 1: Streamlit secrets
    try:
        import streamlit as st
        if hasattr(st, "secrets") and key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass

    # Tier 2: OS / .env environment
    val = os.environ.get(key, "")
    if val:
        return val

    # Tier 3: default
    return default


class Config:
    """Immutable-ish config singleton. All values resolved at import time."""

    # ── Breeze credentials ──────────────────────────────────
    API_KEY: str = _get("BREEZE_API_KEY", "")
    API_SECRET: str = _get("BREEZE_API_SECRET", "")
    SESSION_TOKEN: str = _get("BREEZE_SESSION_TOKEN", "")
    TRADING_MODE: str = _get("TRADING_MODE", "mock").strip().lower()

    # ── Market parameters ───────────────────────────────────
    RISK_FREE_RATE: float = float(_get("RISK_FREE_RATE", "0.07"))
    DEFAULT_STOCK: str = _get("DEFAULT_STOCK", "NIFTY")

    # Breeze SDK uses "NIFTY" for Nifty 50 and "CNXBAN" for Bank Nifty
    # in NFO segment. We map user-friendly names to Breeze codes.
    STOCK_CODE_MAP: dict = {
        "NIFTY": "NIFTY",
        "BANKNIFTY": "CNXBAN",
        "CNXBAN": "CNXBAN",
    }

    LOT_SIZES: dict = {
        "NIFTY": int(_get("NIFTY_LOT_SIZE", "65")),
        "CNXBAN": int(_get("BANKNIFTY_LOT_SIZE", "15")),
    }

    STRIKE_GAPS: dict = {
        "NIFTY": 50,
        "CNXBAN": 100,
    }

    # ── Risk limits ─────────────────────────────────────────
    MAX_LOSS_PER_STRATEGY: float = float(_get("MAX_LOSS_PER_STRATEGY", "-5000"))
    GLOBAL_MAX_LOSS: float = float(_get("GLOBAL_MAX_LOSS", "-15000"))
    SL_PERCENTAGE: float = float(_get("SL_PERCENTAGE", "50"))

    # ── Execution ───────────────────────────────────────────
    CHASE_TIMEOUT: float = float(_get("CHASE_TIMEOUT_SECONDS", "3"))
    CHASE_MAX_RETRIES: int = int(_get("CHASE_MAX_RETRIES", "5"))

    # ── Rate limiting ───────────────────────────────────────
    # Breeze official limit: 100/min. We use 85 to leave headroom.
    API_RATE_LIMIT: int = int(_get("API_RATE_LIMIT", "85"))

    # ── Persistence ─────────────────────────────────────────
    DB_PATH: str = _get("DB_PATH", "trading.db")
    SESSION_CACHE_FILE: str = ".breeze_session_cache"

    # ── Monitoring ──────────────────────────────────────────
    MONITOR_INTERVAL: float = float(_get("MONITOR_INTERVAL", "0.5"))
    LOG_LEVEL: str = _get("LOG_LEVEL", "INFO")

    # ── Circuit breakers ────────────────────────────────────
    MAX_API_LATENCY_MS: int = int(_get("MAX_API_LATENCY_MS", "2000"))
    MAX_SLIPPAGE_PCT: float = float(_get("MAX_SLIPPAGE_PCT", "5.0"))
    WS_HEARTBEAT_TIMEOUT: float = float(_get("WS_HEARTBEAT_TIMEOUT", "30.0"))

    @classmethod
    def breeze_code(cls, user_stock: str) -> str:
        """Convert user-friendly name to Breeze NFO stock_code."""
        return cls.STOCK_CODE_MAP.get(user_stock.upper(), user_stock.upper())

    @classmethod
    def lot_size(cls, breeze_code: str) -> int:
        return cls.LOT_SIZES.get(breeze_code.upper(), 25)

    @classmethod
    def strike_gap(cls, breeze_code: str) -> int:
        return cls.STRIKE_GAPS.get(breeze_code.upper(), 50)

    @classmethod
    def is_live(cls) -> bool:
        return cls.TRADING_MODE == "live"

    @classmethod
    def validate(cls) -> list:
        """Return list of config errors. Empty = all good."""
        errors = []
        if cls.is_live():
            if not cls.API_KEY:
                errors.append("BREEZE_API_KEY is empty")
            if not cls.API_SECRET:
                errors.append("BREEZE_API_SECRET is empty")
            if not cls.SESSION_TOKEN:
                errors.append("BREEZE_SESSION_TOKEN is empty")
        if cls.GLOBAL_MAX_LOSS >= 0:
            errors.append("GLOBAL_MAX_LOSS must be negative")
        return errors

    @classmethod
    def dump(cls) -> dict:
        """Return non-sensitive config for display."""
        return {
            "TRADING_MODE": cls.TRADING_MODE,
            "DEFAULT_STOCK": cls.DEFAULT_STOCK,
            "API_KEY_SET": bool(cls.API_KEY),
            "SECRET_SET": bool(cls.API_SECRET),
            "TOKEN_SET": bool(cls.SESSION_TOKEN),
            "RISK_FREE_RATE": cls.RISK_FREE_RATE,
            "SL_PERCENTAGE": cls.SL_PERCENTAGE,
            "GLOBAL_MAX_LOSS": cls.GLOBAL_MAX_LOSS,
            "API_RATE_LIMIT": cls.API_RATE_LIMIT,
            "CHASE_TIMEOUT": cls.CHASE_TIMEOUT,
            "CHASE_MAX_RETRIES": cls.CHASE_MAX_RETRIES,
        }
