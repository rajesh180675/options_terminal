# ═══════════════════════════════════════════════════════════════
# FILE: app_config.py
# ═══════════════════════════════════════════════════════════════
"""
RENAMED from config.py to avoid collision with breeze_connect's
internal config module which defines SECURITY_MASTER_URL.

Three-tier config resolution:
  1. st.secrets   (Streamlit Cloud)
  2. .env / OS    (Local dev)
  3. Hardcoded defaults
"""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    _env = Path(__file__).parent / ".env"
    if _env.exists():
        load_dotenv(_env, override=True)
except ImportError:
    pass


def _get(key: str, default: str = "") -> str:
    # Tier 1: Streamlit secrets
    try:
        import streamlit as st
        if hasattr(st, "secrets") and key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    # Tier 2: OS env / .env
    val = os.environ.get(key, "")
    if val:
        return val
    # Tier 3: default
    return default


# ── Instrument definitions ───────────────────────────────────
# Each instrument has: breeze_code, exchange, lot_size, strike_gap,
# expiry_weekday (0=Mon, 1=Tue, 2=Wed, 3=Thu, 4=Fri)

INSTRUMENTS = {
    "NIFTY": {
        "breeze_code": "NIFTY",
        "display": "NIFTY 50",
        "exchange": "NFO",
        "lot_size": int(_get("NIFTY_LOT_SIZE", "65")),
        "strike_gap": 50,
        "expiry_weekday": 1,   # Tuesday
    },
    "BANKNIFTY": {
        "breeze_code": "CNXBAN",
        "display": "Bank NIFTY",
        "exchange": "NFO",
        "lot_size": int(_get("BANKNIFTY_LOT_SIZE", "15")),
        "strike_gap": 100,
        "expiry_weekday": 2,   # Wednesday
    },
    "SENSEX": {
        "breeze_code": "BSESEN",
        "display": "SENSEX",
        "exchange": "BFO",
        "lot_size": int(_get("SENSEX_LOT_SIZE", "20")),
        "strike_gap": 100,
        "expiry_weekday": 3,   # Thursday
    },
}


class Config:
    # ── Credentials ─────────────────────────────────────────
    API_KEY: str = _get("BREEZE_API_KEY", "")
    API_SECRET: str = _get("BREEZE_API_SECRET", "")
    SESSION_TOKEN: str = _get("BREEZE_SESSION_TOKEN", "")
    TRADING_MODE: str = _get("TRADING_MODE", "mock").strip().lower()

    # ── Market ──────────────────────────────────────────────
    RISK_FREE_RATE: float = float(_get("RISK_FREE_RATE", "0.07"))
    DEFAULT_STOCK: str = _get("DEFAULT_STOCK", "NIFTY")

    # ── Risk ────────────────────────────────────────────────
    MAX_LOSS_PER_STRATEGY: float = float(_get("MAX_LOSS_PER_STRATEGY", "-5000"))
    GLOBAL_MAX_LOSS: float = float(_get("GLOBAL_MAX_LOSS", "-15000"))
    SL_PERCENTAGE: float = float(_get("SL_PERCENTAGE", "50"))

    # ── Trailing SL ─────────────────────────────────────────
    TRAIL_ENABLED: bool = _get("TRAIL_ENABLED", "true").lower() == "true"
    TRAIL_ACTIVATION_PCT: float = float(_get("TRAIL_ACTIVATION_PCT", "20"))
    TRAIL_SL_PCT: float = float(_get("TRAIL_SL_PCT", "40"))

    # ── Auto-exit ───────────────────────────────────────────
    AUTO_EXIT_ENABLED: bool = _get("AUTO_EXIT_ENABLED", "true").lower() == "true"
    AUTO_EXIT_HOUR: int = int(_get("AUTO_EXIT_HOUR", "15"))
    AUTO_EXIT_MINUTE: int = int(_get("AUTO_EXIT_MINUTE", "15"))

    # ── Execution ───────────────────────────────────────────
    CHASE_TIMEOUT: float = float(_get("CHASE_TIMEOUT_SECONDS", "3"))
    CHASE_MAX_RETRIES: int = int(_get("CHASE_MAX_RETRIES", "5"))
    API_RATE_LIMIT: int = int(_get("API_RATE_LIMIT", "85"))

    # ── Persistence ─────────────────────────────────────────
    DB_PATH: str = _get("DB_PATH", "trading.db")
    SESSION_CACHE: str = ".breeze_session_cache"

    # ── Monitor ─────────────────────────────────────────────
    MONITOR_INTERVAL: float = float(_get("MONITOR_INTERVAL", "0.5"))
    LOG_LEVEL: str = _get("LOG_LEVEL", "INFO")
    MAX_API_LATENCY_MS: int = int(_get("MAX_API_LATENCY_MS", "2000"))
    WS_HEARTBEAT_TIMEOUT: float = float(_get("WS_HEARTBEAT_TIMEOUT", "30"))

    # ── Helpers ─────────────────────────────────────────────

    @classmethod
    def is_live(cls) -> bool:
        return cls.TRADING_MODE == "live"

    @classmethod
    def instrument(cls, user_name: str) -> dict:
        """Get instrument config by user-friendly name or breeze code."""
        key = user_name.upper()
        if key in INSTRUMENTS:
            return INSTRUMENTS[key]
        for k, v in INSTRUMENTS.items():
            if v["breeze_code"] == key:
                return v
        return INSTRUMENTS.get("NIFTY")

    @classmethod
    def breeze_code(cls, user_name: str) -> str:
        return cls.instrument(user_name)["breeze_code"]

    @classmethod
    def exchange(cls, user_name: str) -> str:
        return cls.instrument(user_name)["exchange"]

    @classmethod
    def lot_size(cls, code: str) -> int:
        return cls.instrument(code)["lot_size"]

    @classmethod
    def strike_gap(cls, code: str) -> int:
        return cls.instrument(code)["strike_gap"]

    @classmethod
    def expiry_weekday(cls, code: str) -> int:
        return cls.instrument(code)["expiry_weekday"]

    @classmethod
    def validate(cls) -> list:
        errs = []
        if cls.is_live():
            if not cls.API_KEY:
                errs.append("BREEZE_API_KEY missing")
            if not cls.API_SECRET:
                errs.append("BREEZE_API_SECRET missing")
            if not cls.SESSION_TOKEN:
                errs.append("BREEZE_SESSION_TOKEN missing")
        return errs

    @classmethod
    def dump(cls) -> dict:
        return {
            "TRADING_MODE": cls.TRADING_MODE,
            "DEFAULT_STOCK": cls.DEFAULT_STOCK,
            "API_KEY_SET": bool(cls.API_KEY),
            "SECRET_SET": bool(cls.API_SECRET),
            "TOKEN_SET": bool(cls.SESSION_TOKEN),
            "SL_PERCENTAGE": cls.SL_PERCENTAGE,
            "TRAIL_ENABLED": cls.TRAIL_ENABLED,
            "TRAIL_ACTIVATION_PCT": cls.TRAIL_ACTIVATION_PCT,
            "AUTO_EXIT_ENABLED": cls.AUTO_EXIT_ENABLED,
            "GLOBAL_MAX_LOSS": cls.GLOBAL_MAX_LOSS,
            "INSTRUMENTS": {k: f"{v['breeze_code']}@{v['exchange']}"
                           for k, v in INSTRUMENTS.items()},
        }
