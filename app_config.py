# ═══════════════════════════════════════════════════════════════
# FILE: app_config.py
# ═══════════════════════════════════════════════════════════════
"""
RENAMED from config.py to avoid collision with breeze_connect's
internal config module (SECURITY_MASTER_URL crash).

Three-tier resolver: st.secrets → .env → defaults
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
    try:
        import streamlit as st
        if hasattr(st, "secrets") and key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return os.environ.get(key, "") or default

# Add to INSTRUMENTS, inside each instrument dict:
#   "cash_code": "<CODE>",    # for historical data (NSE/BSE cash segment)
#   "cash_exchange": "<EXC>",

INSTRUMENTS = {
    "NIFTY": {
        "breeze_code": "NIFTY", "display": "NIFTY 50", "exchange": "NFO",
        "lot_size": int(_get("NIFTY_LOT_SIZE", "65")),
        "strike_gap": 50, "expiry_weekday": 1,
        "cash_code": "NIFTY", "cash_exchange": "NSE",
    },
    "BANKNIFTY": {
        "breeze_code": "CNXBAN", "display": "Bank NIFTY", "exchange": "NFO",
        "lot_size": int(_get("BANKNIFTY_LOT_SIZE", "15")),
        "strike_gap": 100, "expiry_weekday": 2,
        "cash_code": "CNXBAN", "cash_exchange": "NSE",
    },
    "SENSEX": {
        "breeze_code": "BSESEN", "display": "SENSEX", "exchange": "BFO",
        "lot_size": int(_get("SENSEX_LOT_SIZE", "20")),
        "strike_gap": 100, "expiry_weekday": 3,
        "cash_code": "BSESEN", "cash_exchange": "BSE",
    },
}


class Config:
    API_KEY: str = _get("BREEZE_API_KEY")
    API_SECRET: str = _get("BREEZE_API_SECRET")
    SESSION_TOKEN: str = _get("BREEZE_SESSION_TOKEN")
    TRADING_MODE: str = _get("TRADING_MODE", "mock").strip().lower()
    RISK_FREE_RATE: float = float(_get("RISK_FREE_RATE", "0.07"))
    DEFAULT_STOCK: str = _get("DEFAULT_STOCK", "NIFTY")
    GLOBAL_MAX_LOSS: float = float(_get("GLOBAL_MAX_LOSS", "-15000"))
    SL_PERCENTAGE: float = float(_get("SL_PERCENTAGE", "50"))
    TRAIL_ENABLED: bool = _get("TRAIL_ENABLED", "true").lower() == "true"
    TRAIL_ACTIVATION_PCT: float = float(_get("TRAIL_ACTIVATION_PCT", "20"))
    TRAIL_SL_PCT: float = float(_get("TRAIL_SL_PCT", "40"))
    AUTO_EXIT_ENABLED: bool = _get("AUTO_EXIT_ENABLED", "true").lower() == "true"
    AUTO_EXIT_HOUR: int = int(_get("AUTO_EXIT_HOUR", "15"))
    AUTO_EXIT_MINUTE: int = int(_get("AUTO_EXIT_MINUTE", "15"))
    CHASE_TIMEOUT: float = float(_get("CHASE_TIMEOUT_SECONDS", "3"))
    CHASE_MAX_RETRIES: int = int(_get("CHASE_MAX_RETRIES", "5"))
    API_RATE_LIMIT: int = int(_get("API_RATE_LIMIT", "85"))
    DB_PATH: str = _get("DB_PATH", "trading.db")
    SESSION_CACHE: str = ".breeze_session_cache"
    MONITOR_INTERVAL: float = float(_get("MONITOR_INTERVAL", "0.5"))
    LOG_LEVEL: str = _get("LOG_LEVEL", "INFO")
    WS_HEARTBEAT_TIMEOUT: float = float(_get("WS_HEARTBEAT_TIMEOUT", "30"))
    ADJUSTMENT_THRESHOLD: float = float(_get("ADJUSTMENT_THRESHOLD", "30"))
    AUTO_ADJUST: bool = _get("AUTO_ADJUST", "false").lower() == "true"
    TELEGRAM_BOT_TOKEN: str = _get("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID: str = _get("TELEGRAM_CHAT_ID", "")
    TELEGRAM_MIN_LEVEL: str = _get("TELEGRAM_MIN_LEVEL", "warn")
    WEBHOOK_URL: str = _get("WEBHOOK_URL", "")
    BACKTEST_DEFAULT_DAYS: int = int(_get("BACKTEST_DEFAULT_DAYS", "180"))
    DELTA_DRIFT_THRESHOLD: float = float(_get("DELTA_DRIFT_THRESHOLD", "30"))
    BROKER_SYNC_INTERVAL: float = float(_get("BROKER_SYNC_INTERVAL", "30"))
    PENDING_POLL_INTERVAL: float = float(_get("PENDING_POLL_INTERVAL", "10"))
    MARGIN_BUFFER_PCT: float = float(_get("MARGIN_BUFFER_PCT", "10"))

    @classmethod
    def is_live(cls) -> bool:
        return cls.TRADING_MODE == "live"

    @classmethod
    def instrument(cls, name: str) -> dict:
        k = name.upper().strip()
        if k in INSTRUMENTS:
            return INSTRUMENTS[k]
        for v in INSTRUMENTS.values():
            if v["breeze_code"] == k:
                return v
        return INSTRUMENTS["NIFTY"]

    @classmethod
    def breeze_code(cls, name: str) -> str:
        return cls.instrument(name)["breeze_code"]

    @classmethod
    def exchange(cls, name: str) -> str:
        return cls.instrument(name)["exchange"]

    @classmethod
    def lot_size(cls, name: str) -> int:
        return cls.instrument(name)["lot_size"]

    @classmethod
    def strike_gap(cls, name: str) -> int:
        return cls.instrument(name)["strike_gap"]

    @classmethod
    def validate(cls) -> list:
        errs = []
        if cls.is_live():
            if not cls.API_KEY: errs.append("BREEZE_API_KEY missing")
            if not cls.API_SECRET: errs.append("BREEZE_API_SECRET missing")
            if not cls.SESSION_TOKEN: errs.append("BREEZE_SESSION_TOKEN missing")
        return errs

    @classmethod
    def dump(cls) -> dict:
        return {
            "MODE": cls.TRADING_MODE, "API_KEY_SET": bool(cls.API_KEY),
            "SL%": cls.SL_PERCENTAGE, "TRAIL": cls.TRAIL_ENABLED,
            "AUTO_EXIT": f"{cls.AUTO_EXIT_HOUR}:{cls.AUTO_EXIT_MINUTE:02d}",
            "MAX_LOSS": cls.GLOBAL_MAX_LOSS, "MARGIN_BUFFER": f"{cls.MARGIN_BUFFER_PCT}%",
            "INSTRUMENTS": {k: f"{v['breeze_code']}@{v['exchange']}" for k, v in INSTRUMENTS.items()},
        }
