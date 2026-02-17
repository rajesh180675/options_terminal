# ═══════════════════════════════════════════════════════════════
# FILE: alerts.py  (NEW)
# ═══════════════════════════════════════════════════════════════
"""
Multi-channel alert system.
  • Telegram bot notifications
  • Severity-based filtering
  • Rate limiting (no spam)
"""

import time
import threading
from collections import deque
from typing import Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from app_config import Config
from utils import LOG


class AlertLevel(str, Enum):
    INFO = "info"
    WARN = "warn"
    CRITICAL = "critical"


@dataclass
class Alert:
    level: AlertLevel
    title: str
    body: str
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().strftime("%H:%M:%S")


class AlertManager:
    """Dispatches alerts to Telegram with duplicate suppression."""

    def __init__(self):
        self._bot_token = Config.TELEGRAM_BOT_TOKEN
        self._chat_id = Config.TELEGRAM_CHAT_ID
        self._enabled = bool(self._bot_token and self._chat_id)
        self._recent: deque = deque(maxlen=100)
        self._lock = threading.Lock()
        self._min_level = AlertLevel(
            Config.TELEGRAM_MIN_LEVEL.lower()
            if hasattr(Config, "TELEGRAM_MIN_LEVEL")
            else "warn"
        )
        self._cooldown: dict = {}  # title → last_sent_time
        self._cooldown_seconds = 60  # Don't repeat same alert within 60s

        if self._enabled:
            LOG.info("Telegram alerts enabled")
        else:
            LOG.info("Telegram alerts disabled (no token/chat_id)")

    def send(self, level: AlertLevel, title: str, body: str):
        """Queue and dispatch an alert."""
        alert = Alert(level=level, title=title, body=body)

        with self._lock:
            self._recent.append(alert)

        # Severity filter
        levels = [AlertLevel.INFO, AlertLevel.WARN, AlertLevel.CRITICAL]
        if levels.index(level) < levels.index(self._min_level):
            return

        # Cooldown check
        now = time.time()
        with self._lock:
            last = self._cooldown.get(title, 0)
            if now - last < self._cooldown_seconds:
                return
            self._cooldown[title] = now

        # Dispatch
        if self._enabled:
            threading.Thread(
                target=self._send_telegram, args=(alert,),
                daemon=True
            ).start()

    def _send_telegram(self, alert: Alert):
        """Send via Telegram Bot API."""
        try:
            import requests

            emoji = {"info": "ℹ️", "warn": "⚠️", "critical": "🚨"}.get(
                alert.level.value, "📢"
            )

            text = (
                f"{emoji} <b>{alert.title}</b>\n"
                f"<i>{alert.timestamp}</i>\n\n"
                f"{alert.body}"
            )

            requests.post(
                f"https://api.telegram.org/bot{self._bot_token}/sendMessage",
                json={
                    "chat_id": self._chat_id,
                    "text": text,
                    "parse_mode": "HTML",
                    "disable_notification": alert.level == AlertLevel.INFO,
                },
                timeout=10,
            )
        except Exception as e:
            LOG.error(f"Telegram send failed: {e}")

    def get_recent(self, n: int = 20) -> list:
        with self._lock:
            return list(self._recent)[-n:]

    # ── Convenience methods ──────────────────────────────────

    def sl_hit(self, leg_name: str, ltp: float, sl: float):
        self.send(AlertLevel.CRITICAL, "Stop Loss Hit",
                  f"{leg_name}\nLTP: ₹{ltp:.2f}\nSL: ₹{sl:.2f}")

    def order_filled(self, leg_name: str, price: float, action: str):
        self.send(AlertLevel.INFO, f"Order {action.upper()}",
                  f"{leg_name} @ ₹{price:.2f}")

    def panic_exit(self, reason: str):
        self.send(AlertLevel.CRITICAL, "PANIC EXIT",
                  f"All positions closed\nReason: {reason}")

    def margin_warning(self, available: float, required: float):
        self.send(AlertLevel.WARN, "Margin Warning",
                  f"Available: ₹{available:,.0f}\nRequired: ₹{required:,.0f}")

    def delta_drift(self, delta: float, threshold: float):
        self.send(AlertLevel.WARN, "Delta Drift",
                  f"Portfolio Δ: {delta:+.1f}\nThreshold: ±{threshold:.0f}")

    def adjustment_suggestion(self, message: str):
        self.send(AlertLevel.INFO, "Adjustment Suggestion", message)

    def auto_exit_triggered(self):
        self.send(AlertLevel.CRITICAL, "Auto Exit",
                  f"Positions closed at {Config.AUTO_EXIT_HOUR}:{Config.AUTO_EXIT_MINUTE:02d}")
