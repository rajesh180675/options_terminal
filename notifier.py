# ═══════════════════════════════════════════════════════════════
# FILE: notifier.py  (NEW — Telegram + webhook alerts)
# ═══════════════════════════════════════════════════════════════
"""
Non-blocking notification system.
Sends alerts via Telegram Bot API and/or webhook.
Uses background thread with queue — never blocks trading logic.

Setup:
  1. Create Telegram bot via @BotFather
  2. Get TELEGRAM_BOT_TOKEN
  3. Send /start to your bot, get TELEGRAM_CHAT_ID
  4. Set both in .env or st.secrets
"""

import json
import threading
import urllib.request
from queue import Queue, Empty
from datetime import datetime
from typing import Optional

from app_config import Config
from utils import LOG


class Notifier:
    """
    Fire-and-forget notification system.
    Events: SL hit, panic, strategy deployed, margin warning, delta drift.
    """

    def __init__(self):
        self._token = Config._get("TELEGRAM_BOT_TOKEN", "")
        self._chat_id = Config._get("TELEGRAM_CHAT_ID", "")
        self._webhook = Config._get("WEBHOOK_URL", "")
        self._queue: Queue = Queue()
        self._enabled = bool(self._token and self._chat_id) or bool(self._webhook)
        self._thread: Optional[threading.Thread] = None

        if self._enabled:
            self._thread = threading.Thread(
                target=self._sender_loop, daemon=True, name="Notifier"
            )
            self._thread.start()
            LOG.info(f"Notifier active: Telegram={'YES' if self._token else 'NO'}, "
                     f"Webhook={'YES' if self._webhook else 'NO'}")

    def send(self, message: str, level: str = "INFO"):
        """Non-blocking send. Returns immediately."""
        if not self._enabled:
            return
        self._queue.put((message, level))

    def _sender_loop(self):
        while True:
            try:
                msg, level = self._queue.get(timeout=5)
            except Empty:
                continue

            ts = datetime.now().strftime("%H:%M:%S")
            formatted = f"[{ts}] {level}\n{msg}"

            if self._token and self._chat_id:
                self._send_telegram(formatted)
            if self._webhook:
                self._send_webhook(formatted)

            self._queue.task_done()

    def _send_telegram(self, text: str):
        try:
            url = f"https://api.telegram.org/bot{self._token}/sendMessage"
            payload = json.dumps({
                "chat_id": self._chat_id,
                "text": text,
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            }).encode("utf-8")
            req = urllib.request.Request(
                url, data=payload,
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(req, timeout=10)
        except Exception as e:
            LOG.debug(f"Telegram send error: {e}")

    def _send_webhook(self, text: str):
        try:
            payload = json.dumps({"text": text, "content": text}).encode("utf-8")
            req = urllib.request.Request(
                self._webhook, data=payload,
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(req, timeout=10)
        except Exception as e:
            LOG.debug(f"Webhook send error: {e}")

    @property
    def is_active(self) -> bool:
        return self._enabled
