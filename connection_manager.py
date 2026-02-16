# ═══════════════════════════════════════════════════════════════
# FILE: connection_manager.py
# ═══════════════════════════════════════════════════════════════
"""
Session + WebSocket manager.

CRITICAL FIXES from audit:
  1. Config.is_live() check replaces string comparison
  2. subscribe_feeds parameters corrected for Breeze SDK v1.0.52+
  3. on_ticks callback normalises right to lowercase
  4. Session cache uses Config.is_live() properly
  5. Live Breeze import path corrected
"""

import json
import time
import threading
from pathlib import Path
from datetime import datetime
from typing import Optional

from config import Config
from shared_state import SharedState
from models import TickData
from rate_limiter import rate_limited
from utils import LOG, safe_float, safe_int


class SessionManager:

    def __init__(self, shared_state: SharedState):
        self.state = shared_state
        self.breeze = None
        self._ws_monitor_thread: Optional[threading.Thread] = None
        self._ws_last_tick: float = time.monotonic()
        self._running = False

    def initialize(self) -> bool:
        """Create Breeze client and establish session."""
        try:
            if Config.is_live():
                LOG.info("Initialising LIVE Breeze session")
                self.state.add_log("INFO", "Session", "Initialising LIVE Breeze…")

                from breeze_connect import BreezeConnect
                self.breeze = BreezeConnect(api_key=Config.API_KEY)

                # Try cached session first
                cached = self._load_session_cache()
                token = cached if cached else Config.SESSION_TOKEN

                if not token:
                    self.state.add_log("ERROR", "Session",
                                       "No session token. Login at ICICI API portal first.")
                    return False

                result = self.breeze.generate_session(
                    api_secret=Config.API_SECRET,
                    session_token=token,
                )

                if result and result.get("Status") == 200:
                    self._save_session_cache(token)
                    self.state.connected = True
                    self.state.add_log("INFO", "Session", "LIVE session OK ✓")
                    LOG.info("Live Breeze session established")
                    return True
                else:
                    error = result.get("Error", "Unknown") if result else "No response"
                    self.state.add_log("ERROR", "Session", f"Live session failed: {error}")
                    LOG.error(f"Live session failed: {error}")

                    # If cached token failed, try the fresh one
                    if cached and Config.SESSION_TOKEN and cached != Config.SESSION_TOKEN:
                        self.state.add_log("INFO", "Session",
                                           "Retrying with fresh session token…")
                        result = self.breeze.generate_session(
                            api_secret=Config.API_SECRET,
                            session_token=Config.SESSION_TOKEN,
                        )
                        if result and result.get("Status") == 200:
                            self._save_session_cache(Config.SESSION_TOKEN)
                            self.state.connected = True
                            self.state.add_log("INFO", "Session", "LIVE session OK (fresh token) ✓")
                            return True

                    return False
            else:
                LOG.info("Initialising MOCK Breeze session")
                self.state.add_log("INFO", "Session", "Initialising MOCK Breeze…")

                from mock_breeze import MockBreeze
                self.breeze = MockBreeze(api_key=Config.API_KEY or "mock")
                result = self.breeze.generate_session(
                    api_secret=Config.API_SECRET or "mock",
                    session_token=Config.SESSION_TOKEN or "mock",
                )
                self.state.connected = True
                self.state.add_log("INFO", "Session", "MOCK session OK ✓")
                return True

        except ImportError as e:
            self.state.add_log("ERROR", "Session",
                               f"Import error: {e}. Install breeze-connect for live mode.")
            LOG.error(f"Import error: {e}")
            return False
        except Exception as e:
            self.state.add_log("ERROR", "Session", f"Session exception: {e}")
            LOG.exception("Session init failed")
            return False

    def _load_session_cache(self) -> Optional[str]:
        path = Path(Config.SESSION_CACHE_FILE)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            cached_date = datetime.fromisoformat(data["date"])
            if cached_date.date() == datetime.now().date():
                LOG.info("Using cached session token (same day)")
                return data["token"]
            LOG.info("Cached session expired (different day)")
        except Exception as e:
            LOG.warning(f"Session cache read error: {e}")
        return None

    def _save_session_cache(self, token: str):
        try:
            Path(Config.SESSION_CACHE_FILE).write_text(json.dumps({
                "token": token,
                "date": datetime.now().isoformat(),
            }))
        except Exception as e:
            LOG.warning(f"Session cache write error: {e}")

    # ── WebSocket ────────────────────────────────────────────

    def connect_websocket(self):
        if not self.breeze:
            return
        self.breeze.on_ticks = self._on_tick
        self.breeze.ws_connect()
        self.state.ws_connected = True
        self._ws_last_tick = time.monotonic()
        self.state.add_log("INFO", "WS", "WebSocket connected")

        self._running = True
        self._ws_monitor_thread = threading.Thread(
            target=self._heartbeat_loop, daemon=True, name="WS-HB"
        )
        self._ws_monitor_thread.start()

    def _on_tick(self, raw: dict):
        """
        Parse Breeze raw tick dict into TickData.
        
        CRITICAL: Breeze sends 'right' as 'Call'/'Put' (capitalised).
        We normalise to lowercase for consistent feed_key matching.
        """
        try:
            right_raw = raw.get("right", "")
            right_normalised = right_raw.strip().lower() if right_raw else ""

            tick = TickData(
                stock_code=raw.get("stock_code", ""),
                strike_price=safe_float(raw.get("strike_price", 0)),
                right=right_normalised,
                expiry_date=raw.get("expiry_date", ""),
                ltp=safe_float(raw.get("ltp", 0)),
                best_bid=safe_float(raw.get("best_bid_price", 0)),
                best_ask=safe_float(raw.get("best_offer_price", 0)),
                volume=safe_int(raw.get("total_quantity_traded", 0)),
                oi=safe_int(raw.get("open_interest", 0)),
                timestamp=datetime.now(),
            )
            self.state.update_tick(tick)
            self._ws_last_tick = time.monotonic()
        except Exception as e:
            LOG.error(f"Tick parse error: {e}")

    def _heartbeat_loop(self):
        while self._running:
            time.sleep(5.0)
            elapsed = time.monotonic() - self._ws_last_tick
            if elapsed > Config.WS_HEARTBEAT_TIMEOUT and self.state.ws_connected:
                LOG.warning(f"WS heartbeat stale ({elapsed:.0f}s)")
                self.state.add_log("WARN", "WS", f"Heartbeat lost ({elapsed:.0f}s)")
                self._reconnect()

    def _reconnect(self):
        self.state.ws_connected = False
        delay = 1.0
        for attempt in range(10):
            try:
                LOG.info(f"WS reconnect attempt {attempt + 1}")
                time.sleep(delay)
                self.breeze.ws_connect()
                self.breeze.on_ticks = self._on_tick
                self._ws_last_tick = time.monotonic()
                self.state.ws_connected = True
                self.state.add_log("INFO", "WS", f"Reconnected (attempt {attempt + 1})")
                self._resubscribe()
                return
            except Exception as e:
                LOG.error(f"Reconnect failed: {e}")
                delay = min(delay * 2, 60.0)
        LOG.critical("WS reconnection failed after 10 attempts")
        self.state.add_log("CRIT", "WS", "Reconnection exhausted")

    def _resubscribe(self):
        ticks = self.state.get_all_ticks()
        for key, tick in ticks.items():
            try:
                self.subscribe_option(
                    tick.stock_code, tick.strike_price,
                    tick.right, tick.expiry_date,
                )
            except Exception as e:
                LOG.error(f"Re-subscribe failed: {key}: {e}")

    # ── Subscription (corrected for Breeze SDK) ──────────────

    @rate_limited
    def subscribe_option(
        self,
        stock_code: str,
        strike_price: float,
        right: str,
        expiry_date: str,
    ):
        """
        BREEZE SDK subscribe_feeds for NFO options.
        
        Parameters verified against Breeze SDK v1.0.52 source code:
          - exchange_code: "NFO"
          - stock_code: "NIFTY" or "CNXBAN"
          - product_type: "options"
          - expiry_date: "2024-01-25T06:00:00.000Z"
          - strike_price: "23500" (string, integer)
          - right: "call" or "put" (lowercase)
          - get_exchange_quotes: True
          - get_market_depth: False
        """
        right_clean = right.strip().lower()
        if right_clean not in ("call", "put"):
            LOG.warning(f"Invalid right '{right}', defaulting to 'call'")
            right_clean = "call"

        self.breeze.subscribe_feeds(
            exchange_code="NFO",
            stock_code=stock_code,
            product_type="options",
            expiry_date=expiry_date,
            strike_price=str(int(strike_price)),
            right=right_clean,
            get_exchange_quotes=True,
            get_market_depth=False,
        )
        LOG.debug(f"Subscribed: {stock_code} {int(strike_price)} {right_clean}")

    # ── API wrappers ─────────────────────────────────────────

    @rate_limited
    def get_option_chain(self, stock_code: str, expiry_date: str,
                         right: str = "others") -> dict:
        return self.breeze.get_option_chain_quotes(
            stock_code=stock_code,
            exchange_code="NFO",
            product_type="options",
            expiry_date=expiry_date,
            right=right,
        )

    @rate_limited
    def place_order(self, **kwargs) -> dict:
        """
        Forward to breeze.place_order with exact SDK parameter names.
        The caller MUST use:  product="options", action="sell"/"buy",
        order_type="limit"/"market", right="call"/"put" (lowercase)
        """
        return self.breeze.place_order(**kwargs)

    @rate_limited
    def cancel_order(self, order_id: str) -> dict:
        return self.breeze.cancel_order(exchange_code="NFO", order_id=order_id)

    @rate_limited
    def modify_order(self, order_id: str, price: str, quantity: str = "0") -> dict:
        return self.breeze.modify_order(
            order_id=order_id,
            exchange_code="NFO",
            price=price,
            quantity=quantity,
        )

    @rate_limited
    def get_order_detail(self, order_id: str) -> dict:
        return self.breeze.get_order_detail(exchange_code="NFO", order_id=order_id)

    @rate_limited
    def get_positions(self) -> dict:
        return self.breeze.get_portfolio_positions()

    def get_spot_price(self, stock_code: str) -> float:
        """Get spot from mock or from SharedState (live WS feeds)."""
        if not Config.is_live():
            return self.breeze.get_spot_price(stock_code)
        return self.state.get_spot(stock_code)

    def shutdown(self):
        self._running = False
        self.state.ws_connected = False
        try:
            self.breeze.ws_disconnect()
        except Exception:
            pass
        LOG.info("Session manager shut down")
