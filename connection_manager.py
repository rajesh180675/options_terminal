# ═══════════════════════════════════════════════════════════════
# FILE: connection_manager.py
# ═══════════════════════════════════════════════════════════════
"""
Manages the Breeze session lifecycle:
  • generate_session with local token caching (avoids redundant logins)
  • WebSocket connect / auto-reconnect with heartbeat monitoring
  • Tick dispatch into SharedState
"""

import json
import time
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

from config import Config
from shared_state import SharedState
from models import TickData
from rate_limiter import rate_limited
from utils import LOG, safe_float, safe_int


class SessionManager:
    """Handles Breeze session establishment, caching, and WebSocket resilience."""

    def __init__(self, shared_state: SharedState):
        self.state = shared_state
        self.breeze = None  # Will hold BreezeConnect or MockBreeze
        self._ws_monitor_thread: Optional[threading.Thread] = None
        self._ws_last_tick_time: float = time.monotonic()
        self._ws_heartbeat_timeout: float = 30.0  # seconds
        self._running = False
        self._reconnect_delay = 1.0
        self._max_reconnect_delay = 60.0

    def initialize(self) -> bool:
        """Create the Breeze client (mock or live) and generate session."""
        try:
            if Config.TRADING_MODE == "mock":
                from mock_breeze import MockBreeze
                self.breeze = MockBreeze(api_key=Config.API_KEY)
                result = self.breeze.generate_session(
                    api_secret=Config.API_SECRET,
                    session_token=Config.SESSION_TOKEN,
                )
            else:
                from breeze_connect import BreezeConnect
                self.breeze = BreezeConnect(api_key=Config.API_KEY)
                session_token = self._load_cached_session()
                if not session_token:
                    session_token = Config.SESSION_TOKEN
                result = self.breeze.generate_session(
                    api_secret=Config.API_SECRET,
                    session_token=session_token,
                )
                self._cache_session(session_token)

            if result and result.get("Status") == 200:
                self.state.connected = True
                self.state.add_log("INFO", "Session", "Breeze session established")
                LOG.info("Breeze session established")
                return True
            else:
                error = result.get("Error", "Unknown error") if result else "No response"
                self.state.add_log("ERROR", "Session", f"Session failed: {error}")
                LOG.error(f"Session generation failed: {error}")
                return False
        except Exception as e:
            self.state.add_log("ERROR", "Session", f"Session exception: {e}")
            LOG.exception("Session initialization failed")
            return False

    def _load_cached_session(self) -> Optional[str]:
        """Load cached session token if it was created today."""
        cache_file = Path(Config.SESSION_CACHE_FILE)
        if not cache_file.exists():
            return None
        try:
            data = json.loads(cache_file.read_text())
            cached_date = datetime.fromisoformat(data["date"])
            if cached_date.date() == datetime.now().date():
                LOG.info("Loaded cached session token (same-day)")
                return data["token"]
            LOG.info("Cached session expired (different day)")
        except Exception as e:
            LOG.warning(f"Cache read error: {e}")
        return None

    def _cache_session(self, token: str):
        """Persist session token with today's date."""
        cache_file = Path(Config.SESSION_CACHE_FILE)
        try:
            cache_file.write_text(json.dumps({
                "token": token,
                "date": datetime.now().isoformat(),
            }))
            LOG.info("Session token cached")
        except Exception as e:
            LOG.warning(f"Cache write error: {e}")

    # ── WebSocket ────────────────────────────────────────────

    def connect_websocket(self):
        """Start WebSocket and heartbeat monitor."""
        if self.breeze is None:
            LOG.error("Cannot connect WS: no Breeze client")
            return

        self.breeze.on_ticks = self._on_tick_callback
        self.breeze.ws_connect()
        self.state.ws_connected = True
        self._ws_last_tick_time = time.monotonic()
        self.state.add_log("INFO", "WebSocket", "WebSocket connected")

        # Start heartbeat monitor
        self._running = True
        self._ws_monitor_thread = threading.Thread(
            target=self._ws_heartbeat_loop, daemon=True, name="WS-Monitor"
        )
        self._ws_monitor_thread.start()

    def _on_tick_callback(self, tick_data: dict):
        """Parse raw Breeze tick into TickData and push to SharedState."""
        try:
            tick = TickData(
                stock_code=tick_data.get("stock_code", ""),
                strike_price=safe_float(tick_data.get("strike_price", 0)),
                right=tick_data.get("right", "").lower(),
                expiry_date=tick_data.get("expiry_date", ""),
                ltp=safe_float(tick_data.get("ltp", 0)),
                best_bid=safe_float(tick_data.get("best_bid_price", 0)),
                best_ask=safe_float(tick_data.get("best_offer_price", 0)),
                volume=safe_int(tick_data.get("total_quantity_traded", 0)),
                oi=safe_int(tick_data.get("open_interest", 0)),
                timestamp=datetime.now(),
            )
            self.state.update_tick(tick)
            self._ws_last_tick_time = time.monotonic()
        except Exception as e:
            LOG.error(f"Tick parse error: {e} | raw={tick_data}")

    def _ws_heartbeat_loop(self):
        """Monitor WS liveness; auto-reconnect if stale."""
        while self._running:
            time.sleep(5.0)
            elapsed = time.monotonic() - self._ws_last_tick_time
            if elapsed > self._ws_heartbeat_timeout and self.state.ws_connected:
                LOG.warning(f"WS heartbeat stale ({elapsed:.0f}s). Reconnecting...")
                self.state.add_log("WARN", "WebSocket", "Heartbeat lost – reconnecting")
                self._reconnect()

    def _reconnect(self):
        """Exponential-backoff reconnection."""
        self.state.ws_connected = False
        delay = self._reconnect_delay
        for attempt in range(10):
            try:
                LOG.info(f"WS reconnect attempt {attempt + 1} (delay={delay:.1f}s)")
                time.sleep(delay)
                self.breeze.ws_connect()
                self.breeze.on_ticks = self._on_tick_callback
                self._ws_last_tick_time = time.monotonic()
                self.state.ws_connected = True
                self.state.add_log("INFO", "WebSocket", f"Reconnected (attempt {attempt + 1})")
                self._reconnect_delay = 1.0  # Reset
                # Re-subscribe existing feeds
                self._resubscribe_active_feeds()
                return
            except Exception as e:
                LOG.error(f"Reconnect failed: {e}")
                delay = min(delay * 2, self._max_reconnect_delay)
        LOG.critical("WebSocket reconnection exhausted")
        self.state.add_log("CRIT", "WebSocket", "Reconnection failed after 10 attempts")

    def _resubscribe_active_feeds(self):
        """Re-subscribe to all ticks we were previously tracking."""
        ticks = self.state.get_all_ticks()
        for feed_key, tick in ticks.items():
            try:
                self.subscribe_option(
                    stock_code=tick.stock_code,
                    strike_price=tick.strike_price,
                    right=tick.right,
                    expiry_date=tick.expiry_date,
                )
            except Exception as e:
                LOG.error(f"Re-subscribe failed for {feed_key}: {e}")

    # ── Subscription helpers ─────────────────────────────────

    @rate_limited
    def subscribe_option(
        self,
        stock_code: str,
        strike_price: float,
        right: str,
        expiry_date: str,
    ):
        self.breeze.subscribe_feeds(
            exchange_code="NFO",
            stock_code=stock_code,
            product_type="options",
            expiry_date=expiry_date,
            strike_price=str(int(strike_price)),
            right=right.lower(),
            get_exchange_quotes=True,
            get_market_depth=False,
        )

    @rate_limited
    def subscribe_spot(self, stock_code: str):
        """Subscribe to the underlying index feed (NSE cash segment)."""
        if Config.TRADING_MODE == "mock":
            # Mock: we get spot from the mock breeze object
            return
        self.breeze.subscribe_feeds(
            exchange_code="NSE",
            stock_code=stock_code,
            product_type="cash",
            expiry_date="",
            strike_price="",
            right="",
            get_exchange_quotes=True,
            get_market_depth=False,
        )

    # ── API wrappers (rate-limited) ──────────────────────────

    @rate_limited
    def get_option_chain(self, stock_code: str, expiry_date: str, right: str = "others") -> dict:
        return self.breeze.get_option_chain_quotes(
            stock_code=stock_code,
            exchange_code="NFO",
            product_type="options",
            expiry_date=expiry_date,
            right=right,
        )

    @rate_limited
    def place_order(self, **kwargs) -> dict:
        return self.breeze.place_order(**kwargs)

    @rate_limited
    def cancel_order(self, order_id: str) -> dict:
        return self.breeze.cancel_order(exchange_code="NFO", order_id=order_id)

    @rate_limited
    def modify_order(self, order_id: str, price: str, **kwargs) -> dict:
        return self.breeze.modify_order(order_id=order_id, exchange_code="NFO", price=price, **kwargs)

    @rate_limited
    def get_order_detail(self, order_id: str) -> dict:
        return self.breeze.get_order_detail(exchange_code="NFO", order_id=order_id)

    @rate_limited
    def get_positions(self) -> dict:
        return self.breeze.get_portfolio_positions()

    def get_spot_price(self, stock_code: str) -> float:
        """Get current spot price from mock or live feed."""
        if Config.TRADING_MODE == "mock":
            return self.breeze.get_spot_price(stock_code)
        # In live mode, the spot comes via WS ticks into SharedState
        return self.state.get_spot(stock_code)

    def shutdown(self):
        self._running = False
        self.state.ws_connected = False
        try:
            if hasattr(self.breeze, "ws_disconnect"):
                self.breeze.ws_disconnect()
        except Exception:
            pass
        LOG.info("Connection manager shut down")
