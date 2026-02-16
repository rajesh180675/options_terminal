# ═══════════════════════════════════════════════════════════════
# FILE: connection_manager.py
# ═══════════════════════════════════════════════════════════════
"""
Session + WebSocket manager. Supports NFO and BFO exchanges.
Normalises tick 'right' to lowercase for consistent feed_key.
"""

import json, time, threading
from pathlib import Path
from datetime import datetime
from typing import Optional
from app_config import Config
from shared_state import SharedState
from models import TickData
from rate_limiter import rate_limited
from utils import LOG, safe_float, safe_int


class SessionManager:

    def __init__(self, state: SharedState):
        self.state = state
        self.breeze = None
        self._ws_mon = None
        self._ws_tick_t = time.monotonic()
        self._running = False

    def initialize(self) -> bool:
        try:
            if Config.is_live():
                LOG.info("Initialising LIVE Breeze")
                self.state.add_log("INFO", "Session", "Connecting LIVE…")
                from breeze_connect import BreezeConnect
                self.breeze = BreezeConnect(api_key=Config.API_KEY)
                tok = self._cached_token() or Config.SESSION_TOKEN
                if not tok:
                    self.state.add_log("ERROR", "Session", "No session token")
                    return False
                r = self.breeze.generate_session(
                    api_secret=Config.API_SECRET, session_token=tok)
                if r and r.get("Status") == 200:
                    self._cache_token(tok)
                    self.state.connected = True
                    self.state.add_log("INFO", "Session", "LIVE session OK ✓")
                    return True
                # Retry with raw token
                if Config.SESSION_TOKEN and tok != Config.SESSION_TOKEN:
                    r = self.breeze.generate_session(
                        api_secret=Config.API_SECRET,
                        session_token=Config.SESSION_TOKEN)
                    if r and r.get("Status") == 200:
                        self._cache_token(Config.SESSION_TOKEN)
                        self.state.connected = True
                        return True
                err = r.get("Error", "Unknown") if r else "No response"
                self.state.add_log("ERROR", "Session", f"Failed: {err}")
                return False
            else:
                LOG.info("Initialising MOCK Breeze")
                self.state.add_log("INFO", "Session", "MOCK mode")
                from mock_breeze import MockBreeze
                self.breeze = MockBreeze()
                self.breeze.generate_session(api_secret="m", session_token="m")
                self.state.connected = True
                self.state.add_log("INFO", "Session", "MOCK session OK ✓")
                return True
        except ImportError as e:
            self.state.add_log("ERROR", "Session", f"Import: {e}")
            LOG.error(f"Import error: {e}")
            return False
        except Exception as e:
            self.state.add_log("ERROR", "Session", f"Exception: {e}")
            LOG.exception("Session init failed")
            return False

    def _cached_token(self) -> Optional[str]:
        p = Path(Config.SESSION_CACHE)
        if not p.exists():
            return None
        try:
            d = json.loads(p.read_text())
            if datetime.fromisoformat(d["date"]).date() == datetime.now().date():
                return d["token"]
        except Exception:
            pass
        return None

    def _cache_token(self, tok):
        try:
            Path(Config.SESSION_CACHE).write_text(
                json.dumps({"token": tok, "date": datetime.now().isoformat()}))
        except Exception:
            pass

    # ── WebSocket ────────────────────────────────────────────

    def connect_ws(self):
        if not self.breeze:
            return
        self.breeze.on_ticks = self._on_tick
        self.breeze.ws_connect()
        self.state.ws_connected = True
        self._ws_tick_t = time.monotonic()
        self._running = True
        self._ws_mon = threading.Thread(target=self._hb, daemon=True, name="WS-HB")
        self._ws_mon.start()
        self.state.add_log("INFO", "WS", "WebSocket connected")

    def _on_tick(self, raw: dict):
        try:
            right_raw = raw.get("right", "")
            right_n = right_raw.strip().lower() if right_raw else ""
            if right_n not in ("call", "put"):
                right_n = "call"
            tick = TickData(
                stock_code=raw.get("stock_code", ""),
                strike_price=safe_float(raw.get("strike_price", 0)),
                right=right_n,
                expiry_date=raw.get("expiry_date", ""),
                ltp=safe_float(raw.get("ltp", 0)),
                best_bid=safe_float(raw.get("best_bid_price", 0)),
                best_ask=safe_float(raw.get("best_offer_price", 0)),
                volume=safe_int(raw.get("total_quantity_traded", 0)),
                oi=safe_int(raw.get("open_interest", 0)),
                timestamp=datetime.now())
            self.state.update_tick(tick)
            self._ws_tick_t = time.monotonic()
        except Exception as e:
            LOG.error(f"Tick parse: {e}")

    def _hb(self):
        while self._running:
            time.sleep(5)
            if time.monotonic() - self._ws_tick_t > Config.WS_HEARTBEAT_TIMEOUT:
                if self.state.ws_connected:
                    self.state.add_log("WARN", "WS", "Heartbeat lost")
                    self._reconnect()

    def _reconnect(self):
        self.state.ws_connected = False
        delay = 1.0
        for att in range(10):
            try:
                time.sleep(delay)
                self.breeze.ws_connect()
                self.breeze.on_ticks = self._on_tick
                self._ws_tick_t = time.monotonic()
                self.state.ws_connected = True
                self.state.add_log("INFO", "WS", f"Reconnected (att {att+1})")
                self._resub()
                return
            except Exception:
                delay = min(delay * 2, 60)
        self.state.add_log("CRIT", "WS", "Reconnect exhausted")

    def _resub(self):
        for k, t in self.state.get_all_ticks().items():
            try:
                self.subscribe_option(t.stock_code, t.strike_price,
                                      t.right, t.expiry_date,
                                      Config.exchange(t.stock_code))
            except Exception:
                pass

    # ── Subscription ─────────────────────────────────────────

    @rate_limited
    def subscribe_option(self, stock_code, strike, right, expiry,
                         exchange_code="NFO"):
        r = right.strip().lower()
        if r not in ("call", "put"):
            r = "call"
        self.breeze.subscribe_feeds(
            exchange_code=exchange_code,
            stock_code=stock_code,
            product_type="options",
            expiry_date=expiry,
            strike_price=str(int(strike)),
            right=r,
            get_exchange_quotes=True,
            get_market_depth=False)

    @rate_limited
    def get_option_chain(self, stock_code, expiry, right="others",
                         exchange_code="NFO"):
        return self.breeze.get_option_chain_quotes(
            stock_code=stock_code, exchange_code=exchange_code,
            product_type="options", expiry_date=expiry, right=right)

    @rate_limited
    def place_order(self, **kw):
        return self.breeze.place_order(**kw)

    @rate_limited
    def cancel_order(self, order_id, exchange_code="NFO"):
        return self.breeze.cancel_order(exchange_code=exchange_code, order_id=order_id)

    @rate_limited
    def modify_order(self, order_id, price, exchange_code="NFO", **kw):
        return self.breeze.modify_order(
            order_id=order_id, exchange_code=exchange_code, price=price, **kw)

    @rate_limited
    def get_order_detail(self, order_id, exchange_code="NFO"):
        return self.breeze.get_order_detail(
            exchange_code=exchange_code, order_id=order_id)

    def get_spot_price(self, stock_code):
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
