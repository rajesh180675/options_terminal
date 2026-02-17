"""
connection_manager.py

Breeze session + WebSocket manager (NFO + BFO).

Key professional-grade fixes:
  - Avoids name collision with breeze_connect internal config by importing app_config, not config
  - Session token caching (same-day) to reduce re-logins
  - WebSocket heartbeat + reconnect loop
  - Canonical tick normalization: right -> 'call'/'put'
  - Spot feed subscription (cash/index) attempt for NIFTY/CNXBAN/BSESEN
  - Rate-limited REST wrappers (option chain, orders, order detail, positions, quotes)
"""

from __future__ import annotations

import json
import time
import threading
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

from app_config import Config, INSTRUMENTS
from shared_state import SharedState
from models import TickData
from rate_limiter import rate_limited
from utils import LOG, safe_float, safe_int


_FILLED_STATUSES = {"executed", "filled", "complete", "traded"}
_DEAD_STATUSES = {"rejected", "cancelled", "canceled"}


class SessionManager:
    def __init__(self, state: SharedState):
        self.state = state
        self.breeze = None

        self._running = False
        self._ws_last_tick = time.monotonic()
        self._ws_monitor_thread: Optional[threading.Thread] = None

        # backoff for reconnect
        self._reconnect_delay = 1.0
        self._reconnect_delay_max = 60.0

    # ─────────────────────────────────────────────────────────
    # Session init
    # ─────────────────────────────────────────────────────────

    def initialize(self) -> bool:
        """
        Creates BreezeConnect (live) or MockBreeze (mock) and performs generate_session.
        """
        try:
            if Config.is_live():
                LOG.info("Initialising LIVE Breeze session")
                self.state.add_log("INFO", "Session", "Initialising LIVE Breeze session...")

                from breeze_connect import BreezeConnect  # MUST NOT shadow with local config.py
                self.breeze = BreezeConnect(api_key=Config.API_KEY)

                token = self._load_cached_session_token() or Config.SESSION_TOKEN
                if not token:
                    self.state.add_log("ERROR", "Session", "Missing BREEZE_SESSION_TOKEN")
                    return False

                res = self.breeze.generate_session(
                    api_secret=Config.API_SECRET,
                    session_token=token,
                )
                if res and res.get("Status") == 200:
                    self._save_cached_session_token(token)
                    self.state.connected = True
                    self.state.add_log("INFO", "Session", "LIVE session OK ✓")
                    return True

                # If cached failed, try env token once
                if token != Config.SESSION_TOKEN and Config.SESSION_TOKEN:
                    res2 = self.breeze.generate_session(
                        api_secret=Config.API_SECRET,
                        session_token=Config.SESSION_TOKEN,
                    )
                    if res2 and res2.get("Status") == 200:
                        self._save_cached_session_token(Config.SESSION_TOKEN)
                        self.state.connected = True
                        self.state.add_log("INFO", "Session", "LIVE session OK (fresh token) ✓")
                        return True

                err = (res or {}).get("Error", "Unknown error")
                self.state.add_log("ERROR", "Session", f"LIVE session failed: {err}")
                return False

            else:
                LOG.info("Initialising MOCK Breeze session")
                self.state.add_log("INFO", "Session", "Initialising MOCK Breeze session...")

                from mock_breeze import MockBreeze
                self.breeze = MockBreeze(api_key=Config.API_KEY or "mock")
                _ = self.breeze.generate_session(api_secret="mock", session_token="mock")

                self.state.connected = True
                self.state.add_log("INFO", "Session", "MOCK session OK ✓")
                return True

        except Exception as e:
            LOG.exception("Session init failed")
            self.state.add_log("ERROR", "Session", f"Session init exception: {e}")
            return False

    def _load_cached_session_token(self) -> Optional[str]:
        p = Path(Config.SESSION_CACHE)
        if not p.exists():
            return None
        try:
            data = json.loads(p.read_text())
            dt = datetime.fromisoformat(data.get("date", "1970-01-01T00:00:00"))
            if dt.date() == datetime.now().date():
                return data.get("token")
        except Exception:
            return None
        return None

    def _save_cached_session_token(self, token: str) -> None:
        try:
            Path(Config.SESSION_CACHE).write_text(json.dumps({
                "token": token,
                "date": datetime.now().isoformat(),
            }))
        except Exception:
            pass

    # ─────────────────────────────────────────────────────────
    # WebSocket
    # ─────────────────────────────────────────────────────────

    def connect_ws(self) -> None:
        """
        Connect WS, set callback, start heartbeat monitoring, and subscribe spot feeds.
        """
        if not self.breeze:
            raise RuntimeError("Breeze client not initialised")

        self.breeze.on_ticks = self._on_tick
        self.breeze.ws_connect()

        self.state.ws_connected = True
        self._ws_last_tick = time.monotonic()

        self.state.add_log("INFO", "WS", "WebSocket connected")

        # Subscribe spot feeds (best-effort)
        self._subscribe_default_spots()

        # Start heartbeat monitor
        self._running = True
        self._ws_monitor_thread = threading.Thread(
            target=self._ws_heartbeat_loop,
            daemon=True,
            name="WS-Heartbeat",
        )
        self._ws_monitor_thread.start()

    def _subscribe_default_spots(self) -> None:
        """
        Best effort: subscribe underlying spot feeds so SharedState gets spot updates.
        Breeze spot/index semantics can vary; this tries the common patterns.

        If this fails, engine still runs (spot may be zero unless you provide it externally).
        """
        if not self.breeze:
            return

        for _, inst in INSTRUMENTS.items():
            stock_code = inst["breeze_code"]
            try:
                self.subscribe_spot(stock_code)
            except Exception as e:
                self.state.add_log("WARN", "WS", f"Spot subscribe failed for {stock_code}: {e}")

    @rate_limited
    def subscribe_spot(self, stock_code: str) -> dict:
        """
        Subscribe to spot/index feed.
        - NSE indices: exchange_code="NSE", stock_code="NIFTY"/"CNXBAN", product_type="cash"
        - BSE indices: exchange_code="BSE", stock_code="BSESEN", product_type="cash"
        """
        if Config.is_live():
            exc = "BSE" if stock_code.upper() == "BSESEN" else "NSE"
            return self.breeze.subscribe_feeds(
                exchange_code=exc,
                stock_code=stock_code,
                product_type="cash",
                expiry_date="",
                strike_price="",
                right="",
                get_exchange_quotes=True,
                get_market_depth=False,
            )
        else:
            # mock doesn't need this; it has get_spot_price
            return {"Status": 200, "Success": None, "Error": None}

    def _on_tick(self, raw: Dict[str, Any]) -> None:
        """
        Parse raw tick into TickData and store. Also updates spot when non-option tick arrives.
        """
        try:
            stock_code = raw.get("stock_code", "")
            expiry = raw.get("expiry_date", "") or ""
            strike = safe_float(raw.get("strike_price", 0))
            right_raw = (raw.get("right", "") or "").strip().lower()

            # Normalize right ("Call"/"Put" -> "call"/"put")
            if right_raw in ("call", "put"):
                right = right_raw
            elif right_raw in ("c", "ce"):
                right = "call"
            elif right_raw in ("p", "pe"):
                right = "put"
            else:
                right = ""  # likely spot/cash

            ltp = safe_float(raw.get("ltp", 0))
            bid = safe_float(raw.get("best_bid_price", 0))
            ask = safe_float(raw.get("best_offer_price", 0))

            tick = TickData(
                stock_code=stock_code,
                strike_price=strike,
                right=right,
                expiry_date=expiry,
                ltp=ltp,
                best_bid=bid,
                best_ask=ask,
                volume=safe_int(raw.get("total_quantity_traded", raw.get("volume", 0))),
                oi=safe_int(raw.get("open_interest", 0)),
                timestamp=datetime.now(),
            )

            # Spot detection: cash ticks often have strike=0 or right=""
            if (not right) and ltp > 0:
                # update spot
                self.state.set_spot(stock_code, ltp)
            else:
                self.state.update_tick(tick)

            self._ws_last_tick = time.monotonic()

        except Exception as e:
            self.state.add_log("ERROR", "WS", f"Tick parse error: {e}")

    def _ws_heartbeat_loop(self) -> None:
        while self._running:
            time.sleep(5.0)
            if not self.state.ws_connected:
                continue
            elapsed = time.monotonic() - self._ws_last_tick
            if elapsed > float(Config.WS_HEARTBEAT_TIMEOUT):
                self.state.add_log("WARN", "WS", f"Heartbeat lost ({elapsed:.0f}s) → reconnect")
                LOG.warning(f"WS heartbeat stale {elapsed:.0f}s; reconnecting")
                self._reconnect_ws()

    def _reconnect_ws(self) -> None:
        self.state.ws_connected = False
        delay = self._reconnect_delay

        for attempt in range(10):
            try:
                time.sleep(delay)
                self.breeze.ws_connect()
                self.breeze.on_ticks = self._on_tick
                self._ws_last_tick = time.monotonic()
                self.state.ws_connected = True
                self.state.add_log("INFO", "WS", f"Reconnected (attempt {attempt + 1})")
                self._reconnect_delay = 1.0
                return
            except Exception as e:
                self.state.add_log("WARN", "WS", f"Reconnect attempt {attempt + 1} failed: {e}")
                delay = min(delay * 2, self._reconnect_delay_max)

        self.state.add_log("CRIT", "WS", "WS reconnection exhausted")

    def shutdown(self) -> None:
        self._running = False
        try:
            self.state.ws_connected = False
            if self.breeze:
                self.breeze.ws_disconnect()
        except Exception:
            pass

    # ─────────────────────────────────────────────────────────
    # REST wrappers (rate limited)
    # ─────────────────────────────────────────────────────────

    @rate_limited
    def get_option_chain(self, stock_code: str, expiry: str, right: str = "others", exchange_code: str = "NFO") -> dict:
        return self.breeze.get_option_chain_quotes(
            stock_code=stock_code,
            exchange_code=exchange_code,
            product_type="options",
            expiry_date=expiry,
            right=right,
        )

    @rate_limited
    def place_order(self, **kwargs) -> dict:
        return self.breeze.place_order(**kwargs)

    @rate_limited
    def cancel_order(self, order_id: str, exchange_code: str = "NFO") -> dict:
        return self.breeze.cancel_order(exchange_code=exchange_code, order_id=order_id)

    @rate_limited
    def modify_order(self, order_id: str, price: str, exchange_code: str = "NFO", quantity: str = "0") -> dict:
        return self.breeze.modify_order(order_id=order_id, exchange_code=exchange_code, price=price, quantity=quantity)

    @rate_limited
    def get_order_detail(self, order_id: str, exchange_code: str = "NFO") -> dict:
        return self.breeze.get_order_detail(exchange_code=exchange_code, order_id=order_id)

    @rate_limited
    def get_positions(self) -> dict:
        return self.breeze.get_portfolio_positions()

    @rate_limited
    def get_quotes(self, **kwargs) -> dict:
        return self.breeze.get_quotes(**kwargs)

    # ─────────────────────────────────────────────────────────
    # Spot getter
    # ─────────────────────────────────────────────────────────

    def get_spot_price(self, stock_code: str) -> float:
        """
        Live:
          - uses SharedState spot (from WS) if available
          - if missing, tries get_quotes cash as fallback (best effort)
        Mock:
          - uses MockBreeze.get_spot_price()
        """
        if not Config.is_live():
            try:
                return float(self.breeze.get_spot_price(stock_code))
            except Exception:
                return 0.0

        sp = self.state.get_spot(stock_code)
        if sp > 0:
            return sp

        # Best-effort fallback (may or may not work depending on Breeze support for indices)
        try:
            exc = "BSE" if stock_code.upper() == "BSESEN" else "NSE"
            r = self.get_quotes(
                stock_code=stock_code,
                exchange_code=exc,
                product_type="cash",
                expiry_date="",
                right="",
                strike_price="",
            )
            if r and r.get("Success"):
                rec = r["Success"][0]
                ltp = safe_float(rec.get("ltp", 0))
                if ltp > 0:
                    self.state.set_spot(stock_code, ltp)
                    return ltp
        except Exception:
            pass

        return 0.0

    # ─────────────────────────────────────────────────────────
    # WS option subscription helper
    # ─────────────────────────────────────────────────────────

    @rate_limited
    def subscribe_option(self, stock_code: str, strike: float, right: str, expiry: str, exchange_code: str = "NFO") -> dict:
        r = (right or "").strip().lower()
        if r not in ("call", "put"):
            raise ValueError(f"Invalid right: {right}")
        return self.breeze.subscribe_feeds(
            exchange_code=exchange_code,
            stock_code=stock_code,
            product_type="options",
            expiry_date=expiry,
            strike_price=str(int(strike)),
            right=r,
            get_exchange_quotes=True,
            get_market_depth=False,
        )
