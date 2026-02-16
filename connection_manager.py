"""
connection_manager.py — Breeze session management, WebSocket resiliency,
and rate-limiting.  Provides both LiveBreeze and MockBreeze backends.
"""

from __future__ import annotations

import functools
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from config import CFG
from shared_state import SharedState, TickData


# ================================================================
# Rate Limiter (Token Bucket)
# ================================================================

class RateLimiter:
    """Token-bucket rate limiter: max `max_calls` per `period` seconds."""

    def __init__(self, max_calls: int = 100, period: int = 60):
        self.max_calls = max_calls
        self.period = period
        self._timestamps: list[float] = []
        self._lock = threading.Lock()

    def acquire(self) -> float:
        """Block until a slot is available.  Returns wait time."""
        with self._lock:
            now = time.time()
            self._timestamps = [t for t in self._timestamps
                                if now - t < self.period]
            if len(self._timestamps) >= self.max_calls:
                sleep_for = self.period - (now - self._timestamps[0]) + 0.05
                self._lock.release()
                time.sleep(sleep_for)
                self._lock.acquire()
                return self.acquire()
            self._timestamps.append(time.time())
            return 0.0


_rate_limiter = RateLimiter(CFG.api_rate_limit, CFG.api_rate_period)


def rate_limited(func):
    """Decorator that enforces API rate limits."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        _rate_limiter.acquire()
        return func(*args, **kwargs)
    return wrapper


# ================================================================
# Breeze Adapter Protocol
# ================================================================

@runtime_checkable
class BreezeAdapter(Protocol):
    def generate_session(self, api_secret: str, session_token: str) -> dict: ...
    def connect(self) -> None: ...
    def disconnect(self) -> None: ...
    def subscribe_feeds(self, **kwargs) -> dict: ...
    def unsubscribe_feeds(self, **kwargs) -> dict: ...
    def get_option_chain_quotes(self, **kwargs) -> dict: ...
    def place_order(self, **kwargs) -> dict: ...
    def modify_order(self, **kwargs) -> dict: ...
    def cancel_order(self, **kwargs) -> dict: ...
    def get_order_detail(self, **kwargs) -> dict: ...
    def get_quotes(self, **kwargs) -> dict: ...
    def get_portfolio_positions(self) -> dict: ...


# ================================================================
# Session Cache
# ================================================================

class SessionCache:
    """Persist session tokens to avoid re-login within 24h."""

    def __init__(self, cache_path: str = ".session_cache"):
        self._path = Path(cache_path)

    def save(self, token: str):
        data = {"token": token, "ts": time.time()}
        self._path.write_text(json.dumps(data))

    def load(self) -> str | None:
        if not self._path.exists():
            return None
        try:
            data = json.loads(self._path.read_text())
            age_hours = (time.time() - data["ts"]) / 3600
            if age_hours < 23:  # 24h expiry with 1h buffer
                return data["token"]
        except Exception:
            pass
        return None

    def clear(self):
        if self._path.exists():
            self._path.unlink()


# ================================================================
# Connection Manager
# ================================================================

class ConnectionManager:
    """
    Manages the Breeze connection lifecycle:
      - Session authentication
      - WebSocket connection & reconnection
      - Tick data routing to SharedState
      - Rate-limited API proxy
    """

    def __init__(self, state: SharedState, mode: str = "mock"):
        self.state = state
        self.mode = mode
        self._breeze: Any = None
        self._session_cache = SessionCache(CFG.session_cache_path)
        self._ws_monitor_thread: threading.Thread | None = None
        self._ws_stop = threading.Event()
        self._connected = False
        self._lock = threading.Lock()

        self._init_breeze()

    def _init_breeze(self):
        if self.mode == "mock":
            from mock_breeze import MockBreeze
            self._breeze = MockBreeze(api_key=CFG.api_key)
        else:
            try:
                from breeze_connect import BreezeConnect
                self._breeze = BreezeConnect(api_key=CFG.api_key)
            except ImportError:
                raise ImportError(
                    "breeze-connect not installed.  "
                    "Run: pip install breeze-connect"
                )

    # -------------------------------------------------------- authentication
    def authenticate(self) -> bool:
        """
        Perform session handshake.
        Uses cached token if available and valid.
        """
        # Try cached session first
        cached = self._session_cache.load()
        token = cached or CFG.session_token

        if not token:
            self.state.add_log("ERROR", "ConnMgr",
                               "No session token available. "
                               "Set BREEZE_SESSION_TOKEN in .env")
            return False

        try:
            resp = self._breeze.generate_session(
                api_secret=CFG.api_secret,
                session_token=token,
            )
            if resp.get("Error"):
                self.state.add_log("ERROR", "ConnMgr",
                                   f"Session generation failed: {resp['Error']}")
                self._session_cache.clear()
                return False

            # Cache the new session
            new_token = ""
            if isinstance(resp.get("Success"), dict):
                new_token = resp["Success"].get("session_token", token)
            self._session_cache.save(new_token or token)

            self._connected = True
            self.state.set_api_connected(True)
            self.state.add_log("INFO", "ConnMgr",
                               "Session authenticated successfully")
            return True

        except Exception as e:
            self.state.add_log("ERROR", "ConnMgr",
                               f"Authentication exception: {e}")
            return False

    # --------------------------------------------------- websocket lifecycle
    def start_websocket(self):
        """Start WebSocket and the reconnection monitor."""
        self._breeze.on_ticks = self._on_tick_callback
        try:
            self._breeze.connect()
            self.state.set_ws_connected(True)
            self.state.add_log("INFO", "ConnMgr", "WebSocket connected")
        except Exception as e:
            self.state.add_log("ERROR", "ConnMgr",
                               f"WebSocket connect failed: {e}")
            self.state.set_ws_connected(False)

        # Start heartbeat monitor
        self._ws_stop.clear()
        self._ws_monitor_thread = threading.Thread(
            target=self._ws_heartbeat_monitor,
            daemon=True,
            name="WS-Heartbeat",
        )
        self._ws_monitor_thread.start()

    def stop_websocket(self):
        self._ws_stop.set()
        try:
            self._breeze.disconnect()
        except Exception:
            pass
        self.state.set_ws_connected(False)
        self.state.add_log("INFO", "ConnMgr", "WebSocket disconnected")

    def _ws_heartbeat_monitor(self):
        """Monitor WebSocket health and reconnect if stale."""
        stale_threshold = 15.0  # seconds without a tick
        while not self._ws_stop.is_set():
            age = self.state.get_last_tick_age()
            if age > stale_threshold and self.state.is_ws_connected():
                self.state.add_log("WARN", "ConnMgr",
                                   f"WebSocket stale ({age:.0f}s). Reconnecting...")
                self.state.set_ws_connected(False)
                try:
                    self._breeze.disconnect()
                except Exception:
                    pass
                time.sleep(2)
                try:
                    self._breeze.on_ticks = self._on_tick_callback
                    self._breeze.connect()
                    self.state.set_ws_connected(True)
                    self.state.add_log("INFO", "ConnMgr",
                                       "WebSocket reconnected")
                    # Re-subscribe all feeds
                    # (Subscriptions are stored in SharedState)
                except Exception as e:
                    self.state.add_log("ERROR", "ConnMgr",
                                       f"Reconnect failed: {e}")
            self._ws_stop.wait(5.0)

    # --------------------------------------------------------- tick callback
    def _on_tick_callback(self, tick_data: dict):
        """
        Callback invoked by the Breeze WebSocket thread.
        Parses tick and updates SharedState.
        """
        try:
            stock_code = tick_data.get("stock_code", "")
            strike_str = tick_data.get("strike_price", "0")
            right_raw = tick_data.get("right", "").lower()
            right = "call" if "call" in right_raw else "put"
            strike = float(strike_str)

            td = TickData(
                symbol=tick_data.get("symbol", ""),
                stock_code=stock_code,
                strike_price=strike,
                right=right,
                expiry_date=tick_data.get("expiry_date", ""),
                ltp=float(tick_data.get("ltp", 0)),
                bid=float(tick_data.get("best_bid_price", 0)),
                ask=float(tick_data.get("best_offer_price", 0)),
                open=float(tick_data.get("open", 0)),
                high=float(tick_data.get("high", 0)),
                low=float(tick_data.get("low", 0)),
                close=float(tick_data.get("previous_close", 0)),
                volume=int(tick_data.get("total_quantity_traded", 0)),
                oi=int(tick_data.get("open_interest", 0)),
                timestamp=time.time(),
            )

            self.state.update_tick(stock_code, strike, right, td)

            # Update spot price if present in tick
            spot_str = tick_data.get("spot_price", "")
            if spot_str:
                self.state.set_spot(stock_code, float(spot_str))

        except Exception as e:
            self.state.add_log("ERROR", "ConnMgr",
                               f"Tick parse error: {e}", tick_data)

    # ------------------------------------------------ rate-limited API proxy
    @rate_limited
    def subscribe(self, stock_code: str, expiry_date: str,
                  strike_price: float, right: str) -> dict:
        """Subscribe to a specific option feed."""
        return self._breeze.subscribe_feeds(
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
    def unsubscribe(self, stock_code: str, expiry_date: str,
                    strike_price: float, right: str) -> dict:
        return self._breeze.unsubscribe_feeds(
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
    def get_option_chain(self, stock_code: str,
                         expiry_date: str) -> list[dict]:
        """Fetch the full option chain (both CE & PE)."""
        resp = self._breeze.get_option_chain_quotes(
            stock_code=stock_code,
            exchange_code="NFO",
            product_type="options",
            expiry_date=expiry_date,
        )
        if resp.get("Error"):
            self.state.add_log("ERROR", "ConnMgr",
                               f"Option chain fetch error: {resp['Error']}")
            return []
        return resp.get("Success", []) or []

    @rate_limited
    def get_spot_quote(self, stock_code: str) -> float:
        """Get the current spot price via REST."""
        resp = self._breeze.get_quotes(
            stock_code=stock_code,
            exchange_code="NSE",
            expiry_date="",
            product_type="cash",
            right="others",
            strike_price="0",
        )
        try:
            return float(resp["Success"][0]["ltp"])
        except (KeyError, IndexError, TypeError, ValueError):
            return 0.0

    @rate_limited
    def place_order(self, stock_code: str, action: str,
                    strike_price: float, right: str,
                    expiry_date: str, quantity: int,
                    price: float,
                    order_type: str = "limit") -> dict:
        """Place an NFO options order."""
        resp = self._breeze.place_order(
            stock_code=stock_code,
            exchange_code="NFO",
            product="options",
            action=action.lower(),
            order_type=order_type.lower(),
            stoploss="",
            quantity=str(quantity),
            price=str(round(price, 2)),
            validity="day",
            validity_date="",
            disclosed_quantity="0",
            expiry_date=expiry_date,
            right=right.lower(),
            strike_price=str(int(strike_price)),
            segment="N",
            settlement_id="",
        )
        self.state.add_log(
            "INFO", "ConnMgr",
            f"Order placed: {action} {stock_code} {strike_price}"
            f"{right[0].upper()}E @{price} qty={quantity}",
            resp,
        )
        return resp

    @rate_limited
    def modify_order(self, order_id: str, price: float,
                     quantity: int | None = None,
                     order_type: str = "limit") -> dict:
        kwargs: dict[str, Any] = {
            "order_id": order_id,
            "exchange_code": "NFO",
            "order_type": order_type,
            "price": str(round(price, 2)),
        }
        if quantity is not None:
            kwargs["quantity"] = str(quantity)
        return self._breeze.modify_order(**kwargs)

    @rate_limited
    def cancel_order(self, order_id: str) -> dict:
        return self._breeze.cancel_order(
            order_id=order_id,
            exchange_code="NFO",
        )

    @rate_limited
    def get_order_status(self, order_id: str) -> dict | None:
        resp = self._breeze.get_order_detail(
            exchange_code="NFO",
            order_id=order_id,
        )
        try:
            items = resp.get("Success", [])
            if items and isinstance(items, list):
                return items[0]
        except Exception:
            pass
        return None

    @rate_limited
    def get_positions(self) -> list[dict]:
        resp = self._breeze.get_portfolio_positions()
        return resp.get("Success", []) or []

    # ------------------------------------------------- convenience
    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def breeze(self) -> Any:
        return self._breeze
