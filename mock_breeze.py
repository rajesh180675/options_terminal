"""
mock_breeze.py — Full MockBreeze class that simulates BreezeConnect for paper trading.
Generates realistic tick data using Black-Scholes pricing.
"""

from __future__ import annotations

import math
import random
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from greeks_engine import BlackScholes


class MockOrder:
    def __init__(self, order_id: str, stock_code: str, action: str,
                 strike_price: float, right: str, quantity: int,
                 price: float, order_type: str, expiry_date: str):
        self.order_id = order_id
        self.stock_code = stock_code
        self.action = action
        self.strike_price = strike_price
        self.right = right
        self.quantity = quantity
        self.price = price
        self.order_type = order_type
        self.expiry_date = expiry_date
        self.status = "pending"
        self.filled_price = 0.0
        self.filled_qty = 0
        self.created_at = datetime.now(timezone.utc).isoformat()


class MockBreeze:
    """
    Complete mock of BreezeConnect for testing.
    Simulates live market data, order fills, and position tracking.
    """

    def __init__(self, api_key: str = "mock_key"):
        self.api_key = api_key
        self.session_generated = False
        self.ws_connected = False
        self.on_ticks: Callable | None = None

        # Simulated market state
        self._spot_prices: dict[str, float] = {
            "NIFTY": 24500.0,
            "CNXBAN": 52000.0,
        }
        self._volatility: dict[str, float] = {
            "NIFTY": 0.13,
            "CNXBAN": 0.15,
        }
        self._subscriptions: list[dict] = []
        self._orders: dict[str, MockOrder] = {}
        self._positions: list[dict] = []

        self._ws_thread: threading.Thread | None = None
        self._ws_stop = threading.Event()
        self._lock = threading.Lock()

        # Drift parameters for realistic simulation
        self._drift_speed = 0.0003  # per tick
        self._tick_interval = 0.5   # seconds between ticks

    # ------------------------------------------------------- session
    def generate_session(self, api_secret: str = "", session_token: str = "") -> dict:
        self.session_generated = True
        return {
            "Success": {"session_token": "mock_session_" + uuid.uuid4().hex[:8]},
            "Status": 200,
            "Error": None,
        }

    # --------------------------------------------------- websocket
    def connect(self):
        """Start the mock WebSocket tick generator."""
        if self._ws_thread and self._ws_thread.is_alive():
            return
        self.ws_connected = True
        self._ws_stop.clear()
        self._ws_thread = threading.Thread(target=self._tick_generator,
                                           daemon=True, name="MockWS")
        self._ws_thread.start()

    def disconnect(self):
        self._ws_stop.set()
        self.ws_connected = False
        if self._ws_thread:
            self._ws_thread.join(timeout=5)

    def _tick_generator(self):
        """Generates realistic ticks for all subscribed instruments."""
        while not self._ws_stop.is_set():
            # Random-walk the spot prices
            with self._lock:
                for code in self._spot_prices:
                    drift = random.gauss(0, self._drift_speed)
                    self._spot_prices[code] *= (1 + drift)

                # Also slight vol changes
                for code in self._volatility:
                    vol_drift = random.gauss(0, 0.0001)
                    self._volatility[code] = max(
                        0.05, min(0.50, self._volatility[code] + vol_drift)
                    )

            # Generate ticks for each subscription
            with self._lock:
                subs = list(self._subscriptions)

            for sub in subs:
                tick = self._generate_tick(sub)
                if tick and self.on_ticks:
                    try:
                        self.on_ticks(tick)
                    except Exception:
                        pass

            # Also attempt to fill pending orders
            self._try_fill_orders()

            self._ws_stop.wait(self._tick_interval)

    def _generate_tick(self, sub: dict) -> dict | None:
        stock_code = sub.get("stock_code", "NIFTY")
        strike = float(sub.get("strike_price", 0))
        right = sub.get("right", "call").lower()
        expiry = sub.get("expiry_date", "")

        with self._lock:
            spot = self._spot_prices.get(stock_code, 24500.0)
            vol = self._volatility.get(stock_code, 0.15)

        # Calculate time to expiry
        tte = self._calc_tte(expiry)
        if tte <= 0:
            tte = 1 / (365.25 * 24 * 60)

        # BS price
        r = 0.07
        if right == "call":
            theo = BlackScholes.call_price(spot, strike, tte, r, vol)
        else:
            theo = BlackScholes.put_price(spot, strike, tte, r, vol)

        # Add noise for bid-ask spread
        spread = max(0.05, theo * 0.002)
        noise = random.gauss(0, spread * 0.3)
        ltp = max(0.05, theo + noise)
        bid = max(0.05, ltp - spread / 2)
        ask = ltp + spread / 2

        return {
            "symbol": f"{stock_code}{strike}{right[0].upper()}E",
            "stock_code": stock_code,
            "strike_price": str(strike),
            "right": "Call" if right == "call" else "Put",
            "expiry_date": expiry,
            "ltp": round(ltp, 2),
            "best_bid_price": round(bid, 2),
            "best_offer_price": round(ask, 2),
            "best_bid_quantity": random.randint(50, 5000),
            "best_offer_quantity": random.randint(50, 5000),
            "open": round(ltp * 0.98, 2),
            "high": round(ltp * 1.03, 2),
            "low": round(ltp * 0.97, 2),
            "previous_close": round(ltp * 0.99, 2),
            "total_quantity_traded": random.randint(10000, 500000),
            "open_interest": random.randint(100000, 5000000),
            "spot_price": round(spot, 2),
        }

    def _calc_tte(self, expiry_str: str) -> float:
        try:
            if "T" in expiry_str:
                expiry = datetime.fromisoformat(expiry_str.replace("Z", "+00:00"))
            else:
                expiry = datetime.strptime(expiry_str, "%Y-%m-%d").replace(
                    hour=15, minute=30, tzinfo=timezone.utc
                )
            now = datetime.now(timezone.utc)
            return max((expiry - now).total_seconds() / (365.25 * 24 * 3600), 0)
        except Exception:
            return 7 / 365.25

    # ------------------------------------------------------ subscribe
    def subscribe_feeds(self, exchange_code: str = "NFO",
                        stock_code: str = "NIFTY",
                        product_type: str = "options",
                        expiry_date: str = "",
                        strike_price: str = "0",
                        right: str = "call",
                        get_exchange_quotes: bool = True,
                        get_market_depth: bool = False,
                        stock_token: str = "",
                        interval: str = "") -> dict:
        sub = {
            "exchange_code": exchange_code,
            "stock_code": stock_code,
            "product_type": product_type,
            "expiry_date": expiry_date,
            "strike_price": strike_price,
            "right": right,
        }
        with self._lock:
            # Avoid duplicate subscriptions
            if sub not in self._subscriptions:
                self._subscriptions.append(sub)
        return {"Success": "Feed subscribed", "Status": 200, "Error": None}

    def unsubscribe_feeds(self, exchange_code: str = "NFO",
                          stock_code: str = "NIFTY",
                          product_type: str = "options",
                          expiry_date: str = "",
                          strike_price: str = "0",
                          right: str = "call",
                          get_exchange_quotes: bool = True,
                          get_market_depth: bool = False,
                          stock_token: str = "",
                          interval: str = "") -> dict:
        sub = {
            "exchange_code": exchange_code,
            "stock_code": stock_code,
            "product_type": product_type,
            "expiry_date": expiry_date,
            "strike_price": strike_price,
            "right": right,
        }
        with self._lock:
            self._subscriptions = [s for s in self._subscriptions if s != sub]
        return {"Success": "Feed unsubscribed", "Status": 200, "Error": None}

    # ------------------------------------------------------ option chain
    def get_option_chain_quotes(self, stock_code: str = "NIFTY",
                                exchange_code: str = "NFO",
                                product_type: str = "options",
                                expiry_date: str = "",
                                right: str = "",
                                strike_price: str = "") -> dict:
        """Return a simulated option chain centered around spot."""
        with self._lock:
            spot = self._spot_prices.get(stock_code, 24500.0)
            vol = self._volatility.get(stock_code, 0.15)

        gap = 50 if stock_code == "NIFTY" else 100
        atm = round(spot / gap) * gap
        strikes = [atm + i * gap for i in range(-15, 16)]
        tte = self._calc_tte(expiry_date)
        r = 0.07

        records = []
        rights = ["call", "put"] if not right else [right.lower()]

        for stk in strikes:
            for rt in rights:
                if strike_price and float(strike_price) != stk:
                    continue
                if rt == "call":
                    theo = BlackScholes.call_price(spot, stk, tte, r, vol)
                else:
                    theo = BlackScholes.put_price(spot, stk, tte, r, vol)

                spread = max(0.05, theo * 0.002)
                ltp = max(0.05, theo + random.gauss(0, spread * 0.2))
                bid = max(0.05, ltp - spread / 2)
                ask = ltp + spread / 2

                records.append({
                    "stock_code": stock_code,
                    "exchange_code": exchange_code,
                    "product_type": "options",
                    "expiry_date": expiry_date,
                    "strike_price": str(stk),
                    "right": "Call" if rt == "call" else "Put",
                    "ltp": round(ltp, 2),
                    "best_bid_price": round(bid, 2),
                    "best_offer_price": round(ask, 2),
                    "open": round(ltp * 0.98, 2),
                    "high": round(ltp * 1.03, 2),
                    "low": round(ltp * 0.97, 2),
                    "previous_close": round(ltp * 0.99, 2),
                    "open_interest": str(random.randint(100000, 5000000)),
                    "total_quantity_traded": str(random.randint(10000, 500000)),
                    "spot_price": str(round(spot, 2)),
                })

        return {"Success": records, "Status": 200, "Error": None}

    # --------------------------------------------------------- orders
    def place_order(self, stock_code: str = "NIFTY",
                    exchange_code: str = "NFO",
                    product: str = "options",
                    action: str = "sell",
                    order_type: str = "limit",
                    stoploss: str = "",
                    quantity: str = "50",
                    price: str = "0",
                    validity: str = "day",
                    validity_date: str = "",
                    disclosed_quantity: str = "0",
                    expiry_date: str = "",
                    right: str = "call",
                    strike_price: str = "0",
                    segment: str = "N",
                    settlement_id: str = "",
                    order_type_fresh: str = "",
                    order_rate_fresh: str = "",
                    user_remark: str = "") -> dict:
        oid = "MOCK_" + uuid.uuid4().hex[:10].upper()
        order = MockOrder(
            order_id=oid,
            stock_code=stock_code,
            action=action.lower(),
            strike_price=float(strike_price),
            right=right.lower(),
            quantity=int(quantity),
            price=float(price),
            order_type=order_type.lower(),
            expiry_date=expiry_date,
        )

        if order_type.lower() == "market":
            # Immediate fill at simulated price
            with self._lock:
                spot = self._spot_prices.get(stock_code, 24500.0)
                vol = self._volatility.get(stock_code, 0.15)
            tte = self._calc_tte(expiry_date)
            if right.lower() == "call":
                theo = BlackScholes.call_price(spot, float(strike_price), tte, 0.07, vol)
            else:
                theo = BlackScholes.put_price(spot, float(strike_price), tte, 0.07, vol)

            slippage = theo * 0.001 * (1 if action.lower() == "buy" else -1)
            order.filled_price = round(max(0.05, theo + slippage), 2)
            order.filled_qty = order.quantity
            order.status = "filled"
        else:
            order.status = "placed"

        with self._lock:
            self._orders[oid] = order

        return {
            "Success": {"order_id": oid},
            "Status": 200,
            "Error": None,
        }

    def modify_order(self, order_id: str, exchange_code: str = "NFO",
                     order_type: str = "limit", stoploss: str = "",
                     quantity: str = "", price: str = "",
                     validity: str = "day",
                     disclosed_quantity: str = "0") -> dict:
        with self._lock:
            order = self._orders.get(order_id)
            if not order:
                return {"Success": None, "Status": 404, "Error": "Order not found"}
            if order.status in ("filled", "cancelled"):
                return {"Success": None, "Status": 400, "Error": "Cannot modify"}
            if price:
                order.price = float(price)
            if quantity:
                order.quantity = int(quantity)
            order.order_type = order_type.lower()
        return {"Success": {"order_id": order_id}, "Status": 200, "Error": None}

    def cancel_order(self, order_id: str,
                     exchange_code: str = "NFO") -> dict:
        with self._lock:
            order = self._orders.get(order_id)
            if not order:
                return {"Success": None, "Status": 404, "Error": "Order not found"}
            if order.status in ("filled", "cancelled"):
                return {"Success": None, "Status": 400, "Error": "Cannot cancel"}
            order.status = "cancelled"
        return {"Success": {"order_id": order_id}, "Status": 200, "Error": None}

    def get_order_detail(self, exchange_code: str = "NFO",
                         order_id: str = "") -> dict:
        with self._lock:
            order = self._orders.get(order_id)
            if not order:
                return {"Success": None, "Status": 404, "Error": "Not found"}
            return {
                "Success": [{
                    "order_id": order.order_id,
                    "status": order.status,
                    "action": order.action,
                    "stock_code": order.stock_code,
                    "strike_price": str(order.strike_price),
                    "right": order.right,
                    "quantity": str(order.quantity),
                    "price": str(order.price),
                    "filled_price": str(order.filled_price),
                    "filled_quantity": str(order.filled_qty),
                    "order_type": order.order_type,
                    "expiry_date": order.expiry_date,
                }],
                "Status": 200,
                "Error": None,
            }

    def get_portfolio_positions(self) -> dict:
        """Return currently tracked positions."""
        with self._lock:
            return {
                "Success": list(self._positions),
                "Status": 200,
                "Error": None,
            }

    def get_quotes(self, stock_code: str = "NIFTY",
                   exchange_code: str = "NSE",
                   expiry_date: str = "",
                   product_type: str = "cash",
                   right: str = "others",
                   strike_price: str = "0") -> dict:
        """Return spot quote for the underlying."""
        with self._lock:
            spot = self._spot_prices.get(stock_code, 24500.0)
        return {
            "Success": [{
                "stock_code": stock_code,
                "ltp": str(round(spot, 2)),
                "best_bid_price": str(round(spot - 0.5, 2)),
                "best_offer_price": str(round(spot + 0.5, 2)),
            }],
            "Status": 200,
            "Error": None,
        }

    # ----------------------------------------------------- order fill sim
    def _try_fill_orders(self):
        """Attempt to fill pending limit orders based on simulated market."""
        with self._lock:
            for oid, order in self._orders.items():
                if order.status != "placed":
                    continue
                spot = self._spot_prices.get(order.stock_code, 24500.0)
                vol = self._volatility.get(order.stock_code, 0.15)
                tte = self._calc_tte(order.expiry_date)
                if order.right == "call":
                    market = BlackScholes.call_price(
                        spot, order.strike_price, tte, 0.07, vol
                    )
                else:
                    market = BlackScholes.put_price(
                        spot, order.strike_price, tte, 0.07, vol
                    )

                # Simulate fill with ~70% probability if price is close
                should_fill = False
                if order.action == "sell" and order.price <= market * 1.005:
                    should_fill = random.random() < 0.7
                elif order.action == "buy" and order.price >= market * 0.995:
                    should_fill = random.random() < 0.7

                if should_fill:
                    order.status = "filled"
                    order.filled_price = round(
                        order.price + random.gauss(0, 0.05), 2
                    )
                    order.filled_qty = order.quantity
