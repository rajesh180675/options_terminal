# ═══════════════════════════════════════════════════════════════
# FILE: mock_breeze.py
# ═══════════════════════════════════════════════════════════════
"""
Complete MockBreeze with:
  - GBM underlying random walk
  - Black-Scholes option pricing
  - Realistic order fill simulation
  - Same method signatures as BreezeConnect SDK v1.0.52+

CRITICAL: Method signatures match the LIVE Breeze SDK exactly.
"""

import time
import math
import random
import threading
import uuid
from datetime import datetime
from typing import Callable, Optional, Dict, List

from greeks_engine import BlackScholes, compute_time_to_expiry
from models import OptionRight
from config import Config
from utils import safe_float


class _MockOrder:
    def __init__(self, order_id, stock_code, exchange_code, strike_price,
                 right, expiry_date, action, order_type, price, quantity):
        self.order_id = order_id
        self.stock_code = stock_code
        self.exchange_code = exchange_code
        self.strike_price = float(strike_price)
        self.right = right
        self.expiry_date = expiry_date
        self.action = action  # "buy" or "sell"
        self.order_type = order_type  # "limit" or "market"
        self.price = float(price)
        self.quantity = int(quantity)
        self.status = "pending"
        self.fill_price = 0.0
        self.timestamp = datetime.now().isoformat()


class MockBreeze:
    """
    Drop-in replacement for BreezeConnect.
    Every public method has the EXACT same signature as the live SDK.
    """

    def __init__(self, api_key: str = "mock_key"):
        self.api_key = api_key
        self._session_active = False
        self._ws_running = False
        self._ws_thread: Optional[threading.Thread] = None
        self._subscriptions: List[Dict] = []
        self._orders: Dict[str, _MockOrder] = {}
        self._positions: Dict[str, dict] = {}

        # Underlying GBM simulation
        self._spot: Dict[str, float] = {
            "NIFTY": 24250.0,
            "CNXBAN": 52500.0,
        }
        self._vol = 0.14
        self._dt = 0.5  # seconds between ticks
        self._r = Config.RISK_FREE_RATE

        self.on_ticks: Optional[Callable] = None
        self._lock = threading.Lock()

    # ── Session (matches BreezeConnect.generate_session) ─────

    def generate_session(self, api_secret: str, session_token: str) -> dict:
        self._session_active = True
        return {
            "Success": {"session_token": "mock_session_token_abc123"},
            "Status": 200,
            "Error": None,
        }

    # ── WebSocket (matches BreezeConnect.ws_connect) ─────────

    def ws_connect(self):
        if self._ws_running:
            return
        self._ws_running = True
        self._ws_thread = threading.Thread(
            target=self._ws_loop, daemon=True, name="MockWS"
        )
        self._ws_thread.start()

    def ws_disconnect(self):
        self._ws_running = False

    def _ws_loop(self):
        while self._ws_running:
            self._evolve_spot()
            with self._lock:
                subs = list(self._subscriptions)
            for sub in subs:
                tick = self._make_tick(sub)
                if tick and self.on_ticks:
                    try:
                        self.on_ticks(tick)
                    except Exception:
                        pass
            self._try_fill_pending()
            time.sleep(self._dt)

    def _evolve_spot(self):
        with self._lock:
            for code in self._spot:
                S = self._spot[code]
                dt_y = self._dt / (365.25 * 24 * 3600)
                drift = (self._r - 0.5 * self._vol ** 2) * dt_y
                diff = self._vol * math.sqrt(dt_y) * random.gauss(0, 1)
                self._spot[code] = S * math.exp(drift + diff)

    def _make_tick(self, sub: dict) -> Optional[dict]:
        stock = sub.get("stock_code", "NIFTY")
        strike = float(sub.get("strike_price", "0") or "0")
        right_str = sub.get("right", "call").lower()
        expiry = sub.get("expiry_date", "")
        right = OptionRight.CALL if right_str == "call" else OptionRight.PUT

        with self._lock:
            S = self._spot.get(stock, 24250.0)

        if strike <= 0:
            return None

        T = compute_time_to_expiry(expiry)
        iv = max(self._vol + random.uniform(-0.01, 0.01), 0.05)
        bs = max(BlackScholes.price(S, strike, T, self._r, iv, right), 0.05)

        spread = max(0.05, bs * 0.002)
        bid = round(max(bs - spread, 0.05), 2)
        ask = round(bs + spread, 2)
        ltp = round(bs + random.uniform(-spread, spread), 2)
        ltp = max(ltp, 0.05)

        # Tick format matches Breeze WebSocket callback exactly
        return {
            "stock_code": stock,
            "exchange_code": "NFO",
            "product_type": "options",
            "strike_price": str(int(strike)),
            "right": "Call" if right == OptionRight.CALL else "Put",
            "expiry_date": expiry,
            "ltp": str(round(ltp, 2)),
            "best_bid_price": str(round(bid, 2)),
            "best_offer_price": str(round(ask, 2)),
            "ltq": str(random.randint(25, 500)),
            "ltt": datetime.now().strftime("%d-%b-%Y %H:%M:%S"),
            "total_quantity_traded": str(random.randint(10000, 500000)),
            "open_interest": str(random.randint(100000, 5000000)),
        }

    # ── subscribe_feeds (EXACT Breeze signature) ─────────────

    def subscribe_feeds(
        self,
        exchange_code: str = "NFO",
        stock_code: str = "",
        product_type: str = "options",
        expiry_date: str = "",
        strike_price: str = "",
        right: str = "others",
        get_exchange_quotes: bool = True,
        get_market_depth: bool = False,
    ) -> dict:
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
            existing_keys = [
                f"{s['stock_code']}|{s['strike_price']}|{s['right']}|{s['expiry_date']}"
                for s in self._subscriptions
            ]
            key = f"{stock_code}|{strike_price}|{right}|{expiry_date}"
            if key not in existing_keys:
                self._subscriptions.append(sub)
        return {"Success": None, "Status": 200, "Error": None}

    def unsubscribe_feeds(
        self,
        exchange_code: str = "NFO",
        stock_code: str = "",
        product_type: str = "options",
        expiry_date: str = "",
        strike_price: str = "",
        right: str = "others",
        get_exchange_quotes: bool = True,
        get_market_depth: bool = False,
    ) -> dict:
        with self._lock:
            self._subscriptions = [
                s for s in self._subscriptions
                if not (
                    s.get("stock_code") == stock_code
                    and s.get("strike_price") == strike_price
                    and s.get("right") == right
                )
            ]
        return {"Success": None, "Status": 200, "Error": None}

    # ── get_option_chain_quotes (EXACT Breeze signature) ─────

    def get_option_chain_quotes(
        self,
        stock_code: str = "",
        exchange_code: str = "NFO",
        product_type: str = "options",
        expiry_date: str = "",
        right: str = "others",
    ) -> dict:
        with self._lock:
            S = self._spot.get(stock_code, 24250.0)

        gap = Config.strike_gap(stock_code)
        atm = round(S / gap) * gap
        strikes = [atm + i * gap for i in range(-10, 11)]
        T = compute_time_to_expiry(expiry_date)

        rights_to_gen = (
            [OptionRight.CALL, OptionRight.PUT]
            if right == "others"
            else [OptionRight.CALL if right.lower() == "call" else OptionRight.PUT]
        )

        chain = []
        for strike in strikes:
            for opt_right in rights_to_gen:
                iv = max(self._vol + random.uniform(-0.02, 0.02), 0.05)
                price = max(BlackScholes.price(S, strike, T, self._r, iv, opt_right), 0.05)
                spread = max(0.05, price * 0.003)

                # Key names match LIVE Breeze response
                chain.append({
                    "stock_code": stock_code,
                    "strike_price": str(int(strike)),
                    "right": "Call" if opt_right == OptionRight.CALL else "Put",
                    "expiry_date": expiry_date,
                    "ltp": str(round(price, 2)),
                    "best_bid_price": str(round(max(price - spread, 0.05), 2)),
                    "best_offer_price": str(round(price + spread, 2)),
                    "open": str(round(price * 1.02, 2)),
                    "high": str(round(price * 1.05, 2)),
                    "low": str(round(price * 0.95, 2)),
                    "close": str(round(price * 0.99, 2)),
                    "volume": str(random.randint(10000, 500000)),
                    "open_interest": str(random.randint(100000, 5000000)),
                })
        return {"Success": chain, "Status": 200, "Error": None}

    # ── place_order (EXACT Breeze SDK signature) ─────────────
    # Breeze SDK source: place_order(stock_code, exchange_code, product,
    #   action, order_type, stoploss, quantity, price, validity,
    #   validity_date, disclosed_quantity, expiry_date, right,
    #   strike_price, ...)

    def place_order(
        self,
        stock_code: str = "",
        exchange_code: str = "NFO",
        product: str = "options",
        action: str = "sell",
        order_type: str = "limit",
        stoploss: str = "",
        quantity: str = "25",
        price: str = "0",
        validity: str = "day",
        validity_date: str = "",
        disclosed_quantity: str = "0",
        expiry_date: str = "",
        right: str = "call",
        strike_price: str = "0",
        user_remark: str = "",
        order_type_fresh: str = "",
        order_rate_fresh: str = "",
        settlement_id: str = "",
        order_segment: str = "",
    ) -> dict:
        oid = f"MOCK{uuid.uuid4().hex[:8].upper()}"
        order = _MockOrder(
            order_id=oid,
            stock_code=stock_code,
            exchange_code=exchange_code,
            strike_price=strike_price,
            right=right,
            expiry_date=expiry_date,
            action=action.lower(),
            order_type=order_type.lower(),
            price=safe_float(price),
            quantity=int(quantity),
        )
        with self._lock:
            self._orders[oid] = order
        return {"Success": {"order_id": oid}, "Status": 200, "Error": None}

    # ── cancel_order ─────────────────────────────────────────

    def cancel_order(
        self,
        exchange_code: str = "NFO",
        order_id: str = "",
    ) -> dict:
        with self._lock:
            if order_id in self._orders:
                self._orders[order_id].status = "cancelled"
                return {"Success": {"order_id": order_id}, "Status": 200, "Error": None}
        return {"Success": None, "Status": 400, "Error": "Order not found"}

    # ── modify_order ─────────────────────────────────────────

    def modify_order(
        self,
        order_id: str = "",
        exchange_code: str = "NFO",
        order_type: str = "limit",
        stoploss: str = "",
        quantity: str = "0",
        price: str = "0",
        validity: str = "day",
        validity_date: str = "",
        disclosed_quantity: str = "0",
    ) -> dict:
        with self._lock:
            if order_id in self._orders:
                self._orders[order_id].price = safe_float(price)
                if int(quantity) > 0:
                    self._orders[order_id].quantity = int(quantity)
                self._orders[order_id].status = "pending"
                return {"Success": {"order_id": order_id}, "Status": 200, "Error": None}
        return {"Success": None, "Status": 400, "Error": "Order not found"}

    # ── get_order_detail ─────────────────────────────────────

    def get_order_detail(
        self,
        exchange_code: str = "NFO",
        order_id: str = "",
    ) -> dict:
        with self._lock:
            if order_id in self._orders:
                o = self._orders[order_id]
                return {
                    "Success": [{
                        "order_id": o.order_id,
                        "status": o.status,
                        "action": o.action,
                        "quantity": str(o.quantity),
                        "price": str(o.price),
                        "average_price": str(o.fill_price) if o.fill_price > 0 else str(o.price),
                        "stock_code": o.stock_code,
                        "strike_price": str(o.strike_price),
                        "right": o.right,
                        "expiry_date": o.expiry_date,
                    }],
                    "Status": 200,
                    "Error": None,
                }
        return {"Success": None, "Status": 400, "Error": "Order not found"}

    # ── get_portfolio_positions ──────────────────────────────

    def get_portfolio_positions(self) -> dict:
        with self._lock:
            pos = list(self._positions.values())
        return {"Success": pos if pos else [], "Status": 200, "Error": None}

    # ── Internal fill engine ─────────────────────────────────

    def _try_fill_pending(self):
        with self._lock:
            for oid, order in list(self._orders.items()):
                if order.status != "pending":
                    continue

                S = self._spot.get(order.stock_code, 24250.0)
                right = OptionRight.CALL if order.right.lower() == "call" else OptionRight.PUT
                T = compute_time_to_expiry(order.expiry_date)
                fair = max(BlackScholes.price(S, order.strike_price, T, self._r, self._vol, right), 0.05)

                if order.order_type == "market":
                    slip = random.uniform(0, 0.3)
                    order.fill_price = round(
                        fair + slip if order.action == "buy" else fair - slip, 2
                    )
                    order.fill_price = max(order.fill_price, 0.05)
                    order.status = "filled"
                    self._update_mock_position(order)
                    continue

                # Limit order fill logic
                if order.action == "sell":
                    if order.price >= fair * 0.97:
                        slip = random.uniform(-0.15, 0.15)
                        order.fill_price = round(max(order.price + slip, 0.05), 2)
                        order.status = "filled"
                        self._update_mock_position(order)
                elif order.action == "buy":
                    if order.price <= fair * 1.03:
                        slip = random.uniform(-0.15, 0.15)
                        order.fill_price = round(max(order.price + slip, 0.05), 2)
                        order.status = "filled"
                        self._update_mock_position(order)

    def _update_mock_position(self, order: _MockOrder):
        key = f"{order.stock_code}|{int(order.strike_price)}|{order.right}|{order.expiry_date}"
        if key not in self._positions:
            self._positions[key] = {
                "stock_code": order.stock_code,
                "strike_price": str(int(order.strike_price)),
                "right": order.right,
                "expiry_date": order.expiry_date,
                "quantity": 0,
                "average_price": 0.0,
            }
        pos = self._positions[key]
        if order.action == "sell":
            pos["quantity"] -= order.quantity
        else:
            pos["quantity"] += order.quantity
        pos["average_price"] = order.fill_price
        if pos["quantity"] == 0:
            del self._positions[key]

    # ── Spot helper (mock only) ──────────────────────────────

    def get_spot_price(self, stock_code: str) -> float:
        with self._lock:
            return self._spot.get(stock_code, 24250.0)

    # ── get_quotes (single instrument) ───────────────────────

    def get_quotes(
        self,
        stock_code: str = "",
        exchange_code: str = "NFO",
        expiry_date: str = "",
        product_type: str = "options",
        right: str = "call",
        strike_price: str = "0",
    ) -> dict:
        S = self.get_spot_price(stock_code)
        opt_right = OptionRight.CALL if right.lower() == "call" else OptionRight.PUT
        T = compute_time_to_expiry(expiry_date)
        price = max(BlackScholes.price(S, float(strike_price), T, self._r, self._vol, opt_right), 0.05)
        spread = max(0.05, price * 0.003)
        return {
            "Success": [{
                "stock_code": stock_code,
                "strike_price": strike_price,
                "right": right,
                "expiry_date": expiry_date,
                "ltp": str(round(price, 2)),
                "best_bid_price": str(round(max(price - spread, 0.05), 2)),
                "best_offer_price": str(round(price + spread, 2)),
            }],
            "Status": 200,
            "Error": None,
        }
