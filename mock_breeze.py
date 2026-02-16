# ═══════════════════════════════════════════════════════════════
# FILE: mock_breeze.py
# ═══════════════════════════════════════════════════════════════
"""
Complete BreezeConnect mock that simulates:
  • Session generation
  • WebSocket tick generation (random-walk underlying + BS option prices)
  • Option chain snapshot
  • Order placement / cancellation / modification with realistic fills
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
from utils import LOG, breeze_date, next_weekly_expiry, safe_float


class MockOrder:
    def __init__(self, order_id, stock_code, exchange_code, strike_price,
                 right, expiry_date, action, order_type, price, quantity):
        self.order_id = order_id
        self.stock_code = stock_code
        self.exchange_code = exchange_code
        self.strike_price = float(strike_price)
        self.right = right
        self.expiry_date = expiry_date
        self.action = action
        self.order_type = order_type
        self.price = float(price)
        self.quantity = int(quantity)
        self.status = "pending"
        self.fill_price = 0.0
        self.timestamp = datetime.now().isoformat()


class MockBreeze:
    """
    Full simulation of BreezeConnect for offline development and testing.
    Generates a GBM random walk for the underlying and prices options via BS.
    """

    def __init__(self, api_key: str = "mock_key"):
        self.api_key = api_key
        self._session_active = False
        self._ws_running = False
        self._ws_thread: Optional[threading.Thread] = None
        self._subscriptions: List[Dict] = []
        self._orders: Dict[str, MockOrder] = {}
        self._positions: Dict[str, dict] = {}

        # Underlying simulation
        self._spot: Dict[str, float] = {
            "NIFTY": 23500.0,
            "CNXBAN": 50000.0,
            "BANKNIFTY": 50000.0,
        }
        self._vol = 0.15  # annualised vol for GBM
        self._dt = 0.5    # tick interval in seconds
        self._r = Config.RISK_FREE_RATE

        # Callback
        self.on_ticks: Optional[Callable] = None

        self._lock = threading.Lock()

    # ── Session ──────────────────────────────────────────────

    def generate_session(self, api_secret: str, session_token: str) -> dict:
        LOG.info("[MockBreeze] Session generated (mock)")
        self._session_active = True
        return {"Success": {"session_token": "mock_session_token"}, "Status": 200, "Error": None}

    # ── WebSocket ────────────────────────────────────────────

    def ws_connect(self):
        if self._ws_running:
            return
        self._ws_running = True
        self._ws_thread = threading.Thread(
            target=self._ws_loop, daemon=True, name="MockWS"
        )
        self._ws_thread.start()
        LOG.info("[MockBreeze] WebSocket connected (mock)")

    def ws_disconnect(self):
        self._ws_running = False
        LOG.info("[MockBreeze] WebSocket disconnected (mock)")

    def _ws_loop(self):
        """Generates ticks every self._dt seconds for all subscriptions."""
        while self._ws_running:
            self._evolve_spot()
            with self._lock:
                subs = list(self._subscriptions)
            for sub in subs:
                tick = self._generate_tick(sub)
                if tick and self.on_ticks:
                    try:
                        self.on_ticks(tick)
                    except Exception as e:
                        LOG.error(f"[MockBreeze] on_ticks error: {e}")
            self._try_fill_orders()
            time.sleep(self._dt)

    def _evolve_spot(self):
        """GBM step for each underlying."""
        with self._lock:
            for code in self._spot:
                S = self._spot[code]
                dt_years = self._dt / (365.25 * 24 * 3600)
                drift = (self._r - 0.5 * self._vol**2) * dt_years
                diffusion = self._vol * math.sqrt(dt_years) * random.gauss(0, 1)
                self._spot[code] = S * math.exp(drift + diffusion)

    def _generate_tick(self, sub: dict) -> Optional[dict]:
        stock_code = sub.get("stock_code", "NIFTY")
        strike = float(sub.get("strike_price", 0))
        right_str = sub.get("right", "call").lower()
        expiry = sub.get("expiry_date", "")
        right = OptionRight.CALL if "call" in right_str else OptionRight.PUT

        with self._lock:
            S = self._spot.get(stock_code, 23500.0)

        T = compute_time_to_expiry(expiry)
        iv = self._vol + random.uniform(-0.01, 0.01)
        iv = max(iv, 0.05)
        bs_price = BlackScholes.price(S, strike, T, self._r, iv, right)
        bs_price = max(bs_price, 0.05)

        spread = max(0.05, bs_price * 0.002)
        bid = round(bs_price - spread, 2)
        ask = round(bs_price + spread, 2)
        ltp = round(bs_price + random.uniform(-spread, spread), 2)
        ltp = max(ltp, 0.05)

        return {
            "stock_code": stock_code,
            "exchange_code": "NFO",
            "product_type": "options",
            "strike_price": str(strike),
            "right": "Call" if right == OptionRight.CALL else "Put",
            "expiry_date": expiry,
            "ltp": str(round(ltp, 2)),
            "best_bid_price": str(round(max(bid, 0.05), 2)),
            "best_offer_price": str(round(max(ask, 0.05), 2)),
            "ltq": str(random.randint(25, 500)),
            "ltt": datetime.now().strftime("%d-%b-%Y %H:%M:%S"),
            "total_quantity_traded": str(random.randint(10000, 500000)),
            "open_interest": str(random.randint(100000, 5000000)),
        }

    # ── Subscription ─────────────────────────────────────────

    def subscribe_feeds(
        self,
        exchange_code: str = "NFO",
        stock_code: str = "NIFTY",
        product_type: str = "options",
        expiry_date: str = "",
        strike_price: str = "",
        right: str = "call",
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
            self._subscriptions.append(sub)
        LOG.info(f"[MockBreeze] Subscribed: {stock_code} {strike_price} {right}")
        return {"Success": None, "Status": 200, "Error": None}

    def unsubscribe_feeds(self, **kwargs) -> dict:
        with self._lock:
            self._subscriptions = [
                s for s in self._subscriptions
                if not all(s.get(k) == v for k, v in kwargs.items() if k in s)
            ]
        return {"Success": None, "Status": 200, "Error": None}

    # ── Option chain ─────────────────────────────────────────

    def get_option_chain_quotes(
        self,
        stock_code: str = "NIFTY",
        exchange_code: str = "NFO",
        product_type: str = "options",
        expiry_date: str = "",
        right: str = "others",
    ) -> dict:
        with self._lock:
            S = self._spot.get(stock_code, 23500.0)

        gap = Config.strike_gap(stock_code)
        atm = round(S / gap) * gap
        strikes = [atm + i * gap for i in range(-10, 11)]
        T = compute_time_to_expiry(expiry_date)

        chain = []
        rights = (
            [OptionRight.CALL, OptionRight.PUT]
            if right == "others"
            else [OptionRight.CALL if "call" in right.lower() else OptionRight.PUT]
        )
        for strike in strikes:
            for opt_right in rights:
                iv = self._vol + random.uniform(-0.02, 0.02)
                iv = max(iv, 0.05)
                price = BlackScholes.price(S, strike, T, self._r, iv, opt_right)
                price = max(price, 0.05)
                spread = max(0.05, price * 0.003)
                chain.append({
                    "stock_code": stock_code,
                    "strike_price": str(strike),
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

    # ── Orders ───────────────────────────────────────────────

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
        settlement_id: str = "",
        order_segment: str = "N",
        order_rate_fresh: str = "",
    ) -> dict:
        oid = f"MOCK{uuid.uuid4().hex[:8].upper()}"
        order = MockOrder(
            order_id=oid,
            stock_code=stock_code,
            exchange_code=exchange_code,
            strike_price=strike_price,
            right=right,
            expiry_date=expiry_date,
            action=action,
            order_type=order_type,
            price=safe_float(price),
            quantity=int(quantity),
        )
        with self._lock:
            self._orders[oid] = order
        LOG.info(f"[MockBreeze] Order placed: {oid} {action} {stock_code} "
                 f"{strike_price}{right[0].upper()} @ {price}")
        return {"Success": {"order_id": oid}, "Status": 200, "Error": None}

    def cancel_order(self, exchange_code: str = "NFO", order_id: str = "") -> dict:
        with self._lock:
            if order_id in self._orders:
                self._orders[order_id].status = "cancelled"
                LOG.info(f"[MockBreeze] Order cancelled: {order_id}")
                return {"Success": {"order_id": order_id}, "Status": 200, "Error": None}
        return {"Success": None, "Status": 400, "Error": "Order not found"}

    def modify_order(
        self,
        order_id: str = "",
        exchange_code: str = "NFO",
        price: str = "0",
        quantity: str = "0",
        **kwargs,
    ) -> dict:
        with self._lock:
            if order_id in self._orders:
                self._orders[order_id].price = safe_float(price)
                if int(quantity) > 0:
                    self._orders[order_id].quantity = int(quantity)
                self._orders[order_id].status = "pending"
                LOG.info(f"[MockBreeze] Order modified: {order_id} new price={price}")
                return {"Success": {"order_id": order_id}, "Status": 200, "Error": None}
        return {"Success": None, "Status": 400, "Error": "Order not found"}

    def get_order_detail(self, exchange_code: str = "NFO", order_id: str = "") -> dict:
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
                        "fill_price": str(o.fill_price),
                        "stock_code": o.stock_code,
                        "strike_price": str(o.strike_price),
                        "right": o.right,
                        "expiry_date": o.expiry_date,
                    }],
                    "Status": 200,
                    "Error": None,
                }
        return {"Success": None, "Status": 400, "Error": "Order not found"}

    def get_portfolio_positions(self) -> dict:
        with self._lock:
            positions = list(self._positions.values())
        return {"Success": positions if positions else [], "Status": 200, "Error": None}

    # ── Internal fill engine ─────────────────────────────────

    def _try_fill_orders(self):
        with self._lock:
            for oid, order in list(self._orders.items()):
                if order.status not in ("pending",):
                    continue
                S = self._spot.get(order.stock_code, 23500.0)
                right = OptionRight.CALL if "call" in order.right.lower() else OptionRight.PUT
                T = compute_time_to_expiry(order.expiry_date)
                iv = self._vol
                fair = BlackScholes.price(S, order.strike_price, T, self._r, iv, right)
                fair = max(fair, 0.05)

                # Market orders fill immediately
                if order.order_type == "market":
                    slip = random.uniform(0, 0.5)
                    if order.action == "buy":
                        order.fill_price = round(fair + slip, 2)
                    else:
                        order.fill_price = round(fair - slip, 2)
                    order.status = "filled"
                    self._update_position(order)
                    continue

                # Limit orders: simulate fill probability
                if order.action == "sell" and order.price >= fair * 0.98:
                    slip = random.uniform(-0.2, 0.3)
                    order.fill_price = round(order.price + slip, 2)
                    order.status = "filled"
                    self._update_position(order)
                elif order.action == "buy" and order.price <= fair * 1.02:
                    slip = random.uniform(-0.3, 0.2)
                    order.fill_price = round(order.price + slip, 2)
                    order.status = "filled"
                    self._update_position(order)

    def _update_position(self, order: MockOrder):
        key = f"{order.stock_code}|{order.strike_price}|{order.right}|{order.expiry_date}"
        if key not in self._positions:
            self._positions[key] = {
                "stock_code": order.stock_code,
                "strike_price": str(order.strike_price),
                "right": order.right,
                "expiry_date": order.expiry_date,
                "quantity": 0,
                "avg_price": 0.0,
            }
        pos = self._positions[key]
        if order.action == "sell":
            pos["quantity"] -= order.quantity
        else:
            pos["quantity"] += order.quantity
        pos["avg_price"] = order.fill_price
        if pos["quantity"] == 0:
            del self._positions[key]

    # ── Helpers ──────────────────────────────────────────────

    def get_spot_price(self, stock_code: str) -> float:
        with self._lock:
            return self._spot.get(stock_code, 23500.0)

    def get_quotes(self, stock_code: str = "", exchange_code: str = "",
                   expiry_date: str = "", product_type: str = "",
                   right: str = "", strike_price: str = "") -> dict:
        """Single-quote fetch."""
        S = self.get_spot_price(stock_code)
        right_enum = OptionRight.CALL if "call" in right.lower() else OptionRight.PUT
        T = compute_time_to_expiry(expiry_date)
        iv = self._vol
        price = BlackScholes.price(S, float(strike_price), T, self._r, iv, right_enum)
        price = max(price, 0.05)
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
