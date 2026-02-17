# ═══════════════════════════════════════════════════════════════
# FILE: mock_breeze.py  (COMPLETE — every SDK method mocked)
# ═══════════════════════════════════════════════════════════════
"""
Mocks ALL 23 BreezeConnect SDK methods.
Previous version only mocked 10.
"""

import time, math, random, threading, uuid
from datetime import datetime, timedelta
from typing import Callable, Optional, Dict, List
from greeks_engine import BlackScholes, time_to_expiry
from models import OptionRight
from app_config import Config
from utils import safe_float, breeze_expiry


class _Order:
    __slots__ = (
        "order_id", "stock_code", "exchange_code", "strike_price",
        "right", "expiry_date", "action", "order_type",
        "price", "quantity", "status", "fill_price", "ts", "user_remark",
    )
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)
        self.status = "Pending"
        self.fill_price = 0.0
        self.ts = datetime.now().strftime("%d-%b-%Y %H:%M:%S")
        if not hasattr(self, "user_remark"):
            self.user_remark = ""


class MockBreeze:
    """Complete BreezeConnect mock — ALL 23 methods."""

    def __init__(self, api_key="mock"):
        self.api_key = api_key
        self._ws_on = False
        self._ws_thread = None
        self._subs: List[Dict] = []
        self._orders: Dict[str, _Order] = {}
        self._positions: Dict[str, dict] = {}
        self._spot = {"NIFTY": 24250.0, "CNXBAN": 52500.0, "BSESEN": 79000.0}
        self._vol = 0.14
        self._r = Config.RISK_FREE_RATE
        self.on_ticks: Optional[Callable] = None
        self._lock = threading.Lock()
        self._funds_base = 500000.0

    # ── 1. generate_session ──────────────────────────────────
    def generate_session(self, api_secret, session_token):
        return {"Success": {"session_token": "mock_tok"}, "Status": 200, "Error": None}

    # ── 2. get_customer_details ──────────────────────────────
    def get_customer_details(self, api_session=""):
        return {
            "Success": {
                "name": "Mock Trader",
                "client_id": "MOCK001",
                "email": "mock@example.com",
                "phone": "9999999999",
                "pan": "XXXPX0000X",
                "demat_id": "MOCK-DEMAT",
            },
            "Status": 200, "Error": None,
        }

    # ── 3. get_demat_holdings ────────────────────────────────
    def get_demat_holdings(self):
        return {"Success": [], "Status": 200, "Error": None}

    # ── 4. get_funds ─────────────────────────────────────────
    def get_funds(self):
        with self._lock:
            utilized = sum(
                abs(p.get("quantity", 0)) * safe_float(p.get("average_price", 0)) * 0.15
                for p in self._positions.values()
            )
        return {
            "Success": [{
                "amount_allocated": str(self._funds_base),
                "amount_utilized": str(round(utilized, 2)),
                "block_by_trade_amount": "0",
                "limit_available": str(round(self._funds_base - utilized, 2)),
            }],
            "Status": 200, "Error": None,
        }

    # ── 5. get_portfolio_holdings ────────────────────────────
    def get_portfolio_holdings(self):
        return {"Success": [], "Status": 200, "Error": None}

    # ── 6. get_portfolio_positions ───────────────────────────
    def get_portfolio_positions(self):
        with self._lock:
            pos = []
            for k, p in self._positions.items():
                if p.get("quantity", 0) != 0:
                    S = self._spot.get(p["stock_code"], 24250)
                    rt = OptionRight.CALL if p.get("right", "").lower() == "call" else OptionRight.PUT
                    T = time_to_expiry(p.get("expiry_date", ""))
                    ltp = max(BlackScholes.price(S, float(p.get("strike_price", 0)),
                              T, self._r, self._vol, rt), 0.05)
                    pos.append({
                        **p,
                        "ltp": str(round(ltp, 2)),
                        "product_type": "options",
                        "booked_profit_loss": "0",
                    })
        return {"Success": pos, "Status": 200, "Error": None}

    # ── 7. get_order_list ────────────────────────────────────
    def get_order_list(self, exchange_code="", from_date="", to_date=""):
        with self._lock:
            orders = []
            for oid, o in self._orders.items():
                orders.append({
                    "order_id": o.order_id, "exchange_code": o.exchange_code,
                    "stock_code": o.stock_code, "product_type": "options",
                    "action": o.action.capitalize(), "order_type": o.order_type.capitalize(),
                    "quantity": str(o.quantity), "price": str(o.price),
                    "average_price": str(o.fill_price) if o.fill_price > 0 else str(o.price),
                    "status": o.status, "validity": "Day",
                    "order_datetime": o.ts,
                    "strike_price": str(int(o.strike_price)),
                    "right": o.right.capitalize(),
                    "expiry_date": o.expiry_date,
                    "user_remark": getattr(o, "user_remark", ""),
                })
        return {"Success": orders, "Status": 200, "Error": None}

    # ── 8. get_order_detail ──────────────────────────────────
    def get_order_detail(self, exchange_code="NFO", order_id=""):
        with self._lock:
            if order_id in self._orders:
                o = self._orders[order_id]
                return {"Success": [{
                    "order_id": o.order_id, "status": o.status,
                    "action": o.action, "quantity": str(o.quantity),
                    "price": str(o.price),
                    "average_price": str(o.fill_price) if o.fill_price > 0 else str(o.price),
                    "stock_code": o.stock_code,
                    "strike_price": str(int(o.strike_price)),
                    "right": o.right, "expiry_date": o.expiry_date,
                }], "Status": 200, "Error": None}
        return {"Success": None, "Status": 400, "Error": "Not found"}

    # ── 9. get_trade_list ────────────────────────────────────
    def get_trade_list(self, exchange_code="", from_date="", to_date=""):
        with self._lock:
            trades = []
            for oid, o in self._orders.items():
                if o.status == "Executed":
                    trades.append({
                        "trade_id": f"T{oid[-8:]}", "order_id": o.order_id,
                        "exchange_code": o.exchange_code,
                        "stock_code": o.stock_code, "product_type": "options",
                        "action": o.action.capitalize(),
                        "quantity": str(o.quantity),
                        "trade_price": str(o.fill_price),
                        "order_type": o.order_type.capitalize(),
                        "trade_datetime": o.ts,
                        "strike_price": str(int(o.strike_price)),
                        "right": o.right.capitalize(),
                        "expiry_date": o.expiry_date,
                    })
        return {"Success": trades, "Status": 200, "Error": None}

    # ── 10. get_trade_detail ─────────────────────────────────
    def get_trade_detail(self, exchange_code="", order_id=""):
        tl = self.get_trade_list(exchange_code)
        if tl and tl.get("Success"):
            matches = [t for t in tl["Success"] if t.get("order_id") == order_id]
            if matches:
                return {"Success": matches, "Status": 200, "Error": None}
        return {"Success": None, "Status": 400, "Error": "Not found"}

    # ── 11. place_order ──────────────────────────────────────
    def place_order(self, stock_code="", exchange_code="NFO", product="options",
                    action="sell", order_type="limit", stoploss="",
                    quantity="25", price="0", validity="day",
                    validity_date="", disclosed_quantity="0",
                    expiry_date="", right="call", strike_price="0",
                    user_remark="", **kw):
        oid = f"MOCK{uuid.uuid4().hex[:8].upper()}"
        o = _Order(order_id=oid, stock_code=stock_code, exchange_code=exchange_code,
                   strike_price=float(strike_price), right=right,
                   expiry_date=expiry_date, action=action.lower(),
                   order_type=order_type.lower(), price=safe_float(price),
                   quantity=int(quantity), user_remark=user_remark)
        with self._lock:
            self._orders[oid] = o
        return {"Success": {"order_id": oid}, "Status": 200, "Error": None}

    # ── 12. modify_order ─────────────────────────────────────
    def modify_order(self, order_id="", exchange_code="NFO", order_type="limit",
                     stoploss="", quantity="0", price="0", validity="day",
                     validity_date="", disclosed_quantity="0"):
        with self._lock:
            if order_id in self._orders:
                self._orders[order_id].price = safe_float(price)
                if int(quantity) > 0:
                    self._orders[order_id].quantity = int(quantity)
                self._orders[order_id].status = "Pending"
                return {"Success": {"order_id": order_id}, "Status": 200, "Error": None}
        return {"Success": None, "Status": 400, "Error": "Not found"}

    # ── 13. cancel_order ─────────────────────────────────────
    def cancel_order(self, exchange_code="NFO", order_id=""):
        with self._lock:
            if order_id in self._orders:
                self._orders[order_id].status = "Cancelled"
                return {"Success": {"order_id": order_id}, "Status": 200, "Error": None}
        return {"Success": None, "Status": 400, "Error": "Not found"}

    # ── 14. square_off ───────────────────────────────────────
    def square_off(self, exchange_code="", product="", stock_code="",
                   quantity="0", price="0", action="", order_type="",
                   validity="", stoploss="", disclosed_quantity="0",
                   expiry_date="", right="", strike_price="", **kw):
        return self.place_order(
            stock_code=stock_code, exchange_code=exchange_code,
            product=product, action=action.lower(),
            order_type=order_type.lower() if order_type else "market",
            quantity=quantity, price=price, expiry_date=expiry_date,
            right=right.lower() if right else "call",
            strike_price=strike_price,
        )

    # ── 15. get_option_chain_quotes ──────────────────────────
    def get_option_chain_quotes(self, stock_code="", exchange_code="NFO",
                                product_type="options", expiry_date="", right="others"):
        with self._lock:
            S = self._spot.get(stock_code, 24250.0)
        gap = Config.strike_gap(stock_code)
        atm = round(S / gap) * gap
        strikes = [atm + i * gap for i in range(-10, 11)]
        T = time_to_expiry(expiry_date)
        rights = ([OptionRight.CALL, OptionRight.PUT] if right == "others"
                  else [OptionRight.CALL if right.lower() == "call" else OptionRight.PUT])
        chain = []
        for st in strikes:
            for rt in rights:
                iv = max(self._vol + random.uniform(-0.02, 0.02), 0.05)
                p = max(BlackScholes.price(S, st, T, self._r, iv, rt), 0.05)
                sp = max(0.05, p * 0.003)
                chain.append({
                    "stock_code": stock_code, "strike_price": str(int(st)),
                    "right": "Call" if rt == OptionRight.CALL else "Put",
                    "expiry_date": expiry_date,
                    "ltp": str(round(p, 2)),
                    "best_bid_price": str(round(max(p - sp, 0.05), 2)),
                    "best_offer_price": str(round(p + sp, 2)),
                    "volume": str(random.randint(10000, 500000)),
                    "open_interest": str(random.randint(100000, 5000000)),
                })
        return {"Success": chain, "Status": 200, "Error": None}

    # ── 16. get_quotes ───────────────────────────────────────
    def get_quotes(self, stock_code="", exchange_code="NFO", expiry_date="",
                   product_type="options", right="call", strike_price="0"):
        S = self.get_spot_price(stock_code)
        rt = OptionRight.CALL if right.lower() == "call" else OptionRight.PUT
        T = time_to_expiry(expiry_date)
        p = max(BlackScholes.price(S, float(strike_price), T, self._r, self._vol, rt), 0.05)
        sp = max(0.05, p * 0.003)
        return {"Success": [{"ltp": str(round(p, 2)),
                "best_bid_price": str(round(max(p - sp, 0.05), 2)),
                "best_offer_price": str(round(p + sp, 2))}],
                "Status": 200, "Error": None}

    # ── 17. get_historical_data ──────────────────────────────
    def get_historical_data(self, interval="1day", from_date="", to_date="",
                            stock_code="", exchange_code="", product_type="",
                            expiry_date="", right="", strike_price=""):
        data = []
        base = self._spot.get(stock_code, 24250)
        for i in range(30):
            dt = datetime.now() - timedelta(days=30 - i)
            o = base * (1 + random.uniform(-0.02, 0.02))
            h = o * (1 + random.uniform(0, 0.015))
            l = o * (1 - random.uniform(0, 0.015))
            c = (h + l) / 2
            data.append({
                "datetime": dt.strftime("%Y-%m-%d %H:%M:%S"),
                "stock_code": stock_code, "exchange_code": exchange_code,
                "open": str(round(o, 2)), "high": str(round(h, 2)),
                "low": str(round(l, 2)), "close": str(round(c, 2)),
                "volume": str(random.randint(100000, 5000000)),
            })
        return {"Success": data, "Status": 200, "Error": None}

    # ── 18. get_historical_data_v2 ───────────────────────────
    def get_historical_data_v2(self, **kw):
        return self.get_historical_data(**kw)

    # ── 19. margin_calculator ────────────────────────────────
    def margin_calculator(self, payload_list=None):
        if not payload_list:
            return {"Success": {}, "Status": 200, "Error": None}
        total = 0
        for p in payload_list:
            strike = safe_float(p.get("strike_price", 0))
            qty = abs(int(p.get("quantity", 0)))
            total += strike * qty * 0.15
        return {
            "Success": {"total_margin": str(round(total, 2)),
                        "span_margin": str(round(total * 0.8, 2)),
                        "exposure_margin": str(round(total * 0.2, 2))},
            "Status": 200, "Error": None,
        }

    # ── 20-23. WebSocket methods (unchanged from before) ─────

    def ws_connect(self):
        if self._ws_on: return
        self._ws_on = True
        self._ws_thread = threading.Thread(target=self._ws_loop, daemon=True, name="MockWS")
        self._ws_thread.start()

    def ws_disconnect(self):
        self._ws_on = False

    def subscribe_feeds(self, exchange_code="NFO", stock_code="",
                        product_type="options", expiry_date="",
                        strike_price="", right="others",
                        get_exchange_quotes=True, get_market_depth=False):
        sub = {"exchange_code": exchange_code, "stock_code": stock_code,
               "product_type": product_type, "expiry_date": expiry_date,
               "strike_price": strike_price, "right": right}
        with self._lock:
            keys = {f"{s['stock_code']}|{s['strike_price']}|{s['right']}" for s in self._subs}
            k = f"{stock_code}|{strike_price}|{right}"
            if k not in keys: self._subs.append(sub)
        return {"Success": None, "Status": 200, "Error": None}

    def unsubscribe_feeds(self, **kw):
        with self._lock:
            self._subs = [s for s in self._subs
                          if not (s.get("stock_code") == kw.get("stock_code")
                                  and s.get("strike_price") == kw.get("strike_price"))]
        return {"Success": None, "Status": 200, "Error": None}

    # ── Helpers ──────────────────────────────────────────────

    def get_spot_price(self, code):
        with self._lock: return self._spot.get(code, 24250.0)

    def _ws_loop(self):
        while self._ws_on:
            self._step()
            with self._lock: subs = list(self._subs)
            for s in subs:
                t = self._tick(s)
                if t and self.on_ticks:
                    try: self.on_ticks(t)
                    except: pass
            self._fill()
            time.sleep(0.5)

    def _step(self):
        with self._lock:
            for c in self._spot:
                S = self._spot[c]
                dt_y = 0.5 / (365.25 * 24 * 3600)
                self._spot[c] = S * math.exp(
                    (self._r - 0.5*self._vol**2)*dt_y +
                    self._vol*math.sqrt(dt_y)*random.gauss(0,1))

    def _tick(self, sub):
        stk = sub.get("stock_code","NIFTY")
        strike = float(sub.get("strike_price","0") or "0")
        if strike <= 0: return None
        rs = sub.get("right","call").lower()
        exp = sub.get("expiry_date","")
        right = OptionRight.CALL if rs=="call" else OptionRight.PUT
        with self._lock: S = self._spot.get(stk, 24250.0)
        T = time_to_expiry(exp)
        iv = max(self._vol+random.uniform(-0.01,0.01), 0.05)
        bs = max(BlackScholes.price(S,strike,T,self._r,iv,right), 0.05)
        sp = max(0.05, bs*0.002)
        return {
            "stock_code":stk, "exchange_code":sub.get("exchange_code","NFO"),
            "strike_price":str(int(strike)),
            "right":"Call" if right==OptionRight.CALL else "Put",
            "expiry_date":exp, "ltp":str(round(max(bs+random.uniform(-sp,sp),0.05),2)),
            "best_bid_price":str(round(max(bs-sp,0.05),2)),
            "best_offer_price":str(round(bs+sp,2)),
            "total_quantity_traded":str(random.randint(10000,500000)),
            "open_interest":str(random.randint(100000,5000000)),
        }

    def _fill(self):
        with self._lock:
            for o in list(self._orders.values()):
                if o.status != "Pending": continue
                S = self._spot.get(o.stock_code, 24250)
                rt = OptionRight.CALL if o.right.lower()=="call" else OptionRight.PUT
                T = time_to_expiry(o.expiry_date)
                fair = max(BlackScholes.price(S,o.strike_price,T,self._r,self._vol,rt), 0.05)
                if o.order_type == "market":
                    slip = random.uniform(0,0.3)
                    o.fill_price = round(max(fair+(slip if o.action=="buy" else -slip), 0.05),2)
                    o.status = "Executed"
                    self._pos_up(o)
                elif o.action=="sell" and o.price >= fair*0.97:
                    o.fill_price = round(max(o.price+random.uniform(-0.15,0.15),0.05),2)
                    o.status = "Executed"; self._pos_up(o)
                elif o.action=="buy" and o.price <= fair*1.03:
                    o.fill_price = round(max(o.price+random.uniform(-0.15,0.15),0.05),2)
                    o.status = "Executed"; self._pos_up(o)

    def _pos_up(self, o):
        k = f"{o.stock_code}|{int(o.strike_price)}|{o.right}|{o.expiry_date}"
        if k not in self._positions:
            self._positions[k] = {"stock_code":o.stock_code,
                "strike_price":str(int(o.strike_price)), "right":o.right,
                "expiry_date":o.expiry_date, "quantity":0, "average_price":0.0}
        p = self._positions[k]
        p["quantity"] += (-o.quantity if o.action=="sell" else o.quantity)
        p["average_price"] = o.fill_price
        if p["quantity"] == 0: del self._positions[k]
