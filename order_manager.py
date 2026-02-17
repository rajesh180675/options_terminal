# ═══════════════════════════════════════════════════════════════
# FILE: order_manager.py
# ═══════════════════════════════════════════════════════════════
"""
Order lifecycle + Chase algorithm.
Status check: Breeze LIVE returns "Executed" (capital E).
We normalise to lowercase for comparison.
"""

import time
from datetime import datetime
from models import Leg, LegStatus, OrderSide
from connection_manager import SessionManager
from database import Database
from shared_state import SharedState
from app_config import Config
from utils import LOG, safe_float


# Breeze order statuses (normalised to lower)
_FILLED = {"executed", "filled", "complete", "traded"}
_REJECTED = {"rejected", "cancelled", "canceled"}


class OrderManager:

    def __init__(self, session: SessionManager, db: Database, state: SharedState):
        self.session = session
        self.db = db
        self.state = state

    def sell_chase(self, leg: Leg) -> bool:
        """Chase-sell: limit → cancel → re-price, then market fallback."""
        self.state.add_log("INFO", "Order",
                           f"Chase SELL: {leg.display_name} qty={leg.quantity}")
        for att in range(Config.CHASE_MAX_RETRIES):
            price = self._bid_price(leg)
            if price <= 0:
                time.sleep(0.5); continue
            self.state.add_log("INFO", "Order", f"Att {att+1}: SELL @₹{price:.2f}")
            r = self.session.place_order(
                stock_code=leg.stock_code, exchange_code=leg.exchange_code,
                product="options", action="sell", order_type="limit",
                stoploss="", quantity=str(leg.quantity),
                price=str(round(price, 2)), validity="day",
                validity_date="", disclosed_quantity="0",
                expiry_date=leg.expiry_date, right=leg.right.value,
                strike_price=str(int(leg.strike_price)))
            if not r or r.get("Status") != 200:
                err = r.get("Error") if r else "No response"
                self.state.add_log("ERROR", "Order", f"Rejected: {err}")
                self.db.log_order(leg.leg_id, "", "sell", "rejected", price, leg.quantity, str(err))
                continue
            oid = r["Success"]["order_id"]
            leg.entry_order_id = oid
            self.db.log_order(leg.leg_id, oid, "sell", "placed", price, leg.quantity)
            if self._poll(oid, Config.CHASE_TIMEOUT, leg.exchange_code):
                fp = self._fill_price(oid, price, leg.exchange_code)
                self._mark_entry(leg, fp)
                return True
            self.session.cancel_order(oid, leg.exchange_code)
            self.db.log_order(leg.leg_id, oid, "sell", "cancelled", price, leg.quantity, "chase")
            time.sleep(0.2)
        self.state.add_log("WARN", "Order", "Chase exhausted → MARKET")
        return self._market(leg, "sell")

    def sell_limit(self, leg: Leg, limit_price: float) -> bool:
        """Place limit sell without chase (manual order screen)."""
        self.state.add_log("INFO", "Order",
                           f"LIMIT SELL: {leg.display_name} @₹{limit_price:.2f}")
        r = self.session.place_order(
            stock_code=leg.stock_code, exchange_code=leg.exchange_code,
            product="options", action="sell", order_type="limit",
            stoploss="", quantity=str(leg.quantity),
            price=str(round(limit_price, 2)), validity="day",
            validity_date="", disclosed_quantity="0",
            expiry_date=leg.expiry_date, right=leg.right.value,
            strike_price=str(int(leg.strike_price)))
        if not r or r.get("Status") != 200:
            err = r.get("Error") if r else "No response"
            self.state.add_log("ERROR", "Order", f"Rejected: {err}")
            return False
        oid = r["Success"]["order_id"]
        leg.entry_order_id = oid
        leg.status = LegStatus.ENTERING
        self.db.save_leg(leg)
        self.db.log_order(leg.leg_id, oid, "sell", "placed", limit_price, leg.quantity, "limit")
        if self._poll(oid, 5.0, leg.exchange_code):
            fp = self._fill_price(oid, limit_price, leg.exchange_code)
            self._mark_entry(leg, fp)
            return True
        leg.entry_price = limit_price
        self.db.save_leg(leg)
        self.state.add_log("INFO", "Order", f"Limit open: {oid}")
        return True

    def buy_market(self, leg: Leg) -> bool:
        return self._market(leg, "buy")

    # ── Internals ────────────────────────────────────────────

    def _market(self, leg, action):
        r = self.session.place_order(
            stock_code=leg.stock_code, exchange_code=leg.exchange_code,
            product="options", action=action, order_type="market",
            stoploss="", quantity=str(leg.quantity), price="0",
            validity="day", validity_date="", disclosed_quantity="0",
            expiry_date=leg.expiry_date, right=leg.right.value,
            strike_price=str(int(leg.strike_price)))
        if not r or r.get("Status") != 200:
            self.state.add_log("ERROR", "Order", f"Market {action} failed")
            return False
        oid = r["Success"]["order_id"]
        self._poll(oid, 5.0, leg.exchange_code)
        fp = self._fill_price(oid, self._ltp(leg), leg.exchange_code)
        if action == "sell":
            self._mark_entry(leg, fp)
        else:
            leg.exit_order_id = oid
            leg.exit_price = fp
            leg.current_price = fp
            leg.status = LegStatus.SQUARED_OFF
            leg.exit_time = datetime.now().isoformat()
            leg.compute_pnl()
            self.db.save_leg(leg)
            self.db.log_order(leg.leg_id, oid, action, "filled", fp, leg.quantity)
        self.state.add_log("INFO", "Order",
                           f"MKT {action.upper()}: {leg.display_name} @₹{fp:.2f}")
        return True

    def _mark_entry(self, leg, fp):
        leg.entry_price = fp
        leg.current_price = fp
        leg.lowest_price = fp
        leg.status = LegStatus.ACTIVE
        leg.entry_time = datetime.now().isoformat()
        sl = leg.sl_percentage if leg.sl_percentage > 0 else Config.SL_PERCENTAGE
        leg.sl_price = round(fp * (1 + sl / 100.0), 2)
        leg.sl_percentage = sl
        self.db.save_leg(leg)
        self.db.log_order(leg.leg_id, leg.entry_order_id, "sell", "filled",
                          fp, leg.quantity)
        self.state.add_log("INFO", "Order",
                           f"FILLED: {leg.display_name} @₹{fp:.2f} SL=₹{leg.sl_price:.2f}")

    def _poll(self, oid, timeout, exchange="NFO"):
        polls = min(3, int(timeout / 1.0))
        interval = timeout / max(polls, 1)
        for _ in range(polls):
            time.sleep(interval)
            d = self.session.get_order_detail(oid, exchange)
            if d and d.get("Success"):
                recs = d["Success"]
                if isinstance(recs, list) and recs:
                    st = str(recs[0].get("status", "")).strip().lower()
                    if st in _FILLED:
                        return True
                    if st in _REJECTED:
                        return False
        return False

    def _fill_price(self, oid, fallback, exchange="NFO"):
        d = self.session.get_order_detail(oid, exchange)
        if d and d.get("Success"):
            recs = d["Success"]
            if isinstance(recs, list) and recs:
                fp = safe_float(recs[0].get("average_price", 0))
                if fp > 0:
                    return fp
                fp = safe_float(recs[0].get("price", 0))
                if fp > 0:
                    return fp
        return fallback

    def _bid_price(self, leg):
        t = self.state.get_tick(leg.feed_key)
        if t:
            if t.best_bid > 0:
                return t.best_bid
            if t.ltp > 0:
                return t.ltp
        return self._ltp(leg)

    def _ltp(self, leg):
        t = self.state.get_tick(leg.feed_key)
        if t and t.ltp > 0:
            return t.ltp
        try:
            r = self.session.breeze.get_quotes(
                stock_code=leg.stock_code, exchange_code=leg.exchange_code,
                expiry_date=leg.expiry_date, product_type="options",
                right=leg.right.value, strike_price=str(int(leg.strike_price)))
            if r and r.get("Success"):
                return safe_float(r["Success"][0].get("ltp", 0))
        except Exception:
            pass
        return 0.0

    def buy_entry_market(self, leg: Leg) -> bool:
        """
        Buy to OPEN a position (for iron condor protective wings).
        Different from buy_market which is buy to CLOSE.
        """
        result = self.session.place_order(
            stock_code=leg.stock_code,
            exchange_code=leg.exchange_code,
            product="options",
            action="buy",
            order_type="market",
            stoploss="",
            quantity=str(leg.quantity),
            price="0",
            validity="day",
            validity_date="",
            disclosed_quantity="0",
            expiry_date=leg.expiry_date,
            right=leg.right.value,
            strike_price=str(int(leg.strike_price)),
        )
        if not result or result.get("Status") != 200:
            self.state.add_log("ERROR", "Order",
                               f"Buy entry failed: {leg.display_name}")
            return False

        order_id = result["Success"]["order_id"]
        self._poll_fill(order_id, 5.0, leg.exchange_code)
        fp = self._extract_fill_price(order_id, self._get_ltp(leg), leg.exchange_code)

        leg.entry_order_id = order_id
        leg.entry_price = fp
        leg.current_price = fp
        leg.status = LegStatus.ACTIVE
        leg.entry_time = datetime.now().isoformat()
        # No SL for protective legs — strategy-level SL applies
        leg.sl_price = 0
        self.db.save_leg(leg)
        self.db.log_order(leg.leg_id, order_id, "buy", "filled",
                          fp, leg.quantity, "protective wing")
        self.state.add_log("INFO", "Order",
                           f"BUY ENTRY: {leg.display_name} @ ₹{fp:.2f}")
        return True
