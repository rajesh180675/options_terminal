"""
order_manager.py

Professional order execution:
  - sell_chase() : limit sell with cancel+replace until filled, then market fallback
  - sell_limit() : single limit sell (no chase) for manual order screen
  - buy_market() : market buy (for SL/panic exit of short legs)

Fill parsing:
  - Breeze often returns status 'Executed' for fills
  - Fill price is typically in 'average_price'
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Optional

from app_config import Config
from connection_manager import SessionManager
from database import Database
from shared_state import SharedState
from models import Leg, LegStatus, OrderSide
from utils import LOG, safe_float


_FILLED = {"executed", "filled", "complete", "traded"}
_DEAD = {"rejected", "cancelled", "canceled"}


class OrderManager:
    def __init__(self, session: SessionManager, db: Database, state: SharedState):
        self.session = session
        self.db = db
        self.state = state

    # ─────────────────────────────────────────────────────────
    # Primary execution methods (expected by StrategyEngine)
    # ─────────────────────────────────────────────────────────

    def sell_chase(self, leg: Leg) -> bool:
        """
        Place limit SELL at best bid; if not filled within CHASE_TIMEOUT,
        cancel and retry at new bid/LTP. After retries, use market SELL.

        Returns True if order placed and filled (or assumed filled for market),
        False if rejected.
        """
        self.state.add_log("INFO", "Order", f"Chase SELL: {leg.display_name} qty={leg.quantity}")

        for attempt in range(int(Config.CHASE_MAX_RETRIES)):
            px = self._get_sell_price(leg)
            if px <= 0:
                time.sleep(0.3)
                continue

            res = self.session.place_order(
                stock_code=leg.stock_code,
                exchange_code=leg.exchange_code,
                product="options",
                action="sell",
                order_type="limit",
                stoploss="",
                quantity=str(leg.quantity),
                price=str(round(px, 2)),
                validity="day",
                validity_date="",
                disclosed_quantity="0",
                expiry_date=leg.expiry_date,
                right=leg.right.value,  # "call" or "put"
                strike_price=str(int(leg.strike_price)),
            )

            if not res or res.get("Status") != 200:
                err = (res or {}).get("Error", "No response")
                self.state.add_log("ERROR", "Order", f"SELL rejected: {err}")
                self.db.log_order(leg.leg_id, "", "sell", "rejected", px, leg.quantity, str(err))
                continue

            oid = res["Success"]["order_id"]
            leg.entry_order_id = oid
            self.db.log_order(leg.leg_id, oid, "sell", "placed", px, leg.quantity, "chase")

            if self._poll_fill(oid, timeout=float(Config.CHASE_TIMEOUT), exchange=leg.exchange_code):
                fill_px = self._fill_price(oid, fallback=px, exchange=leg.exchange_code)
                self._mark_entry_filled(leg, fill_px)
                return True

            # cancel and retry
            try:
                self.session.cancel_order(oid, exchange_code=leg.exchange_code)
            except Exception:
                pass
            self.db.log_order(leg.leg_id, oid, "sell", "cancelled", px, leg.quantity, "chase timeout")
            time.sleep(0.15)

        # market fallback
        self.state.add_log("WARN", "Order", f"Chase exhausted → MARKET SELL: {leg.display_name}")
        return self._market(leg, action="sell")

    def sell_limit(self, leg: Leg, limit_price: float) -> bool:
        """
        Place a single LIMIT SELL (no chase).
        If not filled quickly, keep leg in ENTERING; pending monitor can pick it up.
        """
        px = float(limit_price)
        self.state.add_log("INFO", "Order", f"LIMIT SELL: {leg.display_name} @ {px:.2f}")

        res = self.session.place_order(
            stock_code=leg.stock_code,
            exchange_code=leg.exchange_code,
            product="options",
            action="sell",
            order_type="limit",
            stoploss="",
            quantity=str(leg.quantity),
            price=str(round(px, 2)),
            validity="day",
            validity_date="",
            disclosed_quantity="0",
            expiry_date=leg.expiry_date,
            right=leg.right.value,
            strike_price=str(int(leg.strike_price)),
        )

        if not res or res.get("Status") != 200:
            err = (res or {}).get("Error", "No response")
            self.state.add_log("ERROR", "Order", f"Limit sell rejected: {err}")
            self.db.log_order(leg.leg_id, "", "sell", "rejected", px, leg.quantity, str(err))
            return False

        oid = res["Success"]["order_id"]
        leg.entry_order_id = oid
        leg.status = LegStatus.ENTERING
        self.db.save_leg(leg)
        self.db.log_order(leg.leg_id, oid, "sell", "placed", px, leg.quantity, "limit")

        # quick poll
        if self._poll_fill(oid, timeout=5.0, exchange=leg.exchange_code):
            fill_px = self._fill_price(oid, fallback=px, exchange=leg.exchange_code)
            self._mark_entry_filled(leg, fill_px)
            return True

        # not filled; keep as entering
        leg.entry_price = px  # expected
        self.db.save_leg(leg)
        self.state.add_log("INFO", "Order", f"Limit order pending: {oid}")
        return True

    def buy_market(self, leg: Leg) -> bool:
        """
        Market BUY for closing a short option leg.
        """
        return self._market(leg, action="buy")

    # ─────────────────────────────────────────────────────────
    # Internals
    # ─────────────────────────────────────────────────────────

    def _market(self, leg: Leg, action: str) -> bool:
        res = self.session.place_order(
            stock_code=leg.stock_code,
            exchange_code=leg.exchange_code,
            product="options",
            action=action,
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

        if not res or res.get("Status") != 200:
            err = (res or {}).get("Error", "No response")
            self.state.add_log("ERROR", "Order", f"Market {action} rejected: {err}")
            self.db.log_order(leg.leg_id, "", action, "rejected", 0.0, leg.quantity, str(err))
            return False

        oid = res["Success"]["order_id"]
        filled = self._poll_fill(oid, timeout=5.0, exchange=leg.exchange_code)
        fp = self._fill_price(oid, fallback=self._get_ltp(leg), exchange=leg.exchange_code)

        if action == "sell":
            leg.entry_order_id = oid
            self._mark_entry_filled(leg, fp)
        else:
            # exit
            leg.exit_order_id = oid
            leg.exit_price = fp
            leg.current_price = fp
            leg.exit_time = datetime.now().isoformat()
            leg.status = LegStatus.SQUARED_OFF
            leg.compute_pnl()
            self.db.save_leg(leg)
            self.db.log_order(leg.leg_id, oid, action, "filled", fp, leg.quantity, "market")

        self.state.add_log("INFO", "Order", f"MKT {action.upper()}: {leg.display_name} @ {fp:.2f}")
        return True

    def _mark_entry_filled(self, leg: Leg, fill_price: float) -> None:
        leg.entry_price = float(fill_price)
        leg.current_price = float(fill_price)
        leg.lowest_price = float(fill_price) if leg.lowest_price <= 0 else min(leg.lowest_price, float(fill_price))
        leg.entry_time = datetime.now().isoformat()
        leg.status = LegStatus.ACTIVE

        # SL = entry * (1 + sl%)
        sl_pct = leg.sl_percentage if leg.sl_percentage > 0 else float(Config.SL_PERCENTAGE)
        leg.sl_percentage = float(sl_pct)
        leg.sl_price = round(leg.entry_price * (1 + sl_pct / 100.0), 2)

        self.db.save_leg(leg)
        self.db.log_order(leg.leg_id, leg.entry_order_id, "sell", "filled", leg.entry_price, leg.quantity, "entry fill")
        self.state.add_log("INFO", "Order", f"FILLED: {leg.display_name} @ {leg.entry_price:.2f} SL={leg.sl_price:.2f}")

    def _poll_fill(self, order_id: str, timeout: float, exchange: str) -> bool:
        """
        Poll order status up to 3 times (rate-limit friendly).
        """
        polls = min(3, max(1, int(timeout / 1.0)))
        interval = timeout / polls

        for _ in range(polls):
            time.sleep(interval)
            d = self.session.get_order_detail(order_id, exchange_code=exchange)
            if not d or not d.get("Success"):
                continue
            recs = d["Success"]
            if not isinstance(recs, list) or not recs:
                continue
            st = str(recs[0].get("status", "")).strip().lower()
            if st in _FILLED:
                return True
            if st in _DEAD:
                return False
        return False

    def _fill_price(self, order_id: str, fallback: float, exchange: str) -> float:
        d = self.session.get_order_detail(order_id, exchange_code=exchange)
        if d and d.get("Success"):
            recs = d["Success"]
            if isinstance(recs, list) and recs:
                fp = safe_float(recs[0].get("average_price", 0))
                if fp > 0:
                    return fp
                fp = safe_float(recs[0].get("price", 0))
                if fp > 0:
                    return fp
        return float(fallback)

    def _get_sell_price(self, leg: Leg) -> float:
        """
        Use best bid for SELL; fallback to ltp; fallback to REST quote.
        """
        tick = self.state.get_tick(leg.feed_key)
        if tick:
            if tick.best_bid > 0:
                return float(tick.best_bid)
            if tick.ltp > 0:
                return float(tick.ltp)
        return self._get_ltp(leg)

    def _get_ltp(self, leg: Leg) -> float:
        tick = self.state.get_tick(leg.feed_key)
        if tick and tick.ltp > 0:
            return float(tick.ltp)

        # REST fallback
        try:
            r = self.session.get_quotes(
                stock_code=leg.stock_code,
                exchange_code=leg.exchange_code,
                expiry_date=leg.expiry_date,
                product_type="options",
                right=leg.right.value,
                strike_price=str(int(leg.strike_price)),
            )
            if r and r.get("Success"):
                return safe_float(r["Success"][0].get("ltp", 0))
        except Exception:
            pass
        return 0.0
