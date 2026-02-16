# ═══════════════════════════════════════════════════════════════
# FILE: order_manager.py
# ═══════════════════════════════════════════════════════════════
"""
Order lifecycle with Chase algorithm.

FIXES from audit:
  1. place_order uses `product="options"` (not `product_type`)
  2. right uses lowercase ("call"/"put")
  3. Chase polling is throttled to not burn rate limit
  4. get_order_detail response parsing handles live Breeze's
     'average_price' field (not 'fill_price')
"""

import time
from datetime import datetime
from typing import Optional

from models import Leg, LegStatus, OrderSide
from connection_manager import SessionManager
from database import Database
from shared_state import SharedState
from config import Config
from utils import LOG, safe_float


class OrderManager:

    def __init__(self, session: SessionManager, db: Database, state: SharedState):
        self.session = session
        self.db = db
        self.state = state

    def execute_sell_with_chase(self, leg: Leg) -> bool:
        """
        Chase-sell algorithm:
          1. Place LIMIT SELL at current bid/LTP
          2. Wait CHASE_TIMEOUT seconds
          3. If not filled, cancel → re-place at new LTP
          4. After CHASE_MAX_RETRIES, fall back to MARKET
        """
        self.state.add_log("INFO", "Order",
                           f"Chase SELL: {leg.stock_code} {int(leg.strike_price)}"
                           f"{leg.right.value[0].upper()} qty={leg.quantity}")

        for attempt in range(Config.CHASE_MAX_RETRIES):
            price = self._get_sell_price(leg)
            if price <= 0:
                self.state.add_log("WARN", "Order", f"No price for {leg.feed_key}")
                time.sleep(0.5)
                continue

            self.state.add_log("INFO", "Order",
                               f"Attempt {attempt + 1}: SELL @ ₹{price:.2f}")

            result = self.session.place_order(
                stock_code=leg.stock_code,
                exchange_code=leg.exchange_code,
                product="options",
                action="sell",
                order_type="limit",
                stoploss="",
                quantity=str(leg.quantity),
                price=str(round(price, 2)),
                validity="day",
                validity_date="",
                disclosed_quantity="0",
                expiry_date=leg.expiry_date,
                right=leg.right.value,     # "call" or "put" (lowercase)
                strike_price=str(int(leg.strike_price)),
            )

            if not result or result.get("Status") != 200:
                error = result.get("Error", "Unknown") if result else "No response"
                self.state.add_log("ERROR", "Order", f"Rejected: {error}")
                self.db.log_order(leg.leg_id, "", "sell", "rejected",
                                  price, leg.quantity, str(error))
                continue

            order_id = result["Success"]["order_id"]
            leg.entry_order_id = order_id
            self.db.log_order(leg.leg_id, order_id, "sell", "placed",
                              price, leg.quantity)

            # Wait for fill — use fewer polls to save rate limit
            filled = self._poll_fill(order_id, Config.CHASE_TIMEOUT)

            if filled:
                fill_price = self._extract_fill_price(order_id, price)
                self._mark_leg_filled(leg, fill_price, "sell")
                return True

            # Not filled → cancel and retry
            self.session.cancel_order(order_id)
            self.db.log_order(leg.leg_id, order_id, "sell", "cancelled",
                              price, leg.quantity, "chase timeout")
            self.state.add_log("INFO", "Order", f"Chase cancel #{attempt + 1}")
            time.sleep(0.2)

        # Exhausted → market order
        self.state.add_log("WARN", "Order", "Chase exhausted → MARKET")
        return self._market_order(leg, "sell")

    def execute_limit_sell(self, leg: Leg, limit_price: float) -> bool:
        """
        Place a single LIMIT SELL order without chase.
        Used by the manual limit order screen.
        """
        self.state.add_log("INFO", "Order",
                           f"LIMIT SELL: {leg.stock_code} {int(leg.strike_price)}"
                           f"{leg.right.value[0].upper()} @ ₹{limit_price:.2f}")

        result = self.session.place_order(
            stock_code=leg.stock_code,
            exchange_code=leg.exchange_code,
            product="options",
            action="sell",
            order_type="limit",
            stoploss="",
            quantity=str(leg.quantity),
            price=str(round(limit_price, 2)),
            validity="day",
            validity_date="",
            disclosed_quantity="0",
            expiry_date=leg.expiry_date,
            right=leg.right.value,
            strike_price=str(int(leg.strike_price)),
        )

        if not result or result.get("Status") != 200:
            error = result.get("Error", "Unknown") if result else "No response"
            self.state.add_log("ERROR", "Order", f"Limit sell rejected: {error}")
            self.db.log_order(leg.leg_id, "", "sell", "rejected",
                              limit_price, leg.quantity, str(error))
            return False

        order_id = result["Success"]["order_id"]
        leg.entry_order_id = order_id
        leg.status = LegStatus.ENTERING
        self.db.save_leg(leg)
        self.db.log_order(leg.leg_id, order_id, "sell", "placed",
                          limit_price, leg.quantity, "limit order — no chase")

        # Wait a bit for fill
        filled = self._poll_fill(order_id, timeout=5.0)
        if filled:
            fill_price = self._extract_fill_price(order_id, limit_price)
            self._mark_leg_filled(leg, fill_price, "sell")
            return True

        # Order is open but not yet filled — mark as entering
        leg.status = LegStatus.ENTERING
        leg.entry_price = limit_price  # expected price
        self.db.save_leg(leg)
        self.state.add_log("INFO", "Order",
                           f"Limit order OPEN (not yet filled): {order_id}")
        return True  # Order placed successfully, just not filled

    def execute_buy_market(self, leg: Leg) -> bool:
        """Immediate market buy for exits."""
        return self._market_order(leg, "buy")

    # ── Internal helpers ─────────────────────────────────────

    def _market_order(self, leg: Leg, action: str) -> bool:
        result = self.session.place_order(
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

        if not result or result.get("Status") != 200:
            error = result.get("Error", "Unknown") if result else "No response"
            self.state.add_log("ERROR", "Order", f"Market {action} failed: {error}")
            self.db.log_order(leg.leg_id, "", action, "rejected",
                              0, leg.quantity, str(error))
            return False

        order_id = result["Success"]["order_id"]
        filled = self._poll_fill(order_id, timeout=5.0)
        current = self._get_current_ltp(leg)
        fill_price = self._extract_fill_price(order_id, current)

        if action == "sell":
            self._mark_leg_filled(leg, fill_price, "sell")
        else:
            leg.exit_order_id = order_id
            leg.exit_price = fill_price
            leg.current_price = fill_price
            leg.status = LegStatus.SQUARED_OFF
            leg.exit_time = datetime.now().isoformat()
            leg.compute_pnl()
            self.db.save_leg(leg)
            self.db.log_order(leg.leg_id, order_id, action, "filled",
                              fill_price, leg.quantity)

        self.state.add_log("INFO", "Order",
                           f"MARKET {action.upper()}: {int(leg.strike_price)}"
                           f"{leg.right.value[0].upper()} @ ₹{fill_price:.2f}")
        return True

    def _mark_leg_filled(self, leg: Leg, fill_price: float, action: str):
        leg.entry_price = fill_price
        leg.current_price = fill_price
        leg.status = LegStatus.ACTIVE
        leg.entry_time = datetime.now().isoformat()
        sl_pct = leg.sl_percentage if leg.sl_percentage > 0 else Config.SL_PERCENTAGE
        leg.sl_price = round(fill_price * (1 + sl_pct / 100.0), 2)
        leg.sl_percentage = sl_pct
        self.db.save_leg(leg)
        self.db.log_order(leg.leg_id, leg.entry_order_id, action, "filled",
                          fill_price, leg.quantity)
        self.state.add_log("INFO", "Order",
                           f"FILLED: {int(leg.strike_price)}"
                           f"{leg.right.value[0].upper()} @ ₹{fill_price:.2f} "
                           f"SL=₹{leg.sl_price:.2f}")

    def _poll_fill(self, order_id: str, timeout: float) -> bool:
        """
        Poll order status. Uses at most 4 polls to conserve rate limit.
        """
        polls = min(4, int(timeout / 0.8))
        interval = timeout / max(polls, 1)
        for _ in range(polls):
            time.sleep(interval)
            detail = self.session.get_order_detail(order_id)
            if detail and detail.get("Success"):
                records = detail["Success"]
                if isinstance(records, list) and records:
                    status = str(records[0].get("status", "")).lower()
                    if status in ("filled", "complete", "executed", "traded"):
                        return True
                    if status in ("rejected", "cancelled"):
                        return False
        return False

    def _extract_fill_price(self, order_id: str, fallback: float) -> float:
        """
        Breeze LIVE uses 'average_price' in order detail.
        Mock uses 'average_price' too (we corrected mock_breeze).
        """
        detail = self.session.get_order_detail(order_id)
        if detail and detail.get("Success"):
            records = detail["Success"]
            if isinstance(records, list) and records:
                # Try both field names
                fp = safe_float(records[0].get("average_price", 0))
                if fp > 0:
                    return fp
                fp = safe_float(records[0].get("price", 0))
                if fp > 0:
                    return fp
        return fallback

    def _get_sell_price(self, leg: Leg) -> float:
        """Get bid price for selling. Falls back to LTP."""
        tick = self.state.get_tick(leg.feed_key)
        if tick:
            if tick.best_bid > 0:
                return tick.best_bid
            if tick.ltp > 0:
                return tick.ltp
        return self._get_current_ltp(leg)

    def _get_current_ltp(self, leg: Leg) -> float:
        tick = self.state.get_tick(leg.feed_key)
        if tick and tick.ltp > 0:
            return tick.ltp
        # API fallback
        try:
            result = self.session.breeze.get_quotes(
                stock_code=leg.stock_code,
                exchange_code="NFO",
                expiry_date=leg.expiry_date,
                product_type="options",
                right=leg.right.value,
                strike_price=str(int(leg.strike_price)),
            )
            if result and result.get("Success"):
                records = result["Success"]
                if isinstance(records, list) and records:
                    return safe_float(records[0].get("ltp", 0))
        except Exception as e:
            LOG.error(f"get_quotes fallback error: {e}")
        return 0.0
