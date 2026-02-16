# ═══════════════════════════════════════════════════════════════
# FILE: order_manager.py
# ═══════════════════════════════════════════════════════════════
"""
Order lifecycle management with the "Chase" algorithm.
Handles: place → monitor → modify/cancel → confirm fill.
"""

import time
from datetime import datetime
from typing import Optional

from models import Leg, LegStatus, OrderSide, OptionRight
from connection_manager import SessionManager
from database import Database
from shared_state import SharedState
from config import Config
from utils import LOG


class OrderManager:
    """Execute orders with chase logic and track them in the database."""

    def __init__(self, session: SessionManager, db: Database, state: SharedState):
        self.session = session
        self.db = db
        self.state = state

    def execute_sell_with_chase(self, leg: Leg) -> bool:
        """
        Place a SELL limit order at the current bid.
        If not filled within CHASE_TIMEOUT seconds, cancel and re-place at new LTP.
        Repeat up to CHASE_MAX_RETRIES times, then fall back to market.
        """
        self.state.add_log("INFO", "OrderMgr",
                           f"Chase-sell: {leg.stock_code} {leg.strike_price}"
                           f"{leg.right.value[0].upper()} qty={leg.quantity}")

        for attempt in range(Config.CHASE_MAX_RETRIES):
            # Get current price
            current_price = self._get_current_price(leg)
            if current_price <= 0:
                self.state.add_log("WARN", "OrderMgr", f"No price for {leg.feed_key}")
                time.sleep(0.5)
                continue

            order_price = round(current_price, 2)
            self.state.add_log("INFO", "OrderMgr",
                               f"Attempt {attempt + 1}: SELL @ {order_price}")

            result = self.session.place_order(
                stock_code=leg.stock_code,
                exchange_code=leg.exchange_code,
                product="options",
                action="sell",
                order_type="limit",
                stoploss="",
                quantity=str(leg.quantity),
                price=str(order_price),
                validity="day",
                validity_date="",
                disclosed_quantity="0",
                expiry_date=leg.expiry_date,
                right=leg.right.value,
                strike_price=str(int(leg.strike_price)),
            )

            if not result or result.get("Status") != 200:
                error = result.get("Error", "Unknown") if result else "No response"
                self.state.add_log("ERROR", "OrderMgr", f"Order rejected: {error}")
                self.db.log_order(leg.leg_id, "", "sell", "rejected", order_price,
                                  leg.quantity, str(error))
                continue

            order_id = result["Success"]["order_id"]
            leg.entry_order_id = order_id
            self.db.log_order(leg.leg_id, order_id, "sell", "placed",
                              order_price, leg.quantity)

            # Wait for fill
            filled = self._wait_for_fill(order_id, Config.CHASE_TIMEOUT)
            if filled:
                fill_price = self._get_fill_price(order_id)
                leg.entry_price = fill_price if fill_price > 0 else order_price
                leg.current_price = leg.entry_price
                leg.status = LegStatus.ACTIVE
                leg.entry_time = datetime.now().isoformat()

                # Set stop-loss
                sl_pct = leg.sl_percentage if leg.sl_percentage > 0 else Config.SL_PERCENTAGE
                leg.sl_price = round(leg.entry_price * (1 + sl_pct / 100.0), 2)
                leg.sl_percentage = sl_pct

                self.db.save_leg(leg)
                self.db.log_order(leg.leg_id, order_id, "sell", "filled",
                                  leg.entry_price, leg.quantity)
                self.state.add_log("INFO", "OrderMgr",
                                   f"FILLED: {leg.stock_code} {leg.strike_price}"
                                   f"{leg.right.value[0].upper()} @ {leg.entry_price}")
                return True

            # Not filled – cancel and retry
            self.session.cancel_order(order_id)
            self.db.log_order(leg.leg_id, order_id, "sell", "cancelled",
                              order_price, leg.quantity, "chase timeout")
            self.state.add_log("INFO", "OrderMgr",
                               f"Chase cancel attempt {attempt + 1}")
            time.sleep(0.2)

        # Exhausted retries → fall back to market order
        self.state.add_log("WARN", "OrderMgr", "Chase exhausted → MARKET order")
        return self._execute_market(leg, "sell")

    def execute_buy_market(self, leg: Leg) -> bool:
        """Immediate market buy (for SL exits or panic)."""
        return self._execute_market(leg, "buy")

    def _execute_market(self, leg: Leg, action: str) -> bool:
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
            self.state.add_log("ERROR", "OrderMgr", f"Market order failed: {error}")
            self.db.log_order(leg.leg_id, "", action, "rejected", 0, leg.quantity, str(error))
            return False

        order_id = result["Success"]["order_id"]

        # Wait a brief moment for market fill
        filled = self._wait_for_fill(order_id, timeout=5.0)
        fill_price = self._get_fill_price(order_id)
        if fill_price <= 0:
            fill_price = self._get_current_price(leg)

        if action == "sell":
            leg.entry_order_id = order_id
            leg.entry_price = fill_price
            leg.current_price = fill_price
            leg.status = LegStatus.ACTIVE
            leg.entry_time = datetime.now().isoformat()
            sl_pct = leg.sl_percentage if leg.sl_percentage > 0 else Config.SL_PERCENTAGE
            leg.sl_price = round(leg.entry_price * (1 + sl_pct / 100.0), 2)
            leg.sl_percentage = sl_pct
        else:
            leg.exit_order_id = order_id
            leg.exit_price = fill_price
            leg.current_price = fill_price
            leg.status = LegStatus.SQUARED_OFF
            leg.exit_time = datetime.now().isoformat()
            leg.compute_pnl()

        self.db.save_leg(leg)
        self.db.log_order(leg.leg_id, order_id, action,
                          "filled" if filled else "assumed_filled",
                          fill_price, leg.quantity)
        self.state.add_log("INFO", "OrderMgr",
                           f"MARKET {action.upper()}: {leg.stock_code} "
                           f"{leg.strike_price}{leg.right.value[0].upper()} @ {fill_price}")
        return True

    def _wait_for_fill(self, order_id: str, timeout: float) -> bool:
        """Poll order status until filled or timeout."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            detail = self.session.get_order_detail(order_id)
            if detail and detail.get("Success"):
                records = detail["Success"]
                if isinstance(records, list) and records:
                    status = records[0].get("status", "").lower()
                    if status in ("filled", "complete", "executed"):
                        return True
                    if status in ("rejected", "cancelled"):
                        return False
            time.sleep(0.3)
        return False

    def _get_fill_price(self, order_id: str) -> float:
        detail = self.session.get_order_detail(order_id)
        if detail and detail.get("Success"):
            records = detail["Success"]
            if isinstance(records, list) and records:
                from utils import safe_float
                return safe_float(records[0].get("fill_price",
                                  records[0].get("price", 0)))
        return 0.0

    def _get_current_price(self, leg: Leg) -> float:
        """Get LTP from SharedState ticks, falling back to API."""
        tick = self.state.get_tick(leg.feed_key)
        if tick and tick.ltp > 0:
            return tick.ltp
        # Fallback: query API
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
                    from utils import safe_float
                    return safe_float(records[0].get("ltp", 0))
        except Exception as e:
            LOG.error(f"get_quotes fallback failed: {e}")
        return 0.0
