"""
execution_engine.py — Order execution with "Chase" algorithm.
If a limit order isn't filled within the timeout, cancel and re-place
at the new LTP.  Falls back to market order after max retries.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Literal

from config import CFG
from connection_manager import ConnectionManager
from database import TradingDB
from shared_state import SharedState


@dataclass
class OrderResult:
    success: bool
    order_id: str = ""
    breeze_order_id: str = ""
    filled_price: float = 0.0
    filled_qty: int = 0
    status: str = ""
    error: str = ""


class ExecutionEngine:
    """
    Handles order placement with chase logic and state tracking.
    """

    def __init__(self, conn: ConnectionManager, db: TradingDB,
                 state: SharedState):
        self.conn = conn
        self.db = db
        self.state = state
        self._lock = threading.Lock()

    def execute_with_chase(
        self,
        strategy_id: str,
        stock_code: str,
        action: Literal["sell", "buy"],
        strike_price: float,
        right: str,
        expiry_date: str,
        quantity: int,
        initial_price: float,
        leg_tag: str = "",
        timeout: float = 0.0,
        max_retries: int = 0,
    ) -> OrderResult:
        """
        Place a limit order and chase the price if not filled.

        1. Place limit order at `initial_price`
        2. Wait `timeout` seconds (default from config)
        3. If not filled, cancel and re-place at current LTP
        4. After `max_retries`, place as market order
        """
        timeout = timeout or CFG.chase_timeout_seconds
        max_retries = max_retries or CFG.chase_max_retries

        # Create DB record
        db_oid = self.db.create_order(
            strategy_id=strategy_id,
            stock_code=stock_code,
            action=action,
            strike_price=strike_price,
            right_type=right,
            expiry_date=expiry_date,
            quantity=quantity,
            price=initial_price,
            order_type="limit",
            leg_tag=leg_tag,
        )

        current_price = initial_price
        breeze_oid = ""

        for attempt in range(max_retries + 1):
            is_last = (attempt == max_retries)
            order_type = "market" if is_last else "limit"
            place_price = 0.0 if is_last else current_price

            self.state.add_log(
                "INFO", "Execution",
                f"Chase attempt {attempt + 1}/{max_retries + 1}: "
                f"{action} {stock_code} {strike_price}{right[0].upper()}E "
                f"@{'MKT' if is_last else round(place_price, 2)} "
                f"qty={quantity}",
            )

            # Place order
            resp = self.conn.place_order(
                stock_code=stock_code,
                action=action,
                strike_price=strike_price,
                right=right,
                expiry_date=expiry_date,
                quantity=quantity,
                price=place_price,
                order_type=order_type,
            )

            if resp.get("Error") or not resp.get("Success"):
                error_msg = resp.get("Error", "Unknown error")
                self.state.add_log("ERROR", "Execution",
                                   f"Order place failed: {error_msg}")
                self.db.update_order(db_oid, status="failed")
                return OrderResult(success=False, order_id=db_oid,
                                   error=str(error_msg))

            breeze_oid = resp["Success"].get("order_id", "")
            self.db.update_order(db_oid, breeze_order_id=breeze_oid,
                                 status="placed",
                                 chase_count=attempt + 1)

            if order_type == "market":
                # Market orders fill immediately (or near-immediately)
                time.sleep(0.5)
                fill_status = self._check_fill(breeze_oid)
                if fill_status:
                    self.db.update_order(
                        db_oid,
                        status="filled",
                        filled_price=fill_status["filled_price"],
                        filled_qty=fill_status["filled_qty"],
                    )
                    self.state.add_log(
                        "INFO", "Execution",
                        f"Market order filled: {breeze_oid} "
                        f"@{fill_status['filled_price']}",
                    )
                    return OrderResult(
                        success=True,
                        order_id=db_oid,
                        breeze_order_id=breeze_oid,
                        filled_price=fill_status["filled_price"],
                        filled_qty=fill_status["filled_qty"],
                        status="filled",
                    )
                # Even market order unfilled? Treat as failure
                self.db.update_order(db_oid, status="failed")
                return OrderResult(success=False, order_id=db_oid,
                                   breeze_order_id=breeze_oid,
                                   error="Market order not filled")

            # --- Limit order: poll for fill ---
            filled = self._wait_for_fill(breeze_oid, timeout)
            if filled:
                self.db.update_order(
                    db_oid,
                    status="filled",
                    filled_price=filled["filled_price"],
                    filled_qty=filled["filled_qty"],
                )
                self.state.add_log(
                    "INFO", "Execution",
                    f"Limit filled: {breeze_oid} @{filled['filled_price']}",
                )
                return OrderResult(
                    success=True,
                    order_id=db_oid,
                    breeze_order_id=breeze_oid,
                    filled_price=filled["filled_price"],
                    filled_qty=filled["filled_qty"],
                    status="filled",
                )

            # --- Not filled: cancel and get new price ---
            self.state.add_log("INFO", "Execution",
                               f"Order not filled. Cancelling {breeze_oid}")
            self.conn.cancel_order(breeze_oid)
            time.sleep(0.3)

            # Get new LTP
            new_ltp = self.state.get_ltp(stock_code, strike_price, right)
            if new_ltp > 0:
                current_price = new_ltp
            else:
                # Adjust price slightly to improve fill probability
                if action == "sell":
                    current_price *= 0.998  # sell at slightly lower
                else:
                    current_price *= 1.002  # buy at slightly higher

            self.db.update_order(db_oid, status="chasing",
                                 price=current_price)

        # Should not reach here (last attempt is market order)
        self.db.update_order(db_oid, status="failed")
        return OrderResult(success=False, order_id=db_oid,
                           error="All chase attempts exhausted")

    def _wait_for_fill(self, breeze_oid: str, timeout: float) -> dict | None:
        """Poll order status until filled or timeout."""
        deadline = time.time() + timeout
        poll_interval = 0.5

        while time.time() < deadline:
            fill = self._check_fill(breeze_oid)
            if fill:
                return fill
            time.sleep(poll_interval)
        return None

    def _check_fill(self, breeze_oid: str) -> dict | None:
        """Check if a Breeze order has been filled."""
        status = self.conn.get_order_status(breeze_oid)
        if not status:
            return None

        order_status = str(status.get("status", "")).lower()
        if order_status in ("filled", "complete", "executed"):
            return {
                "filled_price": float(status.get("filled_price", 0) or
                                      status.get("price", 0)),
                "filled_qty": int(status.get("filled_quantity", 0) or
                                  status.get("quantity", 0)),
            }
        return None

    def square_off_position(
        self,
        strategy_id: str,
        stock_code: str,
        strike_price: float,
        right: str,
        expiry_date: str,
        quantity: int,
        leg_tag: str = "exit",
        use_market: bool = False,
    ) -> OrderResult:
        """Buy back a short position (square off)."""
        current_ltp = self.state.get_ltp(stock_code, strike_price, right)
        if current_ltp <= 0:
            current_ltp = 1.0  # fallback

        if use_market:
            # Direct market order for panic exits
            resp = self.conn.place_order(
                stock_code=stock_code,
                action="buy",
                strike_price=strike_price,
                right=right,
                expiry_date=expiry_date,
                quantity=quantity,
                price=0,
                order_type="market",
            )
            if resp.get("Error"):
                return OrderResult(success=False,
                                   error=str(resp.get("Error")))
            breeze_oid = resp["Success"].get("order_id", "")
            time.sleep(0.5)
            fill = self._check_fill(breeze_oid)
            if fill:
                return OrderResult(
                    success=True,
                    breeze_order_id=breeze_oid,
                    filled_price=fill["filled_price"],
                    filled_qty=fill["filled_qty"],
                    status="filled",
                )
            return OrderResult(success=False, breeze_order_id=breeze_oid,
                               error="Market exit not confirmed")

        # Use chase logic for normal exits
        return self.execute_with_chase(
            strategy_id=strategy_id,
            stock_code=stock_code,
            action="buy",
            strike_price=strike_price,
            right=right,
            expiry_date=expiry_date,
            quantity=quantity,
            initial_price=current_ltp,
            leg_tag=leg_tag,
            timeout=2.0,
            max_retries=3,
        )
