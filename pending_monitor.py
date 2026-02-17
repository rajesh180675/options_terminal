# ═══════════════════════════════════════════════════════════════
# FILE: pending_monitor.py  (NEW — fills the critical gap)
# ═══════════════════════════════════════════════════════════════
"""
Background monitor for pending/ENTERING orders.

Problem this solves:
  User places a limit sell at 10:30.  The order sits pending.
  At 10:50, the market moves and the order fills.
  Without this monitor, our system NEVER detects the fill.
  The leg stays in ENTERING status forever.

Solution:
  Every PENDING_POLL_INTERVAL seconds, poll all ENTERING legs'
  order IDs and update their status.
"""

import time
import threading
from typing import List

from models import Leg, LegStatus, OrderSide
from connection_manager import SessionManager
from database import Database
from shared_state import SharedState
from app_config import Config
from utils import LOG, safe_float
from datetime import datetime

_FILLED = {"executed", "filled", "complete", "traded"}
_DEAD = {"rejected", "cancelled", "canceled"}


class PendingOrderMonitor:
    """Daemon thread that checks pending order status."""

    def __init__(self, session: SessionManager, db: Database, state: SharedState):
        self.session = session
        self.db = db
        self.state = state
        self._thread = None
        self._running = False

    def start(self):
        self._running = True
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="PendingMon"
        )
        self._thread.start()
        LOG.info("Pending order monitor started")

    def stop(self):
        self._running = False

    def _loop(self):
        while self._running:
            try:
                self._check_entering_legs()
            except Exception as e:
                LOG.error(f"Pending monitor error: {e}")
            time.sleep(Config.PENDING_POLL_INTERVAL)

    def _check_entering_legs(self):
        """Find all legs in ENTERING status and check their order."""
        from database import Database as DB
        # Get fresh connection in this thread
        db = Database(Config.DB_PATH)
        try:
            legs = db.get_legs_by_status(LegStatus.ENTERING)
        except Exception:
            legs = []
            # Table might not have the method yet; graceful fallback
            try:
                all_active = db.get_active_strategies()
                for s in all_active:
                    for l in s.legs:
                        if l.status == LegStatus.ENTERING:
                            legs.append(l)
            except Exception:
                return

        for leg in legs:
            if not leg.entry_order_id:
                continue

            try:
                detail = self.session.get_order_detail(
                    leg.entry_order_id, leg.exchange_code
                )
                if not detail or not detail.get("Success"):
                    continue

                records = detail["Success"]
                if not isinstance(records, list) or not records:
                    continue

                status = str(records[0].get("status", "")).strip().lower()

                if status in _FILLED:
                    # Order filled! Update leg
                    fp = safe_float(records[0].get("average_price", 0))
                    if fp <= 0:
                        fp = safe_float(records[0].get("price", 0))
                    if fp <= 0:
                        fp = leg.entry_price if leg.entry_price > 0 else 0

                    leg.entry_price = fp
                    leg.current_price = fp
                    leg.lowest_price = fp
                    leg.status = LegStatus.ACTIVE
                    leg.entry_time = datetime.now().isoformat()
                    sl_pct = leg.sl_percentage if leg.sl_percentage > 0 else Config.SL_PERCENTAGE
                    leg.sl_price = round(fp * (1 + sl_pct / 100.0), 2)

                    db.save_leg(leg)
                    db.log_order(
                        leg.leg_id, leg.entry_order_id, "sell", "filled",
                        fp, leg.quantity, "detected by pending monitor"
                    )

                    self.state.add_log(
                        "INFO", "PendMon",
                        f"Order FILLED (detected): {leg.display_name} @ ₹{fp:.2f}"
                    )
                    LOG.info(f"Pending monitor: {leg.leg_id} filled @ {fp}")

                    # Update strategy status
                    strategies = db.get_active_strategies()
                    for s in strategies:
                        if s.strategy_id == leg.strategy_id:
                            from models import StrategyStatus
                            entering = [l for l in s.legs if l.status == LegStatus.ENTERING]
                            if not entering:
                                s.status = StrategyStatus.ACTIVE
                                db.save_strategy(s)

                elif status in _DEAD:
                    leg.status = LegStatus.ERROR
                    db.save_leg(leg)
                    db.log_order(
                        leg.leg_id, leg.entry_order_id, "sell", status,
                        0, leg.quantity, "detected by pending monitor"
                    )
                    self.state.add_log(
                        "WARN", "PendMon",
                        f"Order {status.upper()}: {leg.display_name}"
                    )

            except Exception as e:
                LOG.error(f"Pending check error for {leg.leg_id}: {e}")
