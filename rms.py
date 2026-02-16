# ═══════════════════════════════════════════════════════════════
# FILE: rms.py
# ═══════════════════════════════════════════════════════════════
"""
Risk Management System:
  • Per-leg stop-loss monitoring
  • Global MTM kill threshold
  • panic_exit() — immediate market close of all NFO positions
  • Slippage & latency circuit breakers
"""

import time
import threading
from typing import List

from models import Leg, LegStatus, Strategy, StrategyStatus
from connection_manager import SessionManager
from order_manager import OrderManager
from database import Database
from shared_state import SharedState
from config import Config
from utils import LOG


class RiskManager:
    """Continuously monitors risk and enforces limits."""

    def __init__(
        self,
        session: SessionManager,
        order_mgr: OrderManager,
        db: Database,
        state: SharedState,
    ):
        self.session = session
        self.order_mgr = order_mgr
        self.db = db
        self.state = state
        self._sl_lock = threading.Lock()

    def check_stop_losses(self, strategies: List[Strategy]):
        """
        Iterate all active legs; if current_price >= sl_price for a short,
        trigger an immediate market buyback.
        """
        with self._sl_lock:
            for strategy in strategies:
                if strategy.status not in (StrategyStatus.ACTIVE, StrategyStatus.PARTIAL_EXIT):
                    continue
                for leg in strategy.legs:
                    if leg.status != LegStatus.ACTIVE:
                        continue
                    # Update current price from ticks
                    tick = self.state.get_tick(leg.feed_key)
                    if tick and tick.ltp > 0:
                        leg.current_price = tick.ltp
                        leg.compute_pnl()

                    if leg.sl_price > 0 and leg.current_price >= leg.sl_price:
                        self.state.add_log(
                            "WARN", "RMS",
                            f"SL HIT: {leg.stock_code} {leg.strike_price}"
                            f"{leg.right.value[0].upper()} "
                            f"LTP={leg.current_price} >= SL={leg.sl_price}"
                        )
                        LOG.warning(f"SL triggered for leg {leg.leg_id}")
                        leg.status = LegStatus.SL_TRIGGERED
                        self.db.update_leg_status(leg.leg_id, LegStatus.SL_TRIGGERED)

                        # Square off immediately
                        success = self.order_mgr.execute_buy_market(leg)
                        if success:
                            leg.status = LegStatus.SQUARED_OFF
                            self.state.add_log("INFO", "RMS",
                                               f"SL exit filled: {leg.leg_id} "
                                               f"exit_price={leg.exit_price}")
                        else:
                            leg.status = LegStatus.ERROR
                            self.state.add_log("ERROR", "RMS",
                                               f"SL exit FAILED for {leg.leg_id}")
                        self.db.save_leg(leg)

                        # Check if strategy is now fully closed
                        active_legs = [l for l in strategy.legs
                                       if l.status == LegStatus.ACTIVE]
                        if not active_legs:
                            strategy.status = StrategyStatus.CLOSED
                        else:
                            strategy.status = StrategyStatus.PARTIAL_EXIT
                        strategy.compute_total_pnl()
                        self.db.save_strategy(strategy)

    def check_global_mtm(self, strategies: List[Strategy]) -> bool:
        """
        If total MTM across all strategies breaches GLOBAL_MAX_LOSS,
        trigger panic_exit.
        Returns True if panic was triggered.
        """
        total_mtm = sum(s.compute_total_pnl() for s in strategies
                        if s.status in (StrategyStatus.ACTIVE, StrategyStatus.PARTIAL_EXIT))
        self.state.set_total_mtm(total_mtm)

        if total_mtm <= Config.GLOBAL_MAX_LOSS:
            self.state.add_log(
                "CRIT", "RMS",
                f"GLOBAL MTM BREACH: {total_mtm:.2f} <= {Config.GLOBAL_MAX_LOSS}"
            )
            LOG.critical(f"Global MTM breach: {total_mtm}")
            self.panic_exit(strategies)
            return True
        return False

    def panic_exit(self, strategies: List[Strategy]):
        """
        KILL SWITCH: close all active legs at market price immediately.
        Uses a high-priority sequential loop — no fancy parallelism here;
        correctness > speed when the house is on fire.
        """
        self.state.panic_triggered = True
        self.state.add_log("CRIT", "RMS", "🚨 PANIC EXIT TRIGGERED 🚨")
        LOG.critical("PANIC EXIT TRIGGERED")

        for strategy in strategies:
            if strategy.status not in (StrategyStatus.ACTIVE, StrategyStatus.PARTIAL_EXIT,
                                       StrategyStatus.DEPLOYING):
                continue
            for leg in strategy.legs:
                if leg.status not in (LegStatus.ACTIVE, LegStatus.ENTERING):
                    continue
                self.state.add_log("CRIT", "RMS",
                                   f"Panic closing: {leg.stock_code} {leg.strike_price}"
                                   f"{leg.right.value[0].upper()}")
                try:
                    success = self.order_mgr.execute_buy_market(leg)
                    if success:
                        leg.status = LegStatus.SQUARED_OFF
                        self.state.add_log("INFO", "RMS",
                                           f"Panic close OK: {leg.leg_id}")
                    else:
                        leg.status = LegStatus.ERROR
                        self.state.add_log("ERROR", "RMS",
                                           f"Panic close FAILED: {leg.leg_id}")
                except Exception as e:
                    leg.status = LegStatus.ERROR
                    self.state.add_log("ERROR", "RMS",
                                       f"Panic exception: {leg.leg_id}: {e}")
                    LOG.exception(f"Panic close exception for {leg.leg_id}")
                self.db.save_leg(leg)

            strategy.status = StrategyStatus.CLOSED
            strategy.compute_total_pnl()
            self.db.save_strategy(strategy)

        self.state.add_log("CRIT", "RMS", "Panic exit complete")
        LOG.critical("Panic exit complete")

    def update_greeks(self, strategies: List[Strategy], spot: float):
        """Refresh Greeks for all active legs using local BS engine."""
        from greeks_engine import BlackScholes, compute_time_to_expiry
        r = Config.RISK_FREE_RATE

        for strategy in strategies:
            for leg in strategy.legs:
                if leg.status != LegStatus.ACTIVE:
                    continue
                if leg.current_price <= 0 or spot <= 0:
                    continue
                T = compute_time_to_expiry(leg.expiry_date)
                try:
                    iv = BlackScholes.implied_vol(
                        leg.current_price, spot, leg.strike_price, T, r, leg.right
                    )
                    leg.greeks = BlackScholes.greeks(
                        spot, leg.strike_price, T, r, iv, leg.right
                    )
                except Exception as e:
                    LOG.debug(f"Greek calc error for {leg.leg_id}: {e}")
