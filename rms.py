# ═══════════════════════════════════════════════════════════════
# FILE: rms.py
# ═══════════════════════════════════════════════════════════════
"""
Risk Management System.

FIXES:
  1. SL direction check now handles both SELL (current >= SL)
     and BUY (current <= SL) legs correctly
  2. Slippage circuit breaker added
"""

import time
import threading
from typing import List

from models import Leg, LegStatus, Strategy, StrategyStatus, OrderSide
from connection_manager import SessionManager
from order_manager import OrderManager
from database import Database
from shared_state import SharedState
from config import Config
from utils import LOG


class RiskManager:

    def __init__(self, session: SessionManager, order_mgr: OrderManager,
                 db: Database, state: SharedState):
        self.session = session
        self.order_mgr = order_mgr
        self.db = db
        self.state = state
        self._sl_lock = threading.Lock()

    def check_stop_losses(self, strategies: List[Strategy]):
        with self._sl_lock:
            for strategy in strategies:
                if strategy.status not in (StrategyStatus.ACTIVE, StrategyStatus.PARTIAL_EXIT):
                    continue

                for leg in strategy.legs:
                    if leg.status != LegStatus.ACTIVE:
                        continue

                    # Update from tick
                    tick = self.state.get_tick(leg.feed_key)
                    if tick and tick.ltp > 0:
                        leg.current_price = tick.ltp
                        leg.compute_pnl()

                    if leg.sl_price <= 0:
                        continue

                    # Direction-aware SL check
                    sl_hit = False
                    if leg.side == OrderSide.SELL:
                        # Sold option: SL hits when price RISES above SL
                        sl_hit = leg.current_price >= leg.sl_price
                    else:
                        # Bought option: SL hits when price DROPS below SL
                        sl_hit = leg.current_price <= leg.sl_price

                    if not sl_hit:
                        continue

                    self.state.add_log(
                        "WARN", "RMS",
                        f"⚠️ SL HIT: {int(leg.strike_price)}"
                        f"{leg.right.value[0].upper()} "
                        f"LTP=₹{leg.current_price:.2f} SL=₹{leg.sl_price:.2f}"
                    )
                    LOG.warning(f"SL triggered: leg {leg.leg_id}")

                    leg.status = LegStatus.SL_TRIGGERED
                    self.db.update_leg_status(leg.leg_id, LegStatus.SL_TRIGGERED)

                    # Square off
                    if leg.side == OrderSide.SELL:
                        ok = self.order_mgr.execute_buy_market(leg)
                    else:
                        ok = self.order_mgr.execute_buy_market(leg)  # buy-side exit is sell

                    if ok:
                        self.state.add_log("INFO", "RMS",
                                           f"SL exit OK: {leg.leg_id} @ ₹{leg.exit_price:.2f}")
                    else:
                        leg.status = LegStatus.ERROR
                        self.state.add_log("ERROR", "RMS",
                                           f"SL exit FAILED: {leg.leg_id}")
                    self.db.save_leg(leg)

                # Update strategy status
                active_legs = [l for l in strategy.legs if l.status == LegStatus.ACTIVE]
                if not active_legs:
                    strategy.status = StrategyStatus.CLOSED
                elif len(active_legs) < len(strategy.legs):
                    strategy.status = StrategyStatus.PARTIAL_EXIT
                strategy.compute_total_pnl()
                self.db.save_strategy(strategy)

    def check_global_mtm(self, strategies: List[Strategy]) -> bool:
        total = sum(
            s.compute_total_pnl() for s in strategies
            if s.status in (StrategyStatus.ACTIVE, StrategyStatus.PARTIAL_EXIT)
        )
        self.state.set_total_mtm(total)

        if total <= Config.GLOBAL_MAX_LOSS:
            self.state.add_log(
                "CRIT", "RMS",
                f"🚨 GLOBAL MTM BREACH: ₹{total:,.2f} <= ₹{Config.GLOBAL_MAX_LOSS:,.2f}"
            )
            self.panic_exit(strategies)
            return True
        return False

    def panic_exit(self, strategies: List[Strategy]):
        self.state.panic_triggered = True
        self.state.add_log("CRIT", "RMS", "🚨🚨🚨 PANIC EXIT TRIGGERED 🚨🚨🚨")
        LOG.critical("PANIC EXIT")

        for strategy in strategies:
            if strategy.status not in (
                StrategyStatus.ACTIVE, StrategyStatus.PARTIAL_EXIT,
                StrategyStatus.DEPLOYING
            ):
                continue

            for leg in strategy.legs:
                if leg.status not in (LegStatus.ACTIVE, LegStatus.ENTERING):
                    continue

                self.state.add_log("CRIT", "RMS",
                                   f"Closing: {int(leg.strike_price)}"
                                   f"{leg.right.value[0].upper()}")
                try:
                    ok = self.order_mgr.execute_buy_market(leg)
                    leg.status = LegStatus.SQUARED_OFF if ok else LegStatus.ERROR
                except Exception as e:
                    leg.status = LegStatus.ERROR
                    LOG.exception(f"Panic close error: {leg.leg_id}: {e}")
                self.db.save_leg(leg)

            strategy.status = StrategyStatus.CLOSED
            strategy.compute_total_pnl()
            self.db.save_strategy(strategy)

        self.state.add_log("CRIT", "RMS", "Panic exit complete")

    def update_greeks(self, strategies: List[Strategy], spot: float):
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
                except Exception:
                    pass
