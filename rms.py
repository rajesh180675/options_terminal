# ═══════════════════════════════════════════════════════════════
# FILE: rms.py
# ═══════════════════════════════════════════════════════════════
"""
Risk Management: per-leg SL, trailing SL, global MTM, auto-exit, panic.
"""

import threading
from typing import List
from datetime import datetime
from models import Leg, LegStatus, Strategy, StrategyStatus, OrderSide, Greeks
from connection_manager import SessionManager
from order_manager import OrderManager
from database import Database
from shared_state import SharedState
from app_config import Config
from utils import LOG, is_auto_exit_time


class RiskManager:

    def __init__(self, session: SessionManager, omgr: OrderManager,
                 db: Database, state: SharedState):
        self.session = session
        self.omgr = omgr
        self.db = db
        self.state = state
        self._lock = threading.Lock()

    def check_stop_losses(self, strategies: List[Strategy]):
        with self._lock:
            for s in strategies:
                if s.status not in (StrategyStatus.ACTIVE, StrategyStatus.PARTIAL_EXIT):
                    continue
                for leg in s.legs:
                    if leg.status != LegStatus.ACTIVE:
                        continue
                    tick = self.state.get_tick(leg.feed_key)
                    if tick and tick.ltp > 0:
                        leg.current_price = tick.ltp
                        leg.compute_pnl()

                    # ── Trailing SL logic ─────────────────────
                    if (Config.TRAIL_ENABLED and leg.side == OrderSide.SELL
                            and leg.entry_price > 0 and leg.current_price > 0):
                        # Update lowest price seen
                        if leg.lowest_price <= 0 or leg.current_price < leg.lowest_price:
                            leg.lowest_price = leg.current_price
                        # Check if trailing should activate
                        profit_pct = (leg.entry_price - leg.current_price) / leg.entry_price * 100
                        if profit_pct >= Config.TRAIL_ACTIVATION_PCT:
                            trail_sl = round(
                                leg.lowest_price * (1 + Config.TRAIL_SL_PCT / 100), 2)
                            # Only tighten (decrease) SL, never widen
                            if trail_sl < leg.sl_price:
                                old_sl = leg.sl_price
                                leg.sl_price = max(trail_sl, leg.entry_price)
                                leg.trailing_active = True
                                if abs(old_sl - leg.sl_price) > 0.5:
                                    self.state.add_log("INFO", "RMS",
                                        f"Trail SL: {leg.display_name} "
                                        f"₹{old_sl:.0f}→₹{leg.sl_price:.0f}")
                        self.db.update_leg_price(
                            leg.leg_id, leg.current_price, leg.pnl,
                            sl=leg.sl_price, lowest=leg.lowest_price,
                            trailing=leg.trailing_active)
                    else:
                        self.db.update_leg_price(
                            leg.leg_id, leg.current_price, leg.pnl)

                    # ── SL check ──────────────────────────────
                    if leg.sl_price <= 0:
                        continue
                    sl_hit = (leg.current_price >= leg.sl_price
                              if leg.side == OrderSide.SELL
                              else leg.current_price <= leg.sl_price)
                    if not sl_hit:
                        continue

                    self.state.add_log("WARN", "RMS",
                        f"⚠️ SL HIT: {leg.display_name} "
                        f"LTP=₹{leg.current_price:.2f} SL=₹{leg.sl_price:.2f}"
                        f" {'(trail)' if leg.trailing_active else ''}")
                    leg.status = LegStatus.SL_TRIGGERED
                    self.db.update_leg_status(leg.leg_id, LegStatus.SL_TRIGGERED)
                    ok = self.omgr.buy_market(leg)
                    if not ok:
                        leg.status = LegStatus.ERROR
                    self.db.save_leg(leg)

                # Update strategy status
                active = [l for l in s.legs if l.status == LegStatus.ACTIVE]
                if not active:
                    s.status = StrategyStatus.CLOSED
                elif len(active) < len(s.legs):
                    s.status = StrategyStatus.PARTIAL_EXIT
                s.compute_total_pnl()
                self.db.save_strategy(s)

    def check_global_mtm(self, strategies):
        total = sum(s.compute_total_pnl() for s in strategies
                    if s.status in (StrategyStatus.ACTIVE, StrategyStatus.PARTIAL_EXIT))
        self.state.set_total_mtm(total)
        if total <= Config.GLOBAL_MAX_LOSS:
            self.state.add_log("CRIT", "RMS",
                f"🚨 MTM BREACH: ₹{total:,.0f} <= ₹{Config.GLOBAL_MAX_LOSS:,.0f}")
            self.panic_exit(strategies)
            return True
        return False

    def check_auto_exit(self, strategies):
        if not Config.AUTO_EXIT_ENABLED or self.state.auto_exit_done:
            return False
        if not is_auto_exit_time():
            return False
        self.state.add_log("WARN", "RMS",
            f"⏰ Auto-exit at {Config.AUTO_EXIT_HOUR}:{Config.AUTO_EXIT_MINUTE:02d}")
        self.panic_exit(strategies, reason="auto_exit")
        self.state.auto_exit_done = True
        return True

    def panic_exit(self, strategies, reason="panic"):
        self.state.panic_triggered = True
        self.state.add_log("CRIT", "RMS", f"🚨 {reason.upper()} EXIT TRIGGERED 🚨")
        LOG.critical(f"PANIC EXIT: {reason}")
        for s in strategies:
            if s.status not in (StrategyStatus.ACTIVE, StrategyStatus.PARTIAL_EXIT,
                                StrategyStatus.DEPLOYING):
                continue
            for leg in s.legs:
                if leg.status not in (LegStatus.ACTIVE, LegStatus.ENTERING):
                    continue
                self.state.add_log("CRIT", "RMS", f"Closing: {leg.display_name}")
                try:
                    ok = self.omgr.buy_market(leg)
                    leg.status = LegStatus.SQUARED_OFF if ok else LegStatus.ERROR
                except Exception as e:
                    leg.status = LegStatus.ERROR
                    LOG.exception(f"Panic error: {leg.leg_id}: {e}")
                self.db.save_leg(leg)
            s.status = StrategyStatus.CLOSED
            s.compute_total_pnl()
            self.db.save_strategy(s)
        self.state.add_log("CRIT", "RMS", f"{reason.upper()} exit complete")

    def update_greeks(self, strategies, spot):
        from greeks_engine import BlackScholes, time_to_expiry
        r = Config.RISK_FREE_RATE
        pg = Greeks()  # portfolio greeks
        for s in strategies:
            for leg in s.legs:
                if leg.status != LegStatus.ACTIVE or leg.current_price <= 0 or spot <= 0:
                    continue
                T = time_to_expiry(leg.expiry_date)
                try:
                    iv = BlackScholes.implied_vol(
                        leg.current_price, spot, leg.strike_price, T, r, leg.right)
                    leg.greeks = BlackScholes.greeks(spot, leg.strike_price, T, r, iv, leg.right)
                    sign = -1.0 if leg.side == OrderSide.SELL else 1.0
                    pg.delta += leg.greeks.delta * sign * leg.quantity
                    pg.gamma += leg.greeks.gamma * sign * leg.quantity
                    pg.theta += leg.greeks.theta * sign * leg.quantity
                    pg.vega += leg.greeks.vega * sign * leg.quantity
                except Exception:
                    pass
        self.state.set_portfolio_greeks(pg)
