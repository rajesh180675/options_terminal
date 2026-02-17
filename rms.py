"""
rms.py

Professional RMS (Risk Management System):
  - per-leg stop loss (direction-aware)
  - trailing SL for short options (tighten only)
  - global MTM kill switch
  - auto-exit before close
  - portfolio greeks aggregation (net Δ Γ Θ V)
  - alert hooks (optional)

This module only acts; it does not place new strategies.
"""

from __future__ import annotations

import threading
from typing import List, Optional

from app_config import Config
from models import Strategy, StrategyStatus, Leg, LegStatus, OrderSide, Greeks
from shared_state import SharedState
from database import Database
from order_manager import OrderManager
from utils import LOG, is_auto_exit_time
from greeks_engine import BlackScholes, time_to_expiry


class RiskManager:
    def __init__(self, order_mgr: OrderManager, db: Database, state: SharedState):
        self.omgr = order_mgr
        self.db = db
        self.state = state
        self._lock = threading.Lock()
        self.alerts = None  # optional: alerts.AlertManager

    def update_greeks_and_portfolio(self, strategies: List[Strategy], spot_map: dict) -> None:
        """
        Compute Greeks for each ACTIVE leg and aggregate portfolio greeks.
        spot_map: {stock_code: spot}
        """
        r = Config.RISK_FREE_RATE
        pg = Greeks()

        for s in strategies:
            spot = float(spot_map.get(s.stock_code, 0.0))
            if spot <= 0:
                continue

            for leg in s.legs:
                if leg.status != LegStatus.ACTIVE:
                    continue
                if leg.current_price <= 0:
                    continue

                T = time_to_expiry(leg.expiry_date)
                if T <= 0:
                    continue

                try:
                    iv = leg.greeks.iv if leg.greeks.iv > 0 else BlackScholes.implied_vol(
                        leg.current_price, spot, leg.strike_price, T, r, leg.right
                    )
                    leg.greeks = BlackScholes.greeks(
                        spot, leg.strike_price, T, r, iv, leg.right
                    )

                    sign = -1.0 if leg.side == OrderSide.SELL else 1.0
                    pg.delta += leg.greeks.delta * sign * leg.quantity
                    pg.gamma += leg.greeks.gamma * sign * leg.quantity
                    pg.theta += leg.greeks.theta * sign * leg.quantity
                    pg.vega += leg.greeks.vega * sign * leg.quantity

                    # Persist greeks snapshot optionally via db.save_leg (expensive).
                    # We avoid saving every loop to reduce IO. Leg greeks exist in memory for UI.

                except Exception:
                    # Never fail the monitor loop due to greek calc
                    continue

        self.state.set_portfolio_greeks(pg)

    def apply_trailing_sl(self, leg: Leg) -> None:
        """
        Trailing SL algorithm for short options:
          - lowest_price tracks best favorable price (downwards)
          - once profit% >= TRAIL_ACTIVATION_PCT, set trail_sl = lowest*(1+TRAIL_SL_PCT)
          - SL only tightens (decreases), never widens
          - floor at entry price (breakeven lock)
        """
        if not Config.TRAIL_ENABLED:
            return
        if leg.side != OrderSide.SELL:
            return
        if leg.entry_price <= 0 or leg.current_price <= 0:
            return

        # Update lowest price
        if leg.lowest_price <= 0 or leg.current_price < leg.lowest_price:
            leg.lowest_price = leg.current_price

        profit_pct = (leg.entry_price - leg.current_price) / leg.entry_price * 100.0
        if profit_pct < Config.TRAIL_ACTIVATION_PCT:
            return

        trail_sl = round(leg.lowest_price * (1 + Config.TRAIL_SL_PCT / 100.0), 2)

        # Only tighten
        if leg.sl_price <= 0:
            # initialize if missing
            leg.sl_price = round(leg.entry_price * (1 + Config.SL_PERCENTAGE / 100.0), 2)

        if trail_sl < leg.sl_price:
            old = leg.sl_price
            leg.sl_price = max(trail_sl, leg.entry_price)  # breakeven floor
            leg.trailing_active = True
            if abs(old - leg.sl_price) > 0.5:
                self.state.add_log(
                    "INFO", "RMS",
                    f"Trail SL tightened: {leg.display_name} {old:.2f} -> {leg.sl_price:.2f}"
                )

    def check_stop_losses(self, strategies: List[Strategy]) -> None:
        """Per-leg SL. Exits are immediate market buyback for short legs."""
        with self._lock:
            for s in strategies:
                if s.status not in (StrategyStatus.ACTIVE, StrategyStatus.PARTIAL_EXIT):
                    continue

                for leg in s.legs:
                    if leg.status != LegStatus.ACTIVE:
                        continue

                    # Trailing update before check
                    self.apply_trailing_sl(leg)

                    if leg.sl_price <= 0:
                        continue

                    if leg.side == OrderSide.SELL:
                        sl_hit = leg.current_price >= leg.sl_price
                    else:
                        sl_hit = leg.current_price <= leg.sl_price

                    if not sl_hit:
                        continue

                    tag = " (trail)" if leg.trailing_active else ""
                    self.state.add_log(
                        "WARN", "RMS",
                        f"SL HIT{tag}: {leg.display_name} LTP={leg.current_price:.2f} SL={leg.sl_price:.2f}"
                    )
                    LOG.warning(f"SL hit {leg.leg_id}: {leg.display_name}")

                    # Mark SL triggered and persist
                    leg.status = LegStatus.SL_TRIGGERED
                    self.db.update_leg_status(leg.leg_id, LegStatus.SL_TRIGGERED)

                    # Exit now
                    ok = self.omgr.buy_market(leg)
                    if ok:
                        self.state.add_log("INFO", "RMS", f"SL exit OK: {leg.display_name}")
                        if self.alerts:
                            try:
                                self.alerts.sl_hit(leg.display_name, leg.current_price, leg.sl_price)
                            except Exception:
                                pass
                    else:
                        leg.status = LegStatus.ERROR
                        self.state.add_log("ERROR", "RMS", f"SL exit FAILED: {leg.display_name}")

                    self.db.save_leg(leg)

                # Update strategy status after processing legs
                active_legs = [l for l in s.legs if l.status == LegStatus.ACTIVE]
                if not active_legs:
                    s.status = StrategyStatus.CLOSED
                elif len(active_legs) < len(s.legs):
                    s.status = StrategyStatus.PARTIAL_EXIT
                s.compute_total_pnl()
                self.db.save_strategy(s)

    def check_global_mtm(self, strategies: List[Strategy]) -> bool:
        """Kill switch on global MTM drawdown."""
        total = 0.0
        for s in strategies:
            if s.status in (StrategyStatus.ACTIVE, StrategyStatus.PARTIAL_EXIT):
                total += s.compute_total_pnl()

        self.state.set_total_mtm(total)

        if total <= Config.GLOBAL_MAX_LOSS:
            self.state.add_log(
                "CRIT", "RMS",
                f"GLOBAL MTM BREACH: {total:.2f} <= {Config.GLOBAL_MAX_LOSS:.2f}"
            )
            self.panic_exit(strategies, reason="global_mtm")
            return True
        return False

    def check_auto_exit(self, strategies: List[Strategy]) -> bool:
        """Auto exit at configured time once per day."""
        if not Config.AUTO_EXIT_ENABLED:
            return False
        if self.state.auto_exit_done:
            return False
        if not is_auto_exit_time():
            return False

        self.state.add_log(
            "WARN", "RMS",
            f"AUTO EXIT triggered at {Config.AUTO_EXIT_HOUR}:{Config.AUTO_EXIT_MINUTE:02d}"
        )
        self.panic_exit(strategies, reason="auto_exit")
        self.state.auto_exit_done = True
        return True

    def panic_exit(self, strategies: List[Strategy], reason: str = "panic") -> None:
        """
        Close all ACTIVE/ENTERING legs at market.
        (We prefer correctness over parallel execution here.)
        """
        self.state.panic_triggered = True
        self.state.add_log("CRIT", "RMS", f"PANIC EXIT: {reason}")
        LOG.critical(f"PANIC EXIT: {reason}")

        for s in strategies:
            if s.status not in (StrategyStatus.ACTIVE, StrategyStatus.PARTIAL_EXIT, StrategyStatus.DEPLOYING):
                continue

            for leg in s.legs:
                if leg.status not in (LegStatus.ACTIVE, LegStatus.ENTERING, LegStatus.SL_TRIGGERED):
                    continue

                self.state.add_log("CRIT", "RMS", f"Closing: {leg.display_name}")
                try:
                    ok = self.omgr.buy_market(leg) if leg.side == OrderSide.SELL else self.omgr.buy_market(leg)
                    leg.status = LegStatus.SQUARED_OFF if ok else LegStatus.ERROR
                except Exception as e:
                    leg.status = LegStatus.ERROR
                    self.state.add_log("ERROR", "RMS", f"Panic close error: {leg.display_name}: {e}")
                self.db.save_leg(leg)

            s.status = StrategyStatus.CLOSED
            s.compute_total_pnl()
            self.db.save_strategy(s)

        if self.alerts:
            try:
                self.alerts.panic_exit(reason)
            except Exception:
                pass

        self.state.add_log("CRIT", "RMS", "Panic exit complete")
