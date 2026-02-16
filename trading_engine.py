# ═══════════════════════════════════════════════════════════════
# FILE: trading_engine.py
# ═══════════════════════════════════════════════════════════════
"""
Central orchestrator.
  Boot → recover → monitor loop → graceful shutdown.
"""

import time
import threading
from typing import Optional, List

from config import Config
from models import Strategy, StrategyStatus, LegStatus
from database import Database
from shared_state import SharedState
from connection_manager import SessionManager
from order_manager import OrderManager
from strategy_engine import StrategyEngine
from rms import RiskManager
from utils import LOG, breeze_date, next_weekly_expiry


class TradingEngine:
    """
    The main engine that ties session, orders, strategies, and risk together.
    Runs its monitor loop in a daemon thread so Streamlit can read SharedState
    without blocking.
    """

    def __init__(self):
        self.state = SharedState()
        self.db = Database()
        self.session = SessionManager(self.state)
        self.order_mgr = OrderManager(self.session, self.db, self.state)
        self.strategy_engine = StrategyEngine(
            self.session, self.order_mgr, self.db, self.state
        )
        self.risk_mgr = RiskManager(
            self.session, self.order_mgr, self.db, self.state
        )

        self._monitor_thread: Optional[threading.Thread] = None
        self._running = False
        self._strategies: List[Strategy] = []

    # ── Lifecycle ────────────────────────────────────────────

    def start(self) -> bool:
        """Boot sequence: connect → recover → launch monitor."""
        self.state.add_log("INFO", "Engine", "Starting trading engine…")

        # 1. Connect to Breeze
        if not self.session.initialize():
            self.state.add_log("ERROR", "Engine", "Breeze session failed")
            return False

        # 2. Connect WebSocket
        self.session.connect_websocket()
        time.sleep(1.0)

        # 3. Recover orphaned positions
        self._recover_positions()

        # 4. Launch monitor loop
        self._running = True
        self.state.engine_running = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, daemon=True, name="EngineMonitor"
        )
        self._monitor_thread.start()
        self.state.add_log("INFO", "Engine", "Engine running ✓")
        LOG.info("Trading engine started")
        return True

    def stop(self):
        """Graceful shutdown."""
        self._running = False
        self.state.engine_running = False
        self.session.shutdown()
        self.db.close()
        self.state.add_log("INFO", "Engine", "Engine stopped")
        LOG.info("Trading engine stopped")

    # ── Recovery ─────────────────────────────────────────────

    def _recover_positions(self):
        """
        On startup, reload active strategies/legs from the DB.
        Re-subscribe their feeds so tick data flows again.
        """
        strategies = self.db.get_active_strategies()
        if not strategies:
            self.state.add_log("INFO", "Engine", "No positions to recover")
            return

        self.state.add_log("INFO", "Engine",
                           f"Recovering {len(strategies)} active strategies")
        LOG.info(f"Recovering {len(strategies)} strategies from DB")

        for strategy in strategies:
            for leg in strategy.legs:
                if leg.status in (LegStatus.ACTIVE, LegStatus.ENTERING):
                    try:
                        self.session.subscribe_option(
                            leg.stock_code, leg.strike_price,
                            leg.right.value, leg.expiry_date
                        )
                        self.state.add_log("INFO", "Engine",
                                           f"Re-subscribed: {leg.stock_code} "
                                           f"{leg.strike_price}{leg.right.value[0].upper()}")
                    except Exception as e:
                        LOG.error(f"Re-subscribe failed for {leg.leg_id}: {e}")

        self._strategies = strategies
        self.state.set_strategies(strategies)

    # ── Monitor loop ─────────────────────────────────────────

    def _monitor_loop(self):
        """
        Core loop running at ~2 Hz:
          1. Update prices from tick data
          2. Recompute Greeks
          3. Check per-leg stop-losses
          4. Check global MTM
          5. Push updated state
        """
        while self._running:
            try:
                # Reload strategies from DB (picks up any newly deployed ones)
                self._strategies = self.db.get_active_strategies()

                # Get spot price
                stock_code = Config.DEFAULT_STOCK
                spot = self.session.get_spot_price(stock_code)
                if spot > 0:
                    self.state.set_spot(stock_code, spot)

                # Update leg prices from ticks
                for strategy in self._strategies:
                    for leg in strategy.legs:
                        if leg.status != LegStatus.ACTIVE:
                            continue
                        tick = self.state.get_tick(leg.feed_key)
                        if tick and tick.ltp > 0:
                            leg.current_price = tick.ltp
                            leg.compute_pnl()
                            self.db.update_leg_price(leg.leg_id, leg.current_price, leg.pnl)

                # Update Greeks
                if spot > 0:
                    self.risk_mgr.update_greeks(self._strategies, spot)

                # Check stop-losses
                if not self.state.panic_triggered:
                    self.risk_mgr.check_stop_losses(self._strategies)

                # Check global MTM
                if not self.state.panic_triggered:
                    self.risk_mgr.check_global_mtm(self._strategies)

                # Compute and push total MTM
                total_mtm = sum(
                    s.compute_total_pnl() for s in self._strategies
                    if s.status in (StrategyStatus.ACTIVE, StrategyStatus.PARTIAL_EXIT)
                )
                self.state.set_total_mtm(total_mtm)
                self.state.set_strategies(self._strategies)

            except Exception as e:
                LOG.exception(f"Monitor loop error: {e}")
                self.state.add_log("ERROR", "Engine", f"Monitor error: {e}")

            time.sleep(Config.MONITOR_INTERVAL)

    # ── Public interface for Streamlit ───────────────────────

    def deploy_straddle(self, stock_code: str, expiry: str,
                        lots: int, sl_pct: float) -> Optional[Strategy]:
        return self.strategy_engine.deploy_short_straddle(
            stock_code=stock_code,
            expiry_date=expiry,
            lots=lots,
            sl_percentage=sl_pct,
        )

    def deploy_strangle(self, stock_code: str, target_delta: float,
                        expiry: str, lots: int, sl_pct: float) -> Optional[Strategy]:
        return self.strategy_engine.deploy_short_strangle(
            stock_code=stock_code,
            target_delta=target_delta,
            expiry_date=expiry,
            lots=lots,
            sl_percentage=sl_pct,
        )

    def trigger_panic_exit(self):
        strategies = self.db.get_active_strategies()
        self.risk_mgr.panic_exit(strategies)

    def get_chain_with_greeks(self, stock_code: str, expiry: str) -> list:
        return self.strategy_engine.get_option_chain_with_greeks(stock_code, expiry)
