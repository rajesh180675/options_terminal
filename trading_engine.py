# ═══════════════════════════════════════════════════════════════
# FILE: trading_engine.py
# ═══════════════════════════════════════════════════════════════
"""
Central orchestrator: boot → recover → monitor → shutdown.
"""

import time
import threading
from typing import Optional, List

from config import Config
from models import Strategy, StrategyStatus, LegStatus, OptionRight
from database import Database
from shared_state import SharedState
from connection_manager import SessionManager
from order_manager import OrderManager
from strategy_engine import StrategyEngine
from rms import RiskManager
from utils import LOG, breeze_expiry_format, next_weekly_expiry


class TradingEngine:

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

    def start(self) -> bool:
        self.state.add_log("INFO", "Engine", "Starting…")

        # Validate config
        errors = Config.validate()
        if errors and Config.is_live():
            for e in errors:
                self.state.add_log("ERROR", "Engine", f"Config error: {e}")
            return False

        self.state.add_log("INFO", "Engine",
                           f"Mode: {Config.TRADING_MODE.upper()}")

        if not self.session.initialize():
            return False

        self.session.connect_websocket()
        time.sleep(1.0)

        self._recover_positions()

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
        self._running = False
        self.state.engine_running = False
        self.session.shutdown()
        self.db.close()
        self.state.add_log("INFO", "Engine", "Engine stopped")

    def _recover_positions(self):
        strategies = self.db.get_active_strategies()
        if not strategies:
            self.state.add_log("INFO", "Engine", "No positions to recover")
            return

        self.state.add_log("INFO", "Engine",
                           f"Recovering {len(strategies)} strategies")

        for strategy in strategies:
            for leg in strategy.legs:
                if leg.status in (LegStatus.ACTIVE, LegStatus.ENTERING):
                    try:
                        self.session.subscribe_option(
                            leg.stock_code, leg.strike_price,
                            leg.right.value, leg.expiry_date
                        )
                    except Exception as e:
                        LOG.error(f"Re-subscribe failed: {leg.leg_id}: {e}")

        self._strategies = strategies
        self.state.set_strategies(strategies)

    def _monitor_loop(self):
        while self._running:
            try:
                self._strategies = self.db.get_active_strategies()
                stock_code = Config.breeze_code(Config.DEFAULT_STOCK)
                spot = self.session.get_spot_price(stock_code)
                if spot > 0:
                    self.state.set_spot(stock_code, spot)

                # Update leg prices
                for strategy in self._strategies:
                    for leg in strategy.legs:
                        if leg.status != LegStatus.ACTIVE:
                            continue
                        tick = self.state.get_tick(leg.feed_key)
                        if tick and tick.ltp > 0:
                            leg.current_price = tick.ltp
                            leg.compute_pnl()
                            self.db.update_leg_price(
                                leg.leg_id, leg.current_price, leg.pnl
                            )

                # Greeks
                if spot > 0:
                    self.risk_mgr.update_greeks(self._strategies, spot)

                # Risk checks
                if not self.state.panic_triggered:
                    self.risk_mgr.check_stop_losses(self._strategies)
                if not self.state.panic_triggered:
                    self.risk_mgr.check_global_mtm(self._strategies)

                # Total MTM
                total = sum(
                    s.compute_total_pnl() for s in self._strategies
                    if s.status in (StrategyStatus.ACTIVE, StrategyStatus.PARTIAL_EXIT)
                )
                self.state.set_total_mtm(total)
                self.state.set_strategies(self._strategies)

            except Exception as e:
                LOG.exception(f"Monitor error: {e}")
                self.state.add_log("ERROR", "Engine", f"Monitor error: {e}")

            time.sleep(Config.MONITOR_INTERVAL)

    # ── Public API for Streamlit ─────────────────────────────

    def deploy_straddle(self, stock_code, expiry, lots, sl_pct):
        return self.strategy_engine.deploy_short_straddle(
            stock_code=stock_code, expiry_date=expiry,
            lots=lots, sl_percentage=sl_pct,
        )

    def deploy_strangle(self, stock_code, target_delta, expiry, lots, sl_pct):
        return self.strategy_engine.deploy_short_strangle(
            stock_code=stock_code, target_delta=target_delta,
            expiry_date=expiry, lots=lots, sl_percentage=sl_pct,
        )

    def deploy_limit_sell(self, stock_code, strike, right, expiry,
                          lots, limit_price, sl_pct):
        return self.strategy_engine.sell_single_leg_limit(
            stock_code=stock_code, strike_price=strike,
            right=right, expiry_date=expiry, lots=lots,
            limit_price=limit_price, sl_percentage=sl_pct,
        )

    def trigger_panic_exit(self):
        strategies = self.db.get_active_strategies()
        self.risk_mgr.panic_exit(strategies)

    def get_chain_with_greeks(self, stock_code, expiry):
        return self.strategy_engine.get_option_chain_with_greeks(stock_code, expiry)
