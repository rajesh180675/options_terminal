# ═══════════════════════════════════════════════════════════════
# FILE: trading_engine.py
# ═══════════════════════════════════════════════════════════════
"""
Orchestrator: boot → recover → monitor → shutdown.
"""

import time, threading
from typing import Optional, List
from app_config import Config
from models import Strategy, StrategyStatus, LegStatus, OptionRight
from database import Database
from shared_state import SharedState
from connection_manager import SessionManager
from order_manager import OrderManager
from strategy_engine import StrategyEngine
from rms import RiskManager
from utils import LOG, breeze_expiry, next_weekly_expiry


class TradingEngine:

    def __init__(self):
        self.state = SharedState()
        self.db = Database()
        self.session = SessionManager(self.state)
        self.omgr = OrderManager(self.session, self.db, self.state)
        self.strat = StrategyEngine(self.session, self.omgr, self.db, self.state)
        self.risk = RiskManager(self.session, self.omgr, self.db, self.state)
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._strategies: List[Strategy] = []

    def start(self) -> bool:
        self.state.add_log("INFO", "Engine", f"Mode: {Config.TRADING_MODE.upper()}")
        errs = Config.validate()
        if errs and Config.is_live():
            for e in errs:
                self.state.add_log("ERROR", "Engine", f"Config: {e}")
            return False
        if not self.session.initialize():
            return False
        self.session.connect_ws()
        time.sleep(1.0)
        self._recover()
        self._running = True
        self.state.engine_running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="Monitor")
        self._thread.start()
        self.state.add_log("INFO", "Engine", "Engine running ✓")
        return True

    def stop(self):
        self._running = False
        self.state.engine_running = False
        self.session.shutdown()
        self.db.close()

    def _recover(self):
        strats = self.db.get_active_strategies()
        if not strats:
            self.state.add_log("INFO", "Engine", "No positions to recover")
            return
        self.state.add_log("INFO", "Engine", f"Recovering {len(strats)} strategies")
        for s in strats:
            for leg in s.legs:
                if leg.status in (LegStatus.ACTIVE, LegStatus.ENTERING):
                    try:
                        self.session.subscribe_option(
                            leg.stock_code, leg.strike_price,
                            leg.right.value, leg.expiry_date, leg.exchange_code)
                    except Exception as e:
                        LOG.error(f"Re-sub failed: {e}")
        self._strategies = strats
        self.state.set_strategies(strats)

    def _loop(self):
        while self._running:
            try:
                self._strategies = self.db.get_active_strategies()
                # Update spots for all active instruments
                seen_codes = set()
                for s in self._strategies:
                    if s.stock_code not in seen_codes:
                        seen_codes.add(s.stock_code)
                        spot = self.session.get_spot_price(s.stock_code)
                        if spot > 0:
                            self.state.set_spot(s.stock_code, spot)
                # Also update default stock
                dc = Config.breeze_code(Config.DEFAULT_STOCK)
                if dc not in seen_codes:
                    sp = self.session.get_spot_price(dc)
                    if sp > 0:
                        self.state.set_spot(dc, sp)

                # Update prices
                for s in self._strategies:
                    spot = self.state.get_spot(s.stock_code)
                    for leg in s.legs:
                        if leg.status != LegStatus.ACTIVE: continue
                        t = self.state.get_tick(leg.feed_key)
                        if t and t.ltp > 0:
                            leg.current_price = t.ltp
                            leg.compute_pnl()

                    # Greeks
                    if spot > 0:
                        self.risk.update_greeks([s], spot)

                # Risk
                if not self.state.panic_triggered:
                    self.risk.check_stop_losses(self._strategies)
                if not self.state.panic_triggered:
                    self.risk.check_global_mtm(self._strategies)
                if not self.state.panic_triggered:
                    self.risk.check_auto_exit(self._strategies)

                # MTM
                total = sum(s.compute_total_pnl() for s in self._strategies
                            if s.status in (StrategyStatus.ACTIVE, StrategyStatus.PARTIAL_EXIT))
                self.state.set_total_mtm(total)
                self.state.set_strategies(self._strategies)
            except Exception as e:
                LOG.exception(f"Monitor: {e}")
                self.state.add_log("ERROR", "Engine", f"Monitor: {e}")
            time.sleep(Config.MONITOR_INTERVAL)

    # ── Public API ───────────────────────────────────────────

    def deploy_straddle(self, instrument, expiry, lots, sl):
        return self.strat.deploy_straddle(instrument, expiry, lots, sl)

    def deploy_strangle(self, instrument, delta, expiry, lots, sl):
        return self.strat.deploy_strangle(instrument, delta, expiry, lots, sl)

    def deploy_limit_sell(self, instrument, strike, right, expiry, lots, price, sl):
        return self.strat.sell_single(instrument, strike, right, expiry, lots, price, sl)

    def trigger_panic(self):
        self.risk.panic_exit(self.db.get_active_strategies())

    def get_chain(self, instrument, expiry):
        return self.strat.chain_with_greeks(instrument, expiry)
