# ═══════════════════════════════════════════════════════════════
# FILE: trading_engine.py  (UPDATED — integrates broker_sync + pending_monitor)
# ═══════════════════════════════════════════════════════════════
"""
Orchestrator now runs:
  1. Monitor loop (prices, Greeks, SL)
  2. Broker sync (funds, positions, orders, trades, reconciliation)
  3. Pending order monitor (detects fills on limit orders)
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
from broker_sync import BrokerSync
from pending_monitor import PendingOrderMonitor
from utils import LOG, breeze_expiry, next_weekly_expiry


class TradingEngine:

    def __init__(self):
        self.state = SharedState()
        self.db = Database()
        self.session = SessionManager(self.state)
        self.omgr = OrderManager(self.session, self.db, self.state)
        self.strat = StrategyEngine(self.session, self.omgr, self.db, self.state)
        self.risk = RiskManager(self.session, self.omgr, self.db, self.state)
        self.broker = BrokerSync(self.session, self.db, self.state)
        self.pending_mon = PendingOrderMonitor(self.session, self.db, self.state)
        self._thread = None
        self._running = False

    def start(self) -> bool:
        self.state.add_log("INFO", "Engine", f"Mode: {Config.TRADING_MODE.upper()}")
        errs = Config.validate()
        if errs and Config.is_live():
            for e in errs: self.state.add_log("ERROR", "Engine", e)
            return False

        if not self.session.initialize():
            return False

        self.session.connect_ws()
        time.sleep(1.0)

        # Fetch initial broker data
        self.broker.fetch_funds()
        self.broker.fetch_customer_details()
        self.state.add_log("INFO", "Engine", "Broker data fetched")

        self._recover()

        # Start pending order monitor
        self.pending_mon.start()

        self._running = True
        self.state.engine_running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="Monitor")
        self._thread.start()
        self.state.add_log("INFO", "Engine", "Engine running ✓")
        return True

    def stop(self):
        self._running = False
        self.state.engine_running = False
        self.pending_mon.stop()
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
                        LOG.error(f"Re-sub: {e}")
        self.state.set_strategies(strats)

    def _loop(self):
        sync_counter = 0
        while self._running:
            try:
                strategies = self.db.get_active_strategies()
                seen = set()
                for s in strategies:
                    if s.stock_code not in seen:
                        seen.add(s.stock_code)
                        sp = self.session.get_spot_price(s.stock_code)
                        if sp > 0: self.state.set_spot(s.stock_code, sp)
                dc = Config.breeze_code(Config.DEFAULT_STOCK)
                if dc not in seen:
                    sp = self.session.get_spot_price(dc)
                    if sp > 0: self.state.set_spot(dc, sp)

                for s in strategies:
                    spot = self.state.get_spot(s.stock_code)
                    for leg in s.legs:
                        if leg.status != LegStatus.ACTIVE: continue
                        t = self.state.get_tick(leg.feed_key)
                        if t and t.ltp > 0:
                            leg.current_price = t.ltp
                            leg.compute_pnl()
                    if spot > 0:
                        self.risk.update_greeks([s], spot)

                if not self.state.panic_triggered:
                    self.risk.check_stop_losses(strategies)
                if not self.state.panic_triggered:
                    self.risk.check_global_mtm(strategies)
                if not self.state.panic_triggered:
                    self.risk.check_auto_exit(strategies)

                total = sum(s.compute_total_pnl() for s in strategies
                            if s.status in (StrategyStatus.ACTIVE, StrategyStatus.PARTIAL_EXIT))
                self.state.set_total_mtm(total)
                self.state.set_strategies(strategies)

                # Periodic broker sync (every ~30 seconds)
                sync_counter += 1
                if sync_counter * Config.MONITOR_INTERVAL >= Config.BROKER_SYNC_INTERVAL:
                    sync_counter = 0
                    self.broker.full_sync()

            except Exception as e:
                LOG.exception(f"Monitor: {e}")
                self.state.add_log("ERROR", "Engine", f"Monitor: {e}")
            time.sleep(Config.MONITOR_INTERVAL)

    # ── Public API ───────────────────────────────────────────

    def deploy_straddle(self, inst, expiry, lots, sl):
        # Pre-trade margin check
        ok, req, msg = self._margin_preflight(inst, lots)
        if not ok:
            self.state.add_log("ERROR", "Engine", f"Margin: {msg}")
            return None
        self.state.add_log("INFO", "Engine", f"Margin: {msg}")
        return self.strat.deploy_straddle(inst, expiry, lots, sl)

    def deploy_strangle(self, inst, delta, expiry, lots, sl):
        ok, req, msg = self._margin_preflight(inst, lots)
        if not ok:
            self.state.add_log("ERROR", "Engine", f"Margin: {msg}")
            return None
        return self.strat.deploy_strangle(inst, delta, expiry, lots, sl)

    def deploy_limit_sell(self, inst, strike, right, expiry, lots, price, sl):
        inst_cfg = Config.instrument(inst)
        qty = inst_cfg["lot_size"] * lots
        ok, req, msg = self.broker.check_margin(
            inst_cfg["breeze_code"], inst_cfg["exchange"],
            strike, right.value, expiry, qty, price, "sell"
        )
        if not ok:
            self.state.add_log("ERROR", "Engine", f"Margin: {msg}")
            return None
        self.state.add_log("INFO", "Engine", f"Margin: {msg}")
        return self.strat.sell_single(inst, strike, right, expiry, lots, price, sl)

    def exit_leg(self, leg_id: str) -> bool:
        """Exit a specific leg from UI."""
        strategies = self.db.get_active_strategies()
        for s in strategies:
            for leg in s.legs:
                if leg.leg_id == leg_id and leg.status == LegStatus.ACTIVE:
                    self.state.add_log("INFO", "Engine", f"Manual exit: {leg.display_name}")
                    ok = self.omgr.buy_market(leg)
                    if ok:
                        active = [l for l in s.legs if l.status == LegStatus.ACTIVE and l.leg_id != leg_id]
                        s.status = StrategyStatus.CLOSED if not active else StrategyStatus.PARTIAL_EXIT
                        s.compute_total_pnl()
                        self.db.save_strategy(s)
                    return ok
        return False

    def trigger_panic(self):
        self.risk.panic_exit(self.db.get_active_strategies())

    def get_chain(self, inst, expiry):
        return self.strat.chain_with_greeks(inst, expiry)

    def _margin_preflight(self, inst, lots):
        inst_cfg = Config.instrument(inst)
        spot = self.session.get_spot_price(inst_cfg["breeze_code"])
        if spot <= 0: spot = 24000
        from utils import atm_strike
        atm = atm_strike(spot, inst_cfg["strike_gap"])
        qty = inst_cfg["lot_size"] * lots
        return self.broker.check_margin(
            inst_cfg["breeze_code"], inst_cfg["exchange"],
            atm, "call", breeze_expiry(next_weekly_expiry(inst)),
            qty, spot * 0.01, "sell"
        )
