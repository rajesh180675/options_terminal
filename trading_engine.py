"""
trading_engine.py

Professional orchestrator:
  - session connect + websocket
  - survivable DB recovery
  - monitor loop (ticks -> leg LTP -> pnl)
  - greeks + RMS (SL, trailing, global MTM, auto exit)
  - broker sync + reconciliation (if broker_sync.py exists)
  - pending order fill detection (if pending_monitor.py exists)
  - adjustment suggestions / journal / backtest (optional modules)

This file only depends on other modules; it does not contain UI code.
"""

from __future__ import annotations

import time
import threading
from typing import Optional, List, Dict

from app_config import Config
from connection_manager import SessionManager
from database import Database
from order_manager import OrderManager
from shared_state import SharedState
from strategy_engine import StrategyEngine
from rms import RiskManager
from models import Strategy, StrategyStatus, LegStatus
from utils import LOG


class TradingEngine:
    def __init__(self):
        self.state = SharedState()
        self.db = Database()
        self.session = SessionManager(self.state)
        self.omgr = OrderManager(self.session, self.db, self.state)
        self.strategy = StrategyEngine(self.session, self.omgr, self.db, self.state)
        self.rms = RiskManager(self.omgr, self.db, self.state)

        # Optional subsystems
        self.broker = None
        self.pending_mon = None
        self.adjuster = None
        self.journal = None
        self.backtester = None
        self.alerts = None

        self._running = False
        self._thread: Optional[threading.Thread] = None

        # timers
        self._last_broker_sync = 0.0
        self._last_adjust = 0.0
        self._last_journal = 0.0

        self._init_optional_modules()

    def _init_optional_modules(self):
        # Alerts
        try:
            from alerts import AlertManager
            self.alerts = AlertManager()
            self.rms.alerts = self.alerts
        except Exception:
            self.alerts = None

        # Broker sync
        try:
            from broker_sync import BrokerSync
            self.broker = BrokerSync(self.session, self.db, self.state)
        except Exception:
            self.broker = None

        # Pending monitor
        try:
            from pending_monitor import PendingOrderMonitor
            self.pending_mon = PendingOrderMonitor(self.session, self.db, self.state)
        except Exception:
            self.pending_mon = None

        # Adjustment engine
        try:
            from adjustment_engine import AdjustmentEngine
            self.adjuster = AdjustmentEngine()
        except Exception:
            self.adjuster = None

        # Journal
        try:
            from journal import TradeJournal
            self.journal = TradeJournal()
        except Exception:
            self.journal = None

        # Backtester
        try:
            from backtester import Backtester
            self.backtester = Backtester(self.session)
        except Exception:
            self.backtester = None

    # ── Lifecycle ──────────────────────────────────────────────

    def start(self) -> bool:
        self.state.add_log("INFO", "Engine", f"Starting. Mode={Config.TRADING_MODE.upper()}")

        errs = Config.validate()
        if errs and Config.is_live():
            for e in errs:
                self.state.add_log("ERROR", "Engine", f"Config: {e}")
            return False

        if not self.session.initialize():
            return False

        # connect websocket
        try:
            if hasattr(self.session, "connect_ws"):
                self.session.connect_ws()
            else:
                self.session.connect_websocket()  # older naming
        except Exception as e:
            self.state.add_log("ERROR", "WS", f"WS connect failed: {e}")

        time.sleep(1.0)

        # Broker initial sync
        if self.broker:
            try:
                self.broker.fetch_funds()
                self.broker.fetch_customer_details()
                self.broker.fetch_positions()
                self.broker.fetch_orders()
                self.broker.fetch_trades()
                self.broker.reconcile()
            except Exception as e:
                self.state.add_log("WARN", "Broker", f"Initial broker sync failed: {e}")

        # Recover positions from DB
        self._recover()

        # Start pending order monitor
        if self.pending_mon:
            try:
                self.pending_mon.start()
            except Exception as e:
                self.state.add_log("WARN", "PendMon", f"Start failed: {e}")

        # Start monitor thread
        self._running = True
        self.state.engine_running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="EngineLoop")
        self._thread.start()

        self.state.add_log("INFO", "Engine", "Engine running")
        LOG.info("Engine started")
        return True

    def stop(self):
        self._running = False
        self.state.engine_running = False

        if self.pending_mon:
            try:
                self.pending_mon.stop()
            except Exception:
                pass

        try:
            self.session.shutdown()
        except Exception:
            pass

        try:
            self.db.close()
        except Exception:
            pass

        self.state.add_log("INFO", "Engine", "Engine stopped")

    # ── Recovery ───────────────────────────────────────────────

    def _recover(self):
        strategies = self.db.get_active_strategies()
        if not strategies:
            self.state.add_log("INFO", "Engine", "No strategies to recover")
            return

        self.state.add_log("INFO", "Engine", f"Recovering {len(strategies)} strategies")

        for s in strategies:
            for leg in s.legs:
                if leg.status in (LegStatus.ACTIVE, LegStatus.ENTERING):
                    try:
                        self.session.subscribe_option(
                            stock_code=leg.stock_code,
                            strike=leg.strike_price,
                            right=leg.right.value,
                            expiry=leg.expiry_date,
                            exchange_code=leg.exchange_code,
                        )
                    except Exception as e:
                        self.state.add_log("WARN", "WS", f"Recover subscribe failed: {leg.display_name}: {e}")

        self.state.set_strategies(strategies)

    # ── Monitor loop ───────────────────────────────────────────

    def _loop(self):
        while self._running:
            try:
                strategies = self.db.get_active_strategies()

                # Update spots for each stock_code we see
                spot_map: Dict[str, float] = {}
                for s in strategies:
                    if s.stock_code not in spot_map:
                        sp = float(self.session.get_spot_price(s.stock_code))
                        if sp > 0:
                            spot_map[s.stock_code] = sp
                            self.state.set_spot(s.stock_code, sp)

                # also default stock (for UI header)
                default_code = Config.breeze_code(Config.DEFAULT_STOCK)
                if default_code not in spot_map:
                    sp = float(self.session.get_spot_price(default_code))
                    if sp > 0:
                        spot_map[default_code] = sp
                        self.state.set_spot(default_code, sp)

                # Update leg prices from ticks
                for s in strategies:
                    for leg in s.legs:
                        if leg.status != LegStatus.ACTIVE:
                            continue
                        tick = self.state.get_tick(leg.feed_key)
                        if tick and tick.ltp > 0:
                            leg.current_price = float(tick.ltp)
                            leg.compute_pnl()
                            # Persist lightweight (price/pnl/sl trail updates happen in RMS)
                            self.db.update_leg_price(leg.leg_id, leg.current_price, leg.pnl)

                # Greeks + portfolio aggregation
                self.rms.update_greeks_and_portfolio(strategies, spot_map)

                # RMS checks
                if not self.state.panic_triggered:
                    self.rms.check_stop_losses(strategies)
                if not self.state.panic_triggered:
                    self.rms.check_global_mtm(strategies)
                if not self.state.panic_triggered:
                    self.rms.check_auto_exit(strategies)

                # Total MTM
                total = 0.0
                for s in strategies:
                    if s.status in (StrategyStatus.ACTIVE, StrategyStatus.PARTIAL_EXIT):
                        total += s.compute_total_pnl()
                self.state.set_total_mtm(total)

                # Strategy snapshot to UI
                self.state.set_strategies(strategies)

                # Optional: broker sync periodic
                now = time.time()
                if self.broker and (now - self._last_broker_sync) >= float(getattr(Config, "BROKER_SYNC_INTERVAL", 30)):
                    self._last_broker_sync = now
                    try:
                        self.broker.full_sync()
                    except Exception as e:
                        self.state.add_log("WARN", "Broker", f"Sync failed: {e}")

                # Optional: adjustment suggestions periodic
                if self.adjuster and (now - self._last_adjust) >= 30.0:
                    self._last_adjust = now
                    try:
                        spots = self.state.get_all_spots()
                        adj = self.adjuster.analyze(strategies, spots)
                        self.state.set_adjustments(adj)
                        # Escalate critical suggestions
                        if self.alerts:
                            for a in adj:
                                if getattr(a, "severity", "").upper() == "CRITICAL":
                                    self.alerts.adjustment_suggestion(getattr(a, "message", "Critical adjustment"))
                    except Exception as e:
                        self.state.add_log("WARN", "Adjust", f"Adjustment engine failed: {e}")

                # Optional: journal snapshot periodic
                if self.journal and (now - self._last_journal) >= 60.0:
                    self._last_journal = now
                    try:
                        # We only snapshot daily pnl here; strategy journaling should be triggered
                        # when a strategy closes (not implemented globally without event bus).
                        realized = 0.0
                        unrealized = total
                        opened = sum(1 for s in strategies if s.status == StrategyStatus.DEPLOYING)
                        closed = sum(1 for s in strategies if s.status == StrategyStatus.CLOSED)
                        trades = len(strategies)
                        self.journal.update_daily_snapshot(realized, unrealized, opened, closed, trades)
                    except Exception:
                        pass

            except Exception as e:
                LOG.exception(f"Engine loop error: {e}")
                self.state.add_log("ERROR", "Engine", f"Loop error: {e}")

            time.sleep(float(Config.MONITOR_INTERVAL))

    # ── Strategy API for UI ────────────────────────────────────

    def deploy_straddle(self, instrument: str, expiry: str, lots: int, sl_pct: float):
        # optional margin preflight
        if self.broker:
            try:
                # crude preflight: just ensure funds known; detailed check done in UI for limit sells
                self.broker.fetch_funds()
            except Exception:
                pass
        return self.strategy.deploy_short_straddle(instrument, expiry, lots, sl_pct)

    def deploy_strangle(self, instrument: str, target_delta: float, expiry: str, lots: int, sl_pct: float):
        if self.broker:
            try:
                self.broker.fetch_funds()
            except Exception:
                pass
        return self.strategy.deploy_short_strangle(instrument, target_delta, expiry, lots, sl_pct)

    def deploy_limit_sell(self, instrument: str, strike: float, right, expiry: str, lots: int, price: float, sl_pct: float):
        return self.strategy.sell_single_leg_limit(instrument, strike, right, expiry, lots, price, sl_pct)

    def deploy_iron_condor(self, instrument: str, target_delta: float, wing_width: int, expiry: str, lots: int, sl_pct: float):
        return self.strategy.deploy_iron_condor(instrument, target_delta, wing_width, expiry, lots, sl_pct)

    def deploy_iron_butterfly(self, instrument: str, wing_width: int, expiry: str, lots: int, sl_pct: float):
        return self.strategy.deploy_iron_butterfly(instrument, wing_width, expiry, lots, sl_pct)

    def trigger_panic(self):
        self.rms.panic_exit(self.db.get_active_strategies(), reason="manual_panic")

    def exit_leg(self, leg_id: str) -> bool:
        """Manual exit a specific active leg by leg_id."""
        strategies = self.db.get_active_strategies()
        for s in strategies:
            for leg in s.legs:
                if leg.leg_id == leg_id and leg.status == LegStatus.ACTIVE:
                    self.state.add_log("INFO", "Engine", f"Manual exit: {leg.display_name}")
                    ok = self.omgr.buy_market(leg) if leg.side == LegStatus.ACTIVE else self.omgr.buy_market(leg)
                    if ok:
                        # Update strategy status
                        remaining = [l for l in s.legs if l.status == LegStatus.ACTIVE and l.leg_id != leg_id]
                        s.status = StrategyStatus.CLOSED if not remaining else StrategyStatus.PARTIAL_EXIT
                        s.compute_total_pnl()
                        self.db.save_strategy(s)
                    return bool(ok)
        return False

    def get_chain(self, instrument: str, expiry: str):
        return self.strategy.get_chain_with_greeks(instrument, expiry)

    def run_backtest(self, **params):
        if not self.backtester:
            raise RuntimeError("Backtester module not available")
        return self.backtester.run(**params)
