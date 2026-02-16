"""
trading_engine.py — Main orchestrator that wires all components together.
Manages background threads, crash recovery, and the overall lifecycle.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta, timezone

from config import CFG, INSTRUMENTS
from connection_manager import ConnectionManager
from database import TradingDB
from execution_engine import ExecutionEngine
from greeks_engine import GreeksEngine
from rms import RiskManager
from shared_state import SharedState, GreeksData
from strategy_engine import StrategyEngine, StrategyParams


class TradingEngine:
    """
    Top-level engine.  Initialised once in Streamlit's session_state.
    """

    def __init__(self):
        self.state = SharedState()
        self.db = TradingDB(CFG.db_path)
        self.conn = ConnectionManager(self.state, mode=CFG.trading_mode)
        self.executor = ExecutionEngine(self.conn, self.db, self.state)
        self.strategy_engine = StrategyEngine(
            self.conn, self.db, self.state, self.executor
        )
        self.rms = RiskManager(
            self.db, self.state, self.executor, self.strategy_engine
        )
        self.greeks_engine = GreeksEngine(CFG.risk_free_rate)

        self._greeks_thread: threading.Thread | None = None
        self._greeks_stop = threading.Event()
        self._started = False

    # --------------------------------------------------------- lifecycle
    def start(self) -> bool:
        """Authenticate, connect WS, recover state, start monitors."""
        if self._started:
            return True

        self.state.add_log("INFO", "Engine", "=== Trading Engine Starting ===")
        self.state.add_log("INFO", "Engine", f"Mode: {CFG.trading_mode}")

        # 1. Authenticate
        if not self.conn.authenticate():
            self.state.add_log("ERROR", "Engine", "Authentication failed")
            return False

        # 2. Start WebSocket
        self.conn.start_websocket()
        time.sleep(1)

        # 3. Recover state from DB
        self._recover_state()

        # 4. Start Greeks updater
        self._greeks_stop.clear()
        self._greeks_thread = threading.Thread(
            target=self._greeks_update_loop,
            daemon=True,
            name="Greeks-Updater",
        )
        self._greeks_thread.start()

        # 5. Start RMS
        self.rms.start()

        self._started = True
        self.state.set_engine_running(True)
        self.state.add_log("INFO", "Engine",
                           "=== Trading Engine Running ===")
        return True

    def stop(self):
        """Graceful shutdown."""
        self.state.add_log("INFO", "Engine", "=== Shutting Down ===")
        self.state.set_engine_running(False)
        self.rms.stop()
        self._greeks_stop.set()
        self.conn.stop_websocket()
        self._started = False
        self.state.add_log("INFO", "Engine", "=== Shutdown Complete ===")

    # -------------------------------------------------- crash recovery
    def _recover_state(self):
        """
        On restart, reload active strategies and positions from DB.
        Re-subscribe to WebSocket feeds for monitored strikes.
        """
        self.state.add_log("INFO", "Engine", "Recovering state from DB...")

        active = self.db.get_active_strategies()
        if not active:
            self.state.add_log("INFO", "Engine", "No active strategies to recover")
            return

        for strat in active:
            sid = strat["id"]
            self.state.set_strategy(sid, strat)
            self.state.add_log(
                "INFO", "Engine",
                f"Recovered strategy {sid}: {strat['name']} "
                f"CE={strat['ce_strike']} PE={strat['pe_strike']} "
                f"Status={strat['status']}",
            )

            # Re-subscribe to feeds
            stock_code = strat["stock_code"]
            expiry = strat["expiry_date"]
            if strat["ce_strike"]:
                self.conn.subscribe(stock_code, expiry,
                                    strat["ce_strike"], "call")
            if strat["pe_strike"]:
                self.conn.subscribe(stock_code, expiry,
                                    strat["pe_strike"], "put")

        positions = self.db.get_open_positions()
        for pos in positions:
            self.state.set_position(pos["id"], pos)

        self.state.add_log(
            "INFO", "Engine",
            f"Recovered {len(active)} strategies, "
            f"{len(positions)} open positions",
        )

    # ------------------------------------------------ greeks updater
    def _greeks_update_loop(self):
        """Periodically re-compute Greeks for all open positions."""
        while not self._greeks_stop.is_set():
            try:
                positions = self.db.get_open_positions()
                for pos in positions:
                    stock_code = pos["stock_code"]
                    strike = pos["strike_price"]
                    right = pos["right_type"]
                    expiry = pos["expiry_date"]

                    spot = self.state.get_spot(stock_code)
                    if spot <= 0:
                        continue

                    ltp = self.state.get_ltp(stock_code, strike, right)
                    if ltp <= 0:
                        continue

                    tte = GreeksEngine.time_to_expiry_years(expiry)
                    greeks = self.greeks_engine.compute_for_strike(
                        spot, strike, tte, ltp, right
                    )

                    gd = GreeksData(
                        iv=greeks.iv,
                        delta=greeks.delta,
                        gamma=greeks.gamma,
                        theta=greeks.theta,
                        vega=greeks.vega,
                    )
                    self.state.update_greeks(stock_code, strike, right, gd)

            except Exception as e:
                self.state.add_log("ERROR", "Greeks",
                                   f"Greeks update error: {e}")

            self._greeks_stop.wait(2.0)

    # ------------------------------------------- public API for UI
    def deploy_strategy(self, params: StrategyParams) -> str | None:
        """Called from UI thread. Runs execution in background."""
        result = [None]

        def _run():
            result[0] = self.strategy_engine.deploy_strategy(params)

        t = threading.Thread(target=_run, daemon=True, name="Deploy")
        t.start()
        t.join(timeout=120)
        return result[0]

    def exit_strategy(self, sid: str, use_market: bool = False):
        """Called from UI thread."""
        def _run():
            self.strategy_engine.exit_strategy(sid, use_market)

        t = threading.Thread(target=_run, daemon=True, name="Exit")
        t.start()

    def panic_exit(self):
        """Trigger global kill switch."""
        self.rms.panic_exit()

    def get_expiry_dates(self, instrument: str = "NIFTY") -> list[str]:
        """Generate upcoming weekly expiry dates (Thursday IST)."""
        today = datetime.now(timezone.utc).date()
        expiries = []
        for i in range(8):
            d = today + timedelta(days=i)
            # Find next Thursday (weekday 3)
            days_ahead = (3 - d.weekday()) % 7
            if days_ahead == 0 and d == today:
                expiry = d
            else:
                expiry = d + timedelta(days=days_ahead)
                if expiry in [e.date() if hasattr(e, 'date') else e
                              for e in expiries]:
                    continue
            expiry_iso = (
                datetime(expiry.year, expiry.month, expiry.day,
                         15, 30, 0, tzinfo=timezone.utc)
                .strftime("%Y-%m-%dT%H:%M:%S.000Z")
            )
            if expiry_iso not in expiries:
                expiries.append(expiry_iso)

        return sorted(set(expiries))[:5]

    def refresh_option_chain(self, instrument: str,
                             expiry_date: str) -> list[dict]:
        """Fetch and cache option chain for display."""
        spec = CFG.get_instrument(instrument)
        chain = self.conn.get_option_chain(spec.stock_code, expiry_date)

        # Update spot
        spot = 0.0
        for rec in chain:
            sp = rec.get("spot_price", "")
            if sp:
                try:
                    spot = float(sp)
                    break
                except ValueError:
                    pass

        if spot > 0:
            self.state.set_spot(spec.stock_code, spot)
        else:
            spot = self.conn.get_spot_quote(spec.stock_code)
            if spot > 0:
                self.state.set_spot(spec.stock_code, spot)

        return chain

    @property
    def is_running(self) -> bool:
        return self._started
