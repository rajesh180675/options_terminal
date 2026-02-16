"""
rms.py — Risk Management System.
  - Leg-wise stop-loss monitoring
  - Global kill switch (panic_exit)
  - Circuit breakers (slippage, latency)
  - Real-time MTM computation
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

from config import CFG, INSTRUMENTS
from database import TradingDB
from shared_state import SharedState

if TYPE_CHECKING:
    from execution_engine import ExecutionEngine
    from strategy_engine import StrategyEngine


class RiskManager:
    """
    Continuously monitors all open positions and enforces risk rules.
    Runs in a dedicated background thread.
    """

    def __init__(self, db: TradingDB, state: SharedState,
                 executor: "ExecutionEngine",
                 strategy_engine: "StrategyEngine"):
        self.db = db
        self.state = state
        self.executor = executor
        self.strategy_engine = strategy_engine

        self._monitor_thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._poll_interval = 0.5  # seconds

        # Circuit breaker state
        self._consecutive_errors = 0
        self._max_consecutive_errors = 10
        self._circuit_open = False

    # ----------------------------------------------------- lifecycle
    def start(self):
        if self._monitor_thread and self._monitor_thread.is_alive():
            return
        self._stop.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="RMS-Monitor",
        )
        self._monitor_thread.start()
        self.state.add_log("INFO", "RMS", "Risk monitor started")

    def stop(self):
        self._stop.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=10)
        self.state.add_log("INFO", "RMS", "Risk monitor stopped")

    # ------------------------------------------------- main loop
    def _monitor_loop(self):
        while not self._stop.is_set():
            try:
                if self.state.is_kill_switch_active():
                    self._execute_panic_exit()
                    self.state.deactivate_kill_switch()
                    continue

                if self._circuit_open:
                    self._stop.wait(5.0)
                    self._circuit_open = False
                    self._consecutive_errors = 0
                    continue

                self._update_positions_and_mtm()
                self._check_stop_losses()
                self._check_global_loss()
                self._check_ws_health()

                self._consecutive_errors = 0

            except Exception as e:
                self._consecutive_errors += 1
                self.state.add_log("ERROR", "RMS",
                                   f"Monitor error ({self._consecutive_errors}): {e}")
                if self._consecutive_errors >= self._max_consecutive_errors:
                    self.state.add_log("CRITICAL", "RMS",
                                       "Circuit breaker tripped! "
                                       "Pausing risk monitor for 5s.")
                    self._circuit_open = True

            self._stop.wait(self._poll_interval)

    # ----------------------------------------- position & MTM update
    def _update_positions_and_mtm(self):
        """Update current prices and compute real-time MTM for all positions."""
        positions = self.db.get_open_positions()
        total_mtm = 0.0

        for pos in positions:
            stock_code = pos["stock_code"]
            strike = pos["strike_price"]
            right = pos["right_type"]
            qty = pos["quantity"]  # negative for short
            entry = pos["entry_price"]

            current = self.state.get_ltp(stock_code, strike, right)
            if current <= 0:
                current = pos.get("current_price", entry)

            # For short positions: PnL = (entry - current) * abs(qty)
            pnl = (entry - current) * abs(qty)
            total_mtm += pnl

            self.db.update_position(
                pos["id"],
                current_price=current,
                pnl=round(pnl, 2),
            )

            # Mirror to shared state
            pos_copy = dict(pos)
            pos_copy["current_price"] = current
            pos_copy["pnl"] = round(pnl, 2)
            self.state.set_position(pos["id"], pos_copy)

        self.state.set_total_mtm(round(total_mtm, 2))

        # Update strategy-level MTM
        active_strategies = self.db.get_active_strategies()
        for strat in active_strategies:
            strat_positions = self.db.get_positions_for_strategy(strat["id"])
            strat_mtm = sum(
                (p["entry_price"] - (self.state.get_ltp(
                    p["stock_code"], p["strike_price"], p["right_type"]
                ) or p["current_price"])) * abs(p["quantity"])
                for p in strat_positions if p["status"] == "open"
            )
            self.db.update_strategy(strat["id"], current_mtm=round(strat_mtm, 2))
            strat_data = self.db.get_strategy(strat["id"])
            if strat_data:
                self.state.set_strategy(strat["id"], strat_data)

    # ------------------------------------------- stop loss checking
    def _check_stop_losses(self):
        """Check each leg's SL independently and square off if hit."""
        positions = self.db.get_open_positions()

        for pos in positions:
            sl_price = pos.get("sl_price", 0)
            if sl_price <= 0:
                continue

            current = self.state.get_ltp(
                pos["stock_code"], pos["strike_price"], pos["right_type"]
            )
            if current <= 0:
                continue

            # For short positions: SL hit when current >= sl_price
            if current >= sl_price:
                self.state.add_log(
                    "WARN", "RMS",
                    f"🛑 SL HIT: {pos['strike_price']}"
                    f"{pos['right_type'][0].upper()}E "
                    f"LTP={current} >= SL={sl_price}",
                )

                qty = abs(pos["quantity"])
                result = self.executor.square_off_position(
                    strategy_id=pos["strategy_id"],
                    stock_code=pos["stock_code"],
                    strike_price=pos["strike_price"],
                    right=pos["right_type"],
                    expiry_date=pos["expiry_date"],
                    quantity=qty,
                    leg_tag="sl_exit",
                    use_market=True,
                )

                if result.success:
                    pnl = (pos["entry_price"] - result.filled_price) * qty
                    self.db.update_position(
                        pos["id"],
                        status="sl_triggered",
                        current_price=result.filled_price,
                        pnl=round(pnl, 2),
                    )
                    self.state.add_log(
                        "INFO", "RMS",
                        f"SL exit filled @{result.filled_price} "
                        f"PnL=₹{pnl:,.0f}",
                    )
                    self.db.log("WARN", "RMS",
                                f"SL triggered for position {pos['id']}",
                                {"filled": result.filled_price, "pnl": pnl})

                    # Check if all legs of strategy are closed
                    self._check_strategy_completion(pos["strategy_id"])
                else:
                    self.state.add_log(
                        "ERROR", "RMS",
                        f"SL exit FAILED: {result.error}. "
                        "Will retry next cycle.",
                    )

    def _check_strategy_completion(self, sid: str):
        """If all positions are closed, mark strategy as closed."""
        positions = self.db.get_positions_for_strategy(sid)
        open_count = sum(1 for p in positions if p["status"] == "open")
        if open_count == 0:
            self.db.update_strategy(sid, status="closed")
            strat = self.db.get_strategy(sid)
            if strat:
                self.state.set_strategy(sid, strat)
            self.state.add_log("INFO", "RMS",
                               f"Strategy {sid} fully closed (all legs exited)")

    # ----------------------------------------- global loss check
    def _check_global_loss(self):
        """If total MTM loss exceeds threshold, trigger kill switch."""
        total_mtm = self.state.get_total_mtm()
        if total_mtm < -CFG.global_max_loss:
            self.state.add_log(
                "CRITICAL", "RMS",
                f"🚨 GLOBAL LOSS LIMIT BREACHED: MTM=₹{total_mtm:,.0f} "
                f"< -₹{CFG.global_max_loss:,.0f}. ACTIVATING KILL SWITCH.",
            )
            self.state.activate_kill_switch()

    # --------------------------------------------- websocket health
    def _check_ws_health(self):
        """Log warning if WebSocket feed is stale."""
        age = self.state.get_last_tick_age()
        if age > 10 and self.state.is_engine_running():
            self.state.add_log("WARN", "RMS",
                               f"WebSocket feed stale: {age:.0f}s since "
                               "last tick")

    # ----------------------------------------------- panic exit
    def _execute_panic_exit(self):
        """
        KILL SWITCH: Square off ALL open positions at market price.
        Highest priority — bypasses normal flow.
        """
        self.state.add_log("CRITICAL", "RMS",
                           "🚨🚨🚨 PANIC EXIT INITIATED 🚨🚨🚨")

        positions = self.db.get_open_positions()
        if not positions:
            self.state.add_log("INFO", "RMS", "No open positions to close")
            return

        threads = []
        for pos in positions:
            def _exit(p=pos):
                qty = abs(p["quantity"])
                result = self.executor.square_off_position(
                    strategy_id=p["strategy_id"],
                    stock_code=p["stock_code"],
                    strike_price=p["strike_price"],
                    right=p["right_type"],
                    expiry_date=p["expiry_date"],
                    quantity=qty,
                    leg_tag="panic_exit",
                    use_market=True,
                )
                if result.success:
                    pnl = (p["entry_price"] - result.filled_price) * qty
                    self.db.update_position(
                        p["id"],
                        status="closed",
                        current_price=result.filled_price,
                        pnl=round(pnl, 2),
                    )
                    self.state.add_log(
                        "INFO", "RMS",
                        f"Panic exit filled: {p['strike_price']}"
                        f"{p['right_type'][0].upper()}E @{result.filled_price}",
                    )
                else:
                    self.state.add_log(
                        "ERROR", "RMS",
                        f"Panic exit FAILED for {p['strike_price']}"
                        f"{p['right_type'][0].upper()}E: {result.error}",
                    )

            t = threading.Thread(target=_exit, daemon=True,
                                 name=f"Panic-{pos['id']}")
            threads.append(t)
            t.start()

        for t in threads:
            t.join(timeout=30)

        # Mark all strategies as closed
        for strat in self.db.get_active_strategies():
            self.db.update_strategy(strat["id"], status="closed")
            self.state.set_strategy(strat["id"],
                                    self.db.get_strategy(strat["id"]))

        self.state.add_log("CRITICAL", "RMS",
                           "🚨 PANIC EXIT COMPLETE 🚨")

    def panic_exit(self):
        """Public method to trigger kill switch from UI."""
        self.state.activate_kill_switch()
