"""
strategy_engine.py — Autonomous option-selling strategy engine.
Supports Short Straddle and Short Strangle with delta-based strike selection.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Literal

from config import CFG, INSTRUMENTS
from connection_manager import ConnectionManager
from database import TradingDB
from execution_engine import ExecutionEngine, OrderResult
from greeks_engine import GreeksEngine
from shared_state import SharedState, GreeksData


@dataclass
class StrategyParams:
    instrument: str = "NIFTY"       # NIFTY, BANKNIFTY
    strategy_type: str = "strangle" # "straddle" or "strangle"
    target_delta: float = 0.15      # for strangle
    lots: int = 1
    sl_multiplier: float = 2.0     # SL = entry_price * sl_multiplier
    expiry_date: str = ""           # ISO format


class StrategyEngine:
    """
    Deploys and manages option selling strategies.
    """

    def __init__(self, conn: ConnectionManager, db: TradingDB,
                 state: SharedState, executor: ExecutionEngine):
        self.conn = conn
        self.db = db
        self.state = state
        self.executor = executor
        self.greeks = GreeksEngine(CFG.risk_free_rate)
        self._lock = threading.Lock()

    # ------------------------------------------------------- deploy
    def deploy_strategy(self, params: StrategyParams) -> str | None:
        """
        Main entry point: analyse the option chain, select strikes,
        and execute the strategy.
        Returns strategy_id on success, None on failure.
        """
        instrument = CFG.get_instrument(params.instrument)
        stock_code = instrument.stock_code
        lot_size = instrument.lot_size
        quantity = lot_size * params.lots

        self.state.add_log("INFO", "Strategy",
                           f"Deploying {params.strategy_type} on "
                           f"{params.instrument} ({params.lots} lots)")

        # 1. Get spot price
        spot = self.state.get_spot(stock_code)
        if spot <= 0:
            spot = self.conn.get_spot_quote(stock_code)
            if spot <= 0:
                self.state.add_log("ERROR", "Strategy",
                                   "Cannot determine spot price")
                return None
            self.state.set_spot(stock_code, spot)

        self.state.add_log("INFO", "Strategy", f"Spot price: {spot}")

        # 2. Fetch option chain
        chain = self.conn.get_option_chain(stock_code, params.expiry_date)
        if not chain:
            self.state.add_log("ERROR", "Strategy",
                               "Empty option chain received")
            return None

        # 3. Parse chain into strike -> price mapping
        ce_prices: dict[float, float] = {}
        pe_prices: dict[float, float] = {}
        strikes: set[float] = set()

        for rec in chain:
            try:
                strike = float(rec.get("strike_price", 0))
                ltp = float(rec.get("ltp", 0))
                right = rec.get("right", "").lower()
                strikes.add(strike)
                if "call" in right:
                    ce_prices[strike] = ltp
                elif "put" in right:
                    pe_prices[strike] = ltp
            except (ValueError, TypeError):
                continue

        sorted_strikes = sorted(strikes)
        if not sorted_strikes:
            self.state.add_log("ERROR", "Strategy", "No valid strikes found")
            return None

        tte = self.greeks.time_to_expiry_years(params.expiry_date)
        self.state.add_log("INFO", "Strategy",
                           f"TTE: {tte * 365.25:.2f} days, "
                           f"Strikes: {len(sorted_strikes)}")

        # 4. Select strikes based on strategy type
        if params.strategy_type == "straddle":
            ce_strike, pe_strike = self._select_straddle_strikes(
                spot, sorted_strikes, instrument.strike_gap
            )
        else:
            ce_strike, pe_strike = self._select_strangle_strikes(
                spot, sorted_strikes, tte, ce_prices, pe_prices,
                params.target_delta
            )

        self.state.add_log(
            "INFO", "Strategy",
            f"Selected: CE {ce_strike} / PE {pe_strike}"
        )

        # 5. Get entry prices
        ce_price = ce_prices.get(ce_strike, 0)
        pe_price = pe_prices.get(pe_strike, 0)
        if ce_price <= 0 or pe_price <= 0:
            self.state.add_log("ERROR", "Strategy",
                               f"Invalid prices: CE={ce_price}, PE={pe_price}")
            return None

        # 6. Compute Greeks for selected strikes
        ce_greeks = self.greeks.compute_for_strike(
            spot, ce_strike, tte, ce_price, "call"
        )
        pe_greeks = self.greeks.compute_for_strike(
            spot, pe_strike, tte, pe_price, "put"
        )
        self.state.add_log(
            "INFO", "Strategy",
            f"CE Greeks: Δ={ce_greeks.delta:.4f} IV={ce_greeks.iv:.2%} "
            f"Θ={ce_greeks.theta:.2f} | "
            f"PE Greeks: Δ={pe_greeks.delta:.4f} IV={pe_greeks.iv:.2%} "
            f"Θ={pe_greeks.theta:.2f}"
        )

        # 7. Create strategy in DB
        sid = self.db.create_strategy(
            name=params.strategy_type,
            stock_code=stock_code,
            target_delta=params.target_delta,
            ce_strike=ce_strike,
            pe_strike=pe_strike,
            expiry_date=params.expiry_date,
            lots=params.lots,
        )
        self.db.update_strategy(sid, status="entering")
        self.state.set_strategy(sid, self.db.get_strategy(sid))

        # 8. Subscribe to feeds for both legs
        self.conn.subscribe(stock_code, params.expiry_date, ce_strike, "call")
        self.conn.subscribe(stock_code, params.expiry_date, pe_strike, "put")
        time.sleep(0.5)  # let first ticks arrive

        # 9. Execute both legs (near-simultaneously via threads)
        ce_result: OrderResult | None = None
        pe_result: OrderResult | None = None

        def _exec_ce():
            nonlocal ce_result
            ce_result = self.executor.execute_with_chase(
                strategy_id=sid,
                stock_code=stock_code,
                action="sell",
                strike_price=ce_strike,
                right="call",
                expiry_date=params.expiry_date,
                quantity=quantity,
                initial_price=ce_price,
                leg_tag="ce_entry",
            )

        def _exec_pe():
            nonlocal pe_result
            pe_result = self.executor.execute_with_chase(
                strategy_id=sid,
                stock_code=stock_code,
                action="sell",
                strike_price=pe_strike,
                right="put",
                expiry_date=params.expiry_date,
                quantity=quantity,
                initial_price=pe_price,
                leg_tag="pe_entry",
            )

        t_ce = threading.Thread(target=_exec_ce, name=f"Exec-CE-{sid}")
        t_pe = threading.Thread(target=_exec_pe, name=f"Exec-PE-{sid}")
        t_ce.start()
        t_pe.start()
        t_ce.join(timeout=60)
        t_pe.join(timeout=60)

        # 10. Evaluate results
        ce_ok = ce_result and ce_result.success
        pe_ok = pe_result and pe_result.success

        if ce_ok and pe_ok:
            ce_fill = ce_result.filled_price
            pe_fill = pe_result.filled_price
            total_premium = (ce_fill + pe_fill) * quantity

            ce_sl = round(ce_fill * params.sl_multiplier, 2)
            pe_sl = round(pe_fill * params.sl_multiplier, 2)

            self.db.update_strategy(
                sid,
                status="active",
                ce_entry_price=ce_fill,
                pe_entry_price=pe_fill,
                ce_sl_price=ce_sl,
                pe_sl_price=pe_sl,
                total_premium=total_premium,
            )

            # Create position records
            self.db.create_position(
                strategy_id=sid, stock_code=stock_code,
                strike_price=ce_strike, right_type="call",
                expiry_date=params.expiry_date,
                quantity=-quantity,  # short
                entry_price=ce_fill, sl_price=ce_sl,
            )
            self.db.create_position(
                strategy_id=sid, stock_code=stock_code,
                strike_price=pe_strike, right_type="put",
                expiry_date=params.expiry_date,
                quantity=-quantity,
                entry_price=pe_fill, sl_price=pe_sl,
            )

            self.state.set_strategy(sid, self.db.get_strategy(sid))
            self.state.add_log(
                "INFO", "Strategy",
                f"✅ Strategy {sid} ACTIVE | Premium: ₹{total_premium:,.0f} | "
                f"CE {ce_strike}@{ce_fill} SL={ce_sl} | "
                f"PE {pe_strike}@{pe_fill} SL={pe_sl}",
            )
            self.db.log("INFO", "Strategy", f"Strategy {sid} deployed",
                        {"ce_strike": ce_strike, "pe_strike": pe_strike,
                         "ce_fill": ce_fill, "pe_fill": pe_fill})
            return sid

        else:
            # One or both legs failed — attempt cleanup
            self.state.add_log(
                "ERROR", "Strategy",
                f"Leg failure: CE={'OK' if ce_ok else 'FAIL'}, "
                f"PE={'OK' if pe_ok else 'FAIL'}. Cleaning up...",
            )
            self._cleanup_partial(sid, stock_code, params.expiry_date,
                                  quantity, ce_strike, pe_strike,
                                  ce_result, pe_result)
            self.db.update_strategy(sid, status="error")
            self.state.set_strategy(sid, self.db.get_strategy(sid))
            return None

    # ------------------------------------------------ strike selection
    def _select_straddle_strikes(
        self, spot: float, strikes: list[float], gap: int
    ) -> tuple[float, float]:
        """
        ATM straddle: pick the strike closest to spot (±5 point buffer).
        Both CE and PE use the same strike.
        """
        atm = min(strikes, key=lambda s: abs(s - spot))
        # Apply ±5 point buffer: if spot is very close to a strike,
        # use that strike; otherwise still use closest
        if abs(atm - spot) > gap * 0.1:
            # Check next strike
            idx = strikes.index(atm)
            candidates = strikes[max(0, idx - 1):idx + 2]
            atm = min(candidates, key=lambda s: abs(s - spot))
        return atm, atm

    def _select_strangle_strikes(
        self,
        spot: float,
        strikes: list[float],
        tte: float,
        ce_prices: dict[float, float],
        pe_prices: dict[float, float],
        target_delta: float,
    ) -> tuple[float, float]:
        """
        Select OTM call and put strikes closest to target delta.
        """
        atm = min(strikes, key=lambda s: abs(s - spot))
        atm_idx = strikes.index(atm)

        # CE: strikes above ATM (OTM calls)
        otm_calls = [s for s in strikes if s >= atm]
        ce_strike = self.greeks.find_strike_by_delta(
            spot, otm_calls, tte, ce_prices, target_delta, "call"
        )

        # PE: strikes below ATM (OTM puts)
        otm_puts = [s for s in strikes if s <= atm]
        pe_strike = self.greeks.find_strike_by_delta(
            spot, otm_puts, tte, pe_prices, target_delta, "put"
        )

        return ce_strike, pe_strike

    # -------------------------------------------------- cleanup
    def _cleanup_partial(
        self,
        sid: str,
        stock_code: str,
        expiry_date: str,
        quantity: int,
        ce_strike: float,
        pe_strike: float,
        ce_result: OrderResult | None,
        pe_result: OrderResult | None,
    ):
        """If one leg filled but the other didn't, square off the filled leg."""
        if ce_result and ce_result.success:
            self.state.add_log("WARN", "Strategy",
                               f"Squaring off CE leg of failed strategy {sid}")
            self.executor.square_off_position(
                strategy_id=sid, stock_code=stock_code,
                strike_price=ce_strike, right="call",
                expiry_date=expiry_date, quantity=quantity,
                leg_tag="cleanup", use_market=True,
            )
        if pe_result and pe_result.success:
            self.state.add_log("WARN", "Strategy",
                               f"Squaring off PE leg of failed strategy {sid}")
            self.executor.square_off_position(
                strategy_id=sid, stock_code=stock_code,
                strike_price=pe_strike, right="put",
                expiry_date=expiry_date, quantity=quantity,
                leg_tag="cleanup", use_market=True,
            )

    # ------------------------------------------------ exit strategy
    def exit_strategy(self, sid: str, use_market: bool = False):
        """Exit all legs of a strategy."""
        strat = self.db.get_strategy(sid)
        if not strat:
            self.state.add_log("ERROR", "Strategy",
                               f"Strategy {sid} not found")
            return

        positions = self.db.get_positions_for_strategy(sid)
        instrument = None
        for name, spec in INSTRUMENTS.items():
            if spec.stock_code == strat["stock_code"]:
                instrument = spec
                break
        if not instrument:
            self.state.add_log("ERROR", "Strategy",
                               f"Unknown instrument: {strat['stock_code']}")
            return

        self.db.update_strategy(sid, status="exiting")
        self.state.add_log("INFO", "Strategy",
                           f"Exiting strategy {sid}")

        for pos in positions:
            if pos["status"] != "open":
                continue
            qty = abs(pos["quantity"])
            result = self.executor.square_off_position(
                strategy_id=sid,
                stock_code=pos["stock_code"],
                strike_price=pos["strike_price"],
                right=pos["right_type"],
                expiry_date=pos["expiry_date"],
                quantity=qty,
                leg_tag="exit",
                use_market=use_market,
            )
            if result.success:
                pnl = (pos["entry_price"] - result.filled_price) * qty
                self.db.update_position(
                    pos["id"],
                    status="closed",
                    current_price=result.filled_price,
                    pnl=pnl,
                )
                self.state.add_log(
                    "INFO", "Strategy",
                    f"Leg closed: {pos['strike_price']}"
                    f"{pos['right_type'][0].upper()}E "
                    f"@{result.filled_price} PnL=₹{pnl:,.0f}",
                )
            else:
                self.state.add_log(
                    "ERROR", "Strategy",
                    f"Failed to exit leg: {pos['strike_price']}"
                    f"{pos['right_type'][0].upper()}E — {result.error}",
                )

        self.db.update_strategy(sid, status="closed")
        self.state.set_strategy(sid, self.db.get_strategy(sid))
        self.state.add_log("INFO", "Strategy",
                           f"Strategy {sid} fully closed")
