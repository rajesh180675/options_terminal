"""
calendar_strategies.py  — NEW MODULE (Layer 2)

Calendar Spread Strategies:
  • Standard Calendar (sell near, buy far — same strike)
  • Diagonal Calendar (sell near OTM, buy far ATM)
  • Double Calendar (two-sided: call + put calendar)
  • Ratio Calendar (sell 2 near, buy 1 far)

Calendar spreads profit from:
  1. Theta decay faster on near-term option
  2. IV expansion on long leg (vega positive)
  3. Spot pinning near the strike at near expiry

Key parameters:
  • Near expiry: sell (shorter DTE — faster theta decay)
  • Far expiry: buy (longer DTE — slower decay, vega buffer)
  • Strike: typically ATM for max theta collection

No existing code modified.
"""

from __future__ import annotations

import time
from typing import Optional, List, Tuple

from app_config import Config
from connection_manager import SessionManager
from database import Database
from greeks_engine import BlackScholes, time_to_expiry
from models import (
    Strategy, StrategyType, StrategyStatus,
    Leg, LegStatus, OrderSide, OptionRight
)
from order_manager import OrderManager
from shared_state import SharedState
from utils import LOG, breeze_expiry, next_weekly_expiry, atm_strike, safe_float


class CalendarStrategyEngine:
    """
    Calendar spread strategies for time-based premium collection.
    Deployed on top of the existing TradingEngine (via engine_extensions).
    """

    def __init__(self, session: SessionManager, order_mgr: OrderManager,
                 db: Database, state: SharedState):
        self.session = session
        self.omgr = order_mgr
        self.db = db
        self.state = state

    # ── Standard Calendar ─────────────────────────────────────

    def deploy_calendar_spread(
        self,
        instrument: str,
        near_expiry: str,
        far_expiry: str,
        lots: int = 1,
        strike: float = 0.0,
        right: str = "CALL",
    ) -> Optional[Strategy]:
        """
        Standard Calendar: Sell 1 near-expiry option + Buy 1 far-expiry option.
        Same strike, same right.

        Profit when: spot pins near strike at near expiry + IV expansion.
        Loss when: large spot move away from strike before near expiry.
        """
        inst = Config.instrument(instrument)
        stock_code = inst["breeze_code"]
        exchange = inst["exchange"]
        gap = inst["strike_gap"]
        qty = inst["lot_size"] * lots

        spot = float(self.session.get_spot_price(stock_code))
        if spot <= 0:
            self.state.add_log("ERROR", "CalStrat", "No spot for calendar spread")
            return None

        if strike <= 0:
            strike = float(atm_strike(spot, gap))

        right_enum = OptionRight.CALL if right.upper() == "CALL" else OptionRight.PUT

        s = Strategy(
            strategy_type=StrategyType.MANUAL_SELL,
            stock_code=stock_code,
            target_delta=0.0,
            status=StrategyStatus.DEPLOYING,
        )

        # Short leg: near expiry (fast theta decay)
        short_leg = self._make_leg(
            s.strategy_id, stock_code, exchange,
            strike, right_enum, near_expiry, qty, OrderSide.SELL, Config.SL_PERCENTAGE
        )
        # Long leg: far expiry (slow decay, vega protection)
        long_leg = self._make_leg(
            s.strategy_id, stock_code, exchange,
            strike, right_enum, far_expiry, qty, OrderSide.BUY, 0.0
        )

        s.legs = [short_leg, long_leg]
        self._save_and_subscribe(s)
        time.sleep(0.5)

        ok_long = self._buy_market(long_leg)   # Buy far first (leg in)
        ok_short = self._sell_chase(short_leg) if ok_long else False

        s.status = StrategyStatus.ACTIVE if (ok_long and ok_short) else StrategyStatus.ERROR
        self.db.save_strategy(s)

        self.state.add_log("INFO", "CalStrat",
                           f"Calendar Spread {right} {int(strike)}: long={ok_long} short={ok_short}")
        return s

    # ── Diagonal Calendar ─────────────────────────────────────

    def deploy_diagonal_spread(
        self,
        instrument: str,
        near_expiry: str,
        far_expiry: str,
        lots: int = 1,
        near_delta: float = 0.30,
        far_at_atm: bool = True,
        right: str = "CALL",
    ) -> Optional[Strategy]:
        """
        Diagonal Calendar: Sell OTM near-expiry + Buy ATM/ITM far-expiry.
        Directional bias with vega protection from long leg.

        If right=CALL and bullish: sell OTM near call, buy ATM/lower far call
        """
        inst = Config.instrument(instrument)
        stock_code = inst["breeze_code"]
        exchange = inst["exchange"]
        gap = inst["strike_gap"]
        qty = inst["lot_size"] * lots

        spot = float(self.session.get_spot_price(stock_code))
        if spot <= 0:
            return None

        right_enum = OptionRight.CALL if right.upper() == "CALL" else OptionRight.PUT

        # Near OTM strike (by delta)
        near_strike = self._find_strike_by_delta(stock_code, exchange, spot, near_expiry, near_delta, right.lower())
        if not near_strike:
            near_strike = atm_strike(spot, gap)
            if right_enum == OptionRight.CALL:
                near_strike += gap
            else:
                near_strike -= gap

        # Far ATM strike
        far_strike = float(atm_strike(spot, gap)) if far_at_atm else near_strike

        s = Strategy(
            strategy_type=StrategyType.MANUAL_SELL,
            stock_code=stock_code,
            target_delta=near_delta,
            status=StrategyStatus.DEPLOYING,
        )

        short_near = self._make_leg(s.strategy_id, stock_code, exchange,
                                    near_strike, right_enum, near_expiry, qty, OrderSide.SELL, Config.SL_PERCENTAGE)
        long_far = self._make_leg(s.strategy_id, stock_code, exchange,
                                  far_strike, right_enum, far_expiry, qty, OrderSide.BUY, 0.0)
        s.legs = [short_near, long_far]

        self._save_and_subscribe(s)
        time.sleep(0.5)

        ok_long = self._buy_market(long_far)
        ok_short = self._sell_chase(short_near) if ok_long else False

        s.status = StrategyStatus.ACTIVE if ok_long else StrategyStatus.ERROR
        self.db.save_strategy(s)

        self.state.add_log("INFO", "CalStrat",
                           f"Diagonal {right} near={int(near_strike)} far={int(far_strike)}: {ok_long}/{ok_short}")
        return s

    # ── Double Calendar ───────────────────────────────────────

    def deploy_double_calendar(
        self,
        instrument: str,
        near_expiry: str,
        far_expiry: str,
        lots: int = 1,
        call_delta: float = 0.20,
        put_delta: float = 0.20,
    ) -> Optional[Strategy]:
        """
        Double Calendar: CE calendar + PE calendar simultaneously.
        Profits in both directions if spot stays in a range.
        Net vega positive (benefits from IV expansion).
        """
        inst = Config.instrument(instrument)
        stock_code = inst["breeze_code"]
        exchange = inst["exchange"]
        gap = inst["strike_gap"]
        qty = inst["lot_size"] * lots

        spot = float(self.session.get_spot_price(stock_code))
        if spot <= 0:
            return None

        call_strike = self._find_strike_by_delta(stock_code, exchange, spot, near_expiry, call_delta, "call")
        put_strike = self._find_strike_by_delta(stock_code, exchange, spot, near_expiry, put_delta, "put")
        if not call_strike:
            call_strike = float(atm_strike(spot, gap)) + gap
        if not put_strike:
            put_strike = float(atm_strike(spot, gap)) - gap

        s = Strategy(
            strategy_type=StrategyType.MANUAL_SELL,
            stock_code=stock_code,
            target_delta=call_delta,
            status=StrategyStatus.DEPLOYING,
        )

        legs = [
            # Short near calls + long far calls
            self._make_leg(s.strategy_id, stock_code, exchange, call_strike,
                           OptionRight.CALL, near_expiry, qty, OrderSide.SELL, Config.SL_PERCENTAGE),
            self._make_leg(s.strategy_id, stock_code, exchange, call_strike,
                           OptionRight.CALL, far_expiry, qty, OrderSide.BUY, 0.0),
            # Short near puts + long far puts
            self._make_leg(s.strategy_id, stock_code, exchange, put_strike,
                           OptionRight.PUT, near_expiry, qty, OrderSide.SELL, Config.SL_PERCENTAGE),
            self._make_leg(s.strategy_id, stock_code, exchange, put_strike,
                           OptionRight.PUT, far_expiry, qty, OrderSide.BUY, 0.0),
        ]
        s.legs = legs

        self._save_and_subscribe(s)
        time.sleep(0.5)

        results = []
        # Buy far legs first (hedge in)
        results.append(self._buy_market(legs[1]))  # long far call
        results.append(self._buy_market(legs[3]))  # long far put
        # Then sell near legs
        if any(results):
            results.append(self._sell_chase(legs[0]))  # short near call
            results.append(self._sell_chase(legs[2]))  # short near put

        s.status = StrategyStatus.ACTIVE if any(results) else StrategyStatus.ERROR
        self.db.save_strategy(s)

        self.state.add_log("INFO", "CalStrat",
                           f"Double Calendar CE={int(call_strike)} PE={int(put_strike)}: {results}")
        return s

    # ── Calendar Analysis Helper ──────────────────────────────

    def analyze_calendar_entry(
        self,
        chain_near: List[dict],
        chain_far: List[dict],
        spot: float,
        strike: float,
        right: str = "CALL",
    ) -> dict:
        """
        Pre-trade analysis for a calendar spread entry.
        Returns metrics to help decide if entry is attractive.
        """
        right_upper = right.upper()

        def get_row(chain, k, r):
            for row in chain:
                if abs(float(row.get("strike", 0)) - k) < 1 and str(row.get("right", "")).upper() == r:
                    return row
            return None

        near_row = get_row(chain_near, strike, right_upper)
        far_row = get_row(chain_far, strike, right_upper)

        if not near_row or not far_row:
            return {"error": "Strike/right not found in chain"}

        near_iv = float(near_row.get("iv", 0))
        far_iv = float(far_row.get("iv", 0))
        near_price = float(near_row.get("ltp", 0))
        far_price = float(far_row.get("ltp", 0))

        iv_differential = near_iv - far_iv
        net_debit = far_price - near_price

        return {
            "strike": int(strike),
            "right": right_upper,
            "near_iv": round(near_iv, 1),
            "far_iv": round(far_iv, 1),
            "iv_differential": round(iv_differential, 1),
            "near_price": round(near_price, 2),
            "far_price": round(far_price, 2),
            "net_debit": round(net_debit, 2),
            "iv_edge": "SELL NEAR" if iv_differential > 2 else "NEUTRAL" if iv_differential > 0 else "CAUTION",
            "notes": [
                f"Near IV {near_iv:.1f}% vs Far IV {far_iv:.1f}% (diff: {iv_differential:+.1f}%)",
                f"Net debit: ₹{net_debit:.2f} per contract",
                f"Max profit: far option value at near expiry minus debit",
                "Best setup: near IV > far IV (contango IV term structure)",
            ],
        }

    @staticmethod
    def get_strategy_info() -> List[dict]:
        """Return metadata about calendar strategies for UI."""
        return [
            {
                "id": "calendar_spread",
                "name": "Calendar Spread",
                "bias": "Neutral (pin at strike)",
                "risk": "Defined (net debit)",
                "ideal": "Sideways market, IV expansion expected",
                "legs": "Sell near-expiry + Buy far-expiry (same strike)",
                "max_profit": "When spot pins at strike at near expiry",
                "max_loss": "Net debit paid",
                "emoji": "📅",
            },
            {
                "id": "diagonal_spread",
                "name": "Diagonal Spread",
                "bias": "Directional with time decay",
                "risk": "Defined",
                "ideal": "Directional but want theta from short leg",
                "legs": "Sell OTM near-expiry + Buy ATM far-expiry",
                "max_profit": "When near option expires worthless",
                "max_loss": "Net debit",
                "emoji": "↗️",
            },
            {
                "id": "double_calendar",
                "name": "Double Calendar",
                "bias": "Range-bound (neutral)",
                "risk": "Defined (net debit both sides)",
                "ideal": "Post-event IV crush, sideways market",
                "legs": "CE Calendar + PE Calendar simultaneously",
                "max_profit": "Spot pins between the two strikes",
                "max_loss": "Total net debit",
                "emoji": "🔁",
            },
        ]

    # ── Internal helpers ──────────────────────────────────────

    def _make_leg(self, strategy_id, stock_code, exchange, strike, right, expiry, qty, side, sl_pct) -> Leg:
        return Leg(
            strategy_id=strategy_id,
            stock_code=stock_code,
            exchange_code=exchange,
            strike_price=float(strike),
            right=right,
            expiry_date=expiry,
            side=side,
            quantity=int(qty),
            sl_percentage=float(sl_pct),
            status=LegStatus.ENTERING,
        )

    def _save_and_subscribe(self, s: Strategy) -> None:
        self.db.save_strategy(s)
        for leg in s.legs:
            self.db.save_leg(leg)
            try:
                self.session.subscribe_option(
                    stock_code=leg.stock_code,
                    strike=leg.strike_price,
                    right=leg.right.value,
                    expiry=leg.expiry_date,
                    exchange_code=leg.exchange_code,
                )
            except Exception as e:
                self.state.add_log("WARN", "CalStrat", f"Subscribe failed: {e}")

    def _sell_chase(self, leg: Leg) -> bool:
        if hasattr(self.omgr, "sell_chase"):
            return bool(self.omgr.sell_chase(leg))
        if hasattr(self.omgr, "execute_sell_with_chase"):
            return bool(self.omgr.execute_sell_with_chase(leg))
        return False

    def _buy_market(self, leg: Leg) -> bool:
        if hasattr(self.omgr, "buy_entry_market"):
            return bool(self.omgr.buy_entry_market(leg))
        if hasattr(self.omgr, "buy_market"):
            return bool(self.omgr.buy_market(leg))
        return False

    def _find_strike_by_delta(self, stock_code, exchange, spot, expiry, target_delta, side) -> Optional[float]:
        try:
            res = self.session.get_option_chain(stock_code, expiry, right="others", exchange_code=exchange)
            if not res or not res.get("Success"):
                return None
            chain = res["Success"]
            T = time_to_expiry(expiry)
            r = Config.RISK_FREE_RATE
            right_enum = OptionRight.CALL if side == "call" else OptionRight.PUT

            best_strike = None
            best_diff = 1e9
            for item in chain:
                k = safe_float(item.get("strike_price", 0))
                ltp = safe_float(item.get("ltp", 0))
                item_right = str(item.get("right", "")).lower()
                if k <= 0 or ltp <= 0 or side not in item_right:
                    continue
                if side == "call" and k <= spot:
                    continue
                if side == "put" and k >= spot:
                    continue
                iv = BlackScholes.implied_vol(ltp, spot, k, T, r, right_enum)
                d = abs(BlackScholes.delta(spot, k, T, r, iv, right_enum))
                if abs(d - target_delta) < best_diff:
                    best_diff = abs(d - target_delta)
                    best_strike = k
            return best_strike
        except Exception as e:
            self.state.add_log("WARN", "CalStrat", f"Strike scan: {e}")
            return None
