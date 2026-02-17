# ═══════════════════════════════════════════════════════════════
# FILE: adjustment_engine.py  (NEW)
# ═══════════════════════════════════════════════════════════════
"""
Position intelligence:
  • Delta drift monitoring with thresholds
  • Adjustment suggestions (add hedge, roll, widen)
  • Roll management (detect when to roll near-expiry positions)
  • Auto-adjustment execution (optional)
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional
from enum import Enum

from models import Strategy, Leg, LegStatus, OrderSide, OptionRight, StrategyStatus
from greeks_engine import BlackScholes, time_to_expiry
from app_config import Config
from utils import LOG, atm_strike, breeze_expiry, next_weekly_expiry


class AdjustmentType(str, Enum):
    ADD_HEDGE = "add_hedge"
    ROLL_NEAR = "roll_to_next_expiry"
    WIDEN_STRANGLE = "widen_strangle"
    CLOSE_WINNER = "close_profitable_leg"
    REDUCE_SIZE = "reduce_position_size"


@dataclass
class Adjustment:
    strategy_id: str = ""
    type: AdjustmentType = AdjustmentType.ADD_HEDGE
    severity: str = "INFO"        # INFO, WARN, CRITICAL
    message: str = ""
    details: str = ""
    suggested_action: str = ""
    timestamp: str = field(
        default_factory=lambda: datetime.now().strftime("%H:%M:%S")
    )


class AdjustmentEngine:
    """Generates actionable suggestions for position management."""

    DELTA_WARN_THRESHOLD = 30.0     # per lot
    DELTA_CRITICAL_THRESHOLD = 50.0
    ROLL_DTE_THRESHOLD = 0.5        # days to expiry
    PROFIT_CLOSE_THRESHOLD = 80.0   # % of max profit
    GAMMA_WARN_THRESHOLD = 2.0

    def __init__(self):
        self._last_suggestions: List[Adjustment] = []

    def analyze(self, strategies: List[Strategy],
                spot_prices: dict) -> List[Adjustment]:
        """Run all analysis and return adjustment suggestions."""
        suggestions = []

        for strategy in strategies:
            if strategy.status != StrategyStatus.ACTIVE:
                continue

            spot = spot_prices.get(strategy.stock_code, 0)
            if spot <= 0:
                continue

            suggestions.extend(self._check_delta_drift(strategy, spot))
            suggestions.extend(self._check_expiry_proximity(strategy))
            suggestions.extend(self._check_profit_target(strategy))
            suggestions.extend(self._check_gamma_risk(strategy))

        self._last_suggestions = suggestions
        return suggestions

    def get_suggestions(self) -> List[Adjustment]:
        return list(self._last_suggestions)

    def _check_delta_drift(self, s: Strategy, spot: float) -> List[Adjustment]:
        """Alert when portfolio delta exceeds thresholds."""
        ng = s.net_greeks
        abs_delta = abs(ng.delta)
        out = []

        if abs_delta >= self.DELTA_CRITICAL_THRESHOLD:
            direction = "BULLISH" if ng.delta > 0 else "BEARISH"
            out.append(Adjustment(
                strategy_id=s.strategy_id,
                type=AdjustmentType.ADD_HEDGE,
                severity="CRITICAL",
                message=f"Delta {ng.delta:+.1f} is {direction} — hedge required",
                details=(
                    f"Net delta has drifted to {ng.delta:+.1f}. "
                    f"Spot at ₹{spot:,.0f}. "
                    f"Consider adding a directional hedge."
                ),
                suggested_action=(
                    f"{'Sell 1 lot ATM CE' if ng.delta > 0 else 'Sell 1 lot ATM PE'} "
                    f"to reduce delta exposure"
                ),
            ))
        elif abs_delta >= self.DELTA_WARN_THRESHOLD:
            out.append(Adjustment(
                strategy_id=s.strategy_id,
                type=AdjustmentType.ADD_HEDGE,
                severity="WARN",
                message=f"Delta drift: {ng.delta:+.1f} — monitor closely",
                details=f"Delta approaching critical threshold ({self.DELTA_CRITICAL_THRESHOLD})",
            ))

        return out

    def _check_expiry_proximity(self, s: Strategy) -> List[Adjustment]:
        """Suggest rolling when DTE < threshold."""
        out = []
        for leg in s.legs:
            if leg.status != LegStatus.ACTIVE:
                continue
            T = time_to_expiry(leg.expiry_date)
            dte = T * 365

            if dte <= self.ROLL_DTE_THRESHOLD and dte > 0:
                next_exp = next_weekly_expiry(s.stock_code)
                out.append(Adjustment(
                    strategy_id=s.strategy_id,
                    type=AdjustmentType.ROLL_NEAR,
                    severity="WARN",
                    message=f"Near expiry: {leg.display_name} — DTE={dte:.1f}",
                    details=(
                        f"Position expiring in {dte:.1f} days. "
                        f"Gamma risk increases exponentially near expiry."
                    ),
                    suggested_action=(
                        f"Roll {leg.display_name} to next expiry "
                        f"({next_exp.strftime('%d-%b')})"
                    ),
                ))
                break  # One suggestion per strategy is enough

        return out

    def _check_profit_target(self, s: Strategy) -> List[Adjustment]:
        """Suggest closing when strategy has captured most of max profit."""
        out = []
        total_premium = sum(
            l.entry_price * l.quantity for l in s.legs
            if l.side == OrderSide.SELL and l.status == LegStatus.ACTIVE
        )
        total_premium -= sum(
            l.entry_price * l.quantity for l in s.legs
            if l.side == OrderSide.BUY and l.status == LegStatus.ACTIVE
        )

        if total_premium <= 0:
            return out

        current_pnl = s.compute_total_pnl()
        profit_pct = (current_pnl / total_premium) * 100

        if profit_pct >= self.PROFIT_CLOSE_THRESHOLD:
            out.append(Adjustment(
                strategy_id=s.strategy_id,
                type=AdjustmentType.CLOSE_WINNER,
                severity="INFO",
                message=f"Profit at {profit_pct:.0f}% of max — consider closing",
                details=(
                    f"P&L: ₹{current_pnl:+,.0f} out of max ₹{total_premium:,.0f}. "
                    f"Remaining premium may not justify gamma risk."
                ),
                suggested_action="Close strategy to lock in profits",
            ))

        return out

    def _check_gamma_risk(self, s: Strategy) -> List[Adjustment]:
        """Alert on high gamma exposure."""
        ng = s.net_greeks
        out = []

        if abs(ng.gamma) > self.GAMMA_WARN_THRESHOLD:
            out.append(Adjustment(
                strategy_id=s.strategy_id,
                type=AdjustmentType.REDUCE_SIZE,
                severity="WARN",
                message=f"High gamma: {ng.gamma:+.2f} — spot moves amplified",
                details="Consider reducing position size or adding wings",
            ))

        return out
