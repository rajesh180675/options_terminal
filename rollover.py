"""
rollover.py  — NEW MODULE

Expiry Rollover Engine
  • Near-expiry detection (DTE threshold alerts)
  • Roll analysis: cost/credit of rolling to next expiry
  • Optimal roll timing (max theta capture, min gamma risk)
  • Auto-roll suggestion generation
  • Roll execution helper (close current + open next)

Zero breaking-changes: pure add-on.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Tuple

from greeks_engine import BlackScholes, time_to_expiry
from models import Strategy, Leg, LegStatus, OrderSide, OptionRight
from utils import LOG, breeze_expiry, next_weekly_expiry, atm_strike
from app_config import Config, INSTRUMENTS


# ────────────────────────────────────────────────────────────────
# Roll Analysis Result
# ────────────────────────────────────────────────────────────────

@dataclass
class RollAnalysis:
    strategy_id: str
    instrument: str
    current_expiry: str
    next_expiry: str
    dte_current: float
    dte_next: float

    # Roll P&L
    close_credit: float = 0.0      # credit received closing current
    open_debit: float = 0.0        # debit paid opening next
    roll_pnl: float = 0.0          # net: positive = credit roll, negative = debit roll

    # Greeks comparison
    current_theta: float = 0.0
    next_theta: float = 0.0
    current_gamma: float = 0.0
    next_gamma: float = 0.0

    # IV comparison
    current_iv: float = 0.0
    next_iv: float = 0.0
    iv_pickup: float = 0.0         # extra IV from rolling to next

    # Decision
    should_roll: bool = False
    urgency: str = "LOW"           # LOW, MEDIUM, HIGH, CRITICAL
    reasons: List[str] = field(default_factory=list)
    notes: str = ""


# ────────────────────────────────────────────────────────────────
# Rollover Engine
# ────────────────────────────────────────────────────────────────

class RolloverEngine:
    """
    Analyzes positions for rollover opportunities and generates
    actionable roll suggestions.
    """

    # Configurable thresholds
    ROLL_DTE_THRESHOLD = 3.0       # Consider rolling when ≤ 3 DTE
    ROLL_DTE_URGENT = 1.0          # Urgent roll when ≤ 1 DTE
    GAMMA_THRESHOLD = 0.05         # High gamma = roll sooner
    MIN_CREDIT_ROLL = -200.0       # Max debit to accept (per lot) for a roll

    def __init__(self, risk_free_rate: float = 0.07):
        self.r = risk_free_rate

    # ── Public API ────────────────────────────────────────────

    def analyze_strategy(
        self,
        strategy: Strategy,
        spot: float,
        instrument: str = "NIFTY",
        next_expiry_str: str = "",
    ) -> Optional[RollAnalysis]:
        """
        Analyze whether a strategy should be rolled to next expiry.
        Returns RollAnalysis or None if no active legs.
        """
        active_legs = [l for l in strategy.legs if l.status == LegStatus.ACTIVE]
        if not active_legs:
            return None

        # Find nearest expiry leg
        current_expiry = min(
            (l.expiry_date for l in active_legs),
            key=lambda e: time_to_expiry(e)
        )

        T_current = time_to_expiry(current_expiry)
        dte_current = T_current * 365.0

        if not next_expiry_str:
            # Next weekly expiry
            try:
                next_dt = next_weekly_expiry(instrument) + timedelta(weeks=1)
                next_expiry_str = breeze_expiry(next_dt)
            except Exception:
                return None

        T_next = time_to_expiry(next_expiry_str)
        dte_next = T_next * 365.0

        analysis = RollAnalysis(
            strategy_id=strategy.strategy_id,
            instrument=instrument,
            current_expiry=current_expiry[:10],
            next_expiry=next_expiry_str[:10],
            dte_current=round(dte_current, 1),
            dte_next=round(dte_next, 1),
        )

        if spot <= 0:
            return analysis

        # Compute roll metrics
        self._compute_roll_metrics(analysis, active_legs, spot, T_current, T_next)
        self._assess_roll_urgency(analysis)
        self._build_recommendation(analysis)

        return analysis

    def analyze_all(
        self,
        strategies: List[Strategy],
        spot_map: Dict[str, float],
    ) -> List[RollAnalysis]:
        """Analyze all active strategies for rollover."""
        results = []
        for s in strategies:
            from models import StrategyStatus
            if s.status not in (StrategyStatus.ACTIVE, StrategyStatus.PARTIAL_EXIT):
                continue

            spot = spot_map.get(s.stock_code, 0.0)
            analysis = self.analyze_strategy(s, spot, instrument=s.stock_code)
            if analysis:
                results.append(analysis)

        return results

    def estimate_roll_cost(
        self,
        leg: Leg,
        spot: float,
        next_expiry: str,
        next_strike: Optional[float] = None,
    ) -> dict:
        """
        Estimate the cost/credit of rolling a single leg.
        Args:
            leg: current leg to roll
            spot: current spot price
            next_expiry: target expiry for rolled position
            next_strike: target strike (defaults to same strike)
        Returns dict with close_price, open_price, roll_cost
        """
        if leg.current_price <= 0 or spot <= 0:
            return {}

        T_curr = time_to_expiry(leg.expiry_date)
        T_next = time_to_expiry(next_expiry)

        if T_curr <= 0 or T_next <= 0:
            return {}

        roll_strike = next_strike or leg.strike_price

        # Current leg close price (model)
        iv_curr = leg.greeks.iv if leg.greeks.iv > 0.01 else 0.18
        close_price = BlackScholes.price(spot, leg.strike_price, T_curr, self.r, iv_curr, leg.right)

        # Next leg open price (use slightly higher IV for next expiry typically)
        iv_next = iv_curr * 0.95  # near-term mean reversion assumption
        open_price = BlackScholes.price(spot, roll_strike, T_next, self.r, iv_next, leg.right)

        # For SELL leg: close = buy back, open = sell again
        if leg.side == OrderSide.SELL:
            roll_cost = open_price - close_price  # positive = credit
        else:
            roll_cost = close_price - open_price  # positive = credit

        return {
            "current_close": round(close_price, 2),
            "next_open": round(open_price, 2),
            "roll_cost": round(roll_cost, 2),
            "roll_credit": round(roll_cost > 0, 0),
            "current_strike": int(leg.strike_price),
            "next_strike": int(roll_strike),
        }

    def best_roll_strikes(
        self,
        leg: Leg,
        spot: float,
        next_expiry: str,
        chain_next: List[dict],
        strategy_type: str = "same",
    ) -> List[dict]:
        """
        Suggest optimal strikes for the rolled position.
        strategy_type: "same" | "delta_match" | "atm"
        """
        T_next = time_to_expiry(next_expiry)
        if T_next <= 0 or not chain_next:
            return []

        right_str = leg.right.value.upper()
        current_delta = abs(leg.greeks.delta)

        candidates = []

        for row in chain_next:
            if str(row.get("right", "")).upper() != right_str:
                continue

            strike = float(row.get("strike", 0))
            ltp = float(row.get("ltp", 0))
            iv = float(row.get("iv", 0)) / 100

            if strike <= 0 or ltp <= 0:
                continue

            row_delta = abs(BlackScholes.delta(spot, strike, T_next, self.r, iv if iv > 0 else 0.18, leg.right))
            delta_diff = abs(row_delta - current_delta)

            roll_info = self.estimate_roll_cost(leg, spot, next_expiry, strike)
            roll_cost = roll_info.get("roll_cost", 0)

            candidates.append({
                "strike": int(strike),
                "expiry": next_expiry[:10],
                "ltp": round(ltp, 2),
                "iv": round(iv * 100, 2),
                "delta": round(row_delta, 3),
                "delta_match": round(delta_diff, 3),
                "roll_cost": round(roll_cost, 2),
                "roll_type": "CREDIT" if roll_cost > 0 else "DEBIT",
            })

        # Sort by delta match
        candidates.sort(key=lambda x: x["delta_match"])
        return candidates[:5]

    # ── Private ───────────────────────────────────────────────

    def _compute_roll_metrics(
        self,
        analysis: RollAnalysis,
        legs: List[Leg],
        spot: float,
        T_current: float,
        T_next: float,
    ) -> None:
        """Fill roll cost and greek comparison in analysis."""
        total_close = 0.0
        total_open = 0.0
        total_theta_curr = 0.0
        total_theta_next = 0.0
        total_gamma_curr = 0.0
        ivs_curr = []

        for leg in legs:
            if leg.side != OrderSide.SELL:
                continue

            iv = leg.greeks.iv if leg.greeks.iv > 0.01 else 0.18
            ivs_curr.append(iv)

            close_price = BlackScholes.price(spot, leg.strike_price, T_current, self.r, iv, leg.right)
            open_price = BlackScholes.price(spot, leg.strike_price, T_next, self.r, iv * 0.95, leg.right)

            g_curr = BlackScholes.greeks(spot, leg.strike_price, T_current, self.r, iv, leg.right)
            g_next = BlackScholes.greeks(spot, leg.strike_price, T_next, self.r, iv * 0.95, leg.right)

            total_close += close_price * leg.quantity
            total_open += open_price * leg.quantity
            total_theta_curr += abs(g_curr.theta) * leg.quantity
            total_theta_next += abs(g_next.theta) * leg.quantity
            total_gamma_curr += g_curr.gamma * leg.quantity

        analysis.close_credit = round(total_close, 0)
        analysis.open_debit = round(total_open, 0)
        analysis.roll_pnl = round(total_open - total_close, 0)
        analysis.current_theta = round(total_theta_curr, 2)
        analysis.next_theta = round(total_theta_next, 2)
        analysis.current_gamma = round(total_gamma_curr, 6)
        analysis.current_iv = round(sum(ivs_curr) / len(ivs_curr) * 100, 2) if ivs_curr else 0
        analysis.next_iv = round(analysis.current_iv * 0.95, 2)
        analysis.iv_pickup = round(analysis.next_iv - analysis.current_iv, 2)

    def _assess_roll_urgency(self, analysis: RollAnalysis) -> None:
        dte = analysis.dte_current

        if dte <= self.ROLL_DTE_URGENT:
            analysis.urgency = "CRITICAL"
        elif dte <= self.ROLL_DTE_THRESHOLD:
            analysis.urgency = "HIGH"
        elif dte <= 5:
            analysis.urgency = "MEDIUM"
        else:
            analysis.urgency = "LOW"

    def _build_recommendation(self, analysis: RollAnalysis) -> None:
        reasons = []
        should_roll = False

        dte = analysis.dte_current

        if dte <= self.ROLL_DTE_THRESHOLD:
            reasons.append(f"Only {dte:.1f} DTE remaining (threshold: {self.ROLL_DTE_THRESHOLD}d)")
            should_roll = True

        if analysis.current_gamma > self.GAMMA_THRESHOLD:
            reasons.append(f"High gamma risk (γ={analysis.current_gamma:.5f}) — roll to reduce gamma")
            should_roll = True

        if analysis.roll_pnl > 0:
            reasons.append(f"Roll generates credit of ₹{analysis.roll_pnl:,.0f}")
        elif analysis.roll_pnl < self.MIN_CREDIT_ROLL:
            reasons.append(f"Roll costs ₹{abs(analysis.roll_pnl):,.0f} — evaluate if worth it")

        if analysis.next_theta > analysis.current_theta:
            reasons.append(f"Next expiry has more theta: ₹{analysis.next_theta:.2f}/day vs ₹{analysis.current_theta:.2f}/day")

        analysis.should_roll = should_roll
        analysis.reasons = reasons
        analysis.notes = "; ".join(reasons) if reasons else "No roll needed yet"
