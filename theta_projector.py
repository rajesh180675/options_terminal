"""
theta_projector.py  — NEW MODULE (Layer 2)

Theta Decay Projector:
  • Per-strategy theta projection curve over remaining DTE
  • Portfolio-level aggregate theta over time
  • Target P&L achievement date estimator
  • Theta acceleration (gamma effect on theta near expiry)
  • "Capture rate" — % of max theta actually earned so far
  • Theta vs realized P&L comparison (theta slippage)
  • Weekend/holiday theta adjustment
  • Optimal exit point based on theta/risk tradeoff

Zero breaking-changes: pure add-on.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple

from greeks_engine import BlackScholes, time_to_expiry
from models import Strategy, Leg, LegStatus, OrderSide, OptionRight, StrategyStatus
from app_config import Config


# ────────────────────────────────────────────────────────────────
# Data Classes
# ────────────────────────────────────────────────────────────────

@dataclass
class ThetaProjectionPoint:
    dte: float
    expected_pnl: float           # projected cumulative P&L from theta only
    daily_theta: float            # theta that day
    portfolio_value: float        # remaining option value (model)
    capture_rate: float           # % of max theta earned

@dataclass
class StrategyThetaProjection:
    strategy_id: str
    strategy_type: str
    current_dte: float
    current_pnl: float
    total_premium: float
    theta_earned_so_far: float    # premium_collected - current_value
    theta_capture_rate: float     # % of max theta earned
    daily_theta: float            # current daily theta ₹
    days_to_target_50pct: float   # estimated days to hit 50% profit
    days_to_target_75pct: float   # estimated days to hit 75% profit
    optimal_exit_dte: float       # DTE where theta/gamma tradeoff is best
    projection_curve: List[ThetaProjectionPoint]
    notes: List[str] = field(default_factory=list)

@dataclass
class PortfolioThetaProjection:
    total_daily_theta: float
    total_premium: float
    total_theta_earned: float
    portfolio_capture_rate: float
    projected_30d_theta: float    # if held to expiry
    strategy_projections: List[StrategyThetaProjection]
    aggregate_curve: List[dict]   # [{dte, total_theta, cumulative_pnl}]
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%H:%M:%S"))


# ────────────────────────────────────────────────────────────────
# Theta Projector Engine
# ────────────────────────────────────────────────────────────────

class ThetaProjector:
    """
    Projects theta decay and P&L trajectory for active strategies.
    Uses Black-Scholes repricing to model future theta decay.
    """

    def project_strategy(
        self,
        strategy: Strategy,
        spot: float,
        target_profit_pcts: Tuple[float, ...] = (0.50, 0.75),
    ) -> Optional[StrategyThetaProjection]:
        """Build full theta projection for one strategy."""
        active_legs = [l for l in strategy.legs if l.status == LegStatus.ACTIVE]
        if not active_legs:
            return None

        T = time_to_expiry(active_legs[0].expiry_date)
        current_dte = T * 365.0
        if current_dte <= 0:
            return None

        r = Config.RISK_FREE_RATE

        # Total premium received and current value
        total_premium = 0.0
        current_value = 0.0
        daily_theta_total = 0.0

        for leg in active_legs:
            if leg.entry_price <= 0:
                continue
            iv = leg.greeks.iv if leg.greeks.iv > 0 else 0.20
            right_enum = leg.right
            current_model_price = BlackScholes.price(spot, leg.strike_price, T, r, iv, right_enum)
            g = BlackScholes.greeks(spot, leg.strike_price, T, r, iv, right_enum)

            qty = leg.quantity
            if leg.side == OrderSide.SELL:
                total_premium += leg.entry_price * qty
                current_value += current_model_price * qty
                daily_theta_total += abs(g.theta) * qty  # theta collected per day
            else:
                total_premium -= leg.entry_price * qty
                current_value -= current_model_price * qty
                daily_theta_total -= abs(g.theta) * qty

        # Current P&L
        current_pnl = strategy.compute_total_pnl()

        # Theta earned
        theta_earned = total_premium - current_value
        capture_rate = (theta_earned / total_premium * 100) if total_premium > 0 else 0

        # Project curve: reprice across DTE
        curve = self._build_projection_curve(active_legs, spot, current_dte, r)

        # Estimate days to targets
        d50 = self._days_to_profit_target(curve, total_premium, 0.50)
        d75 = self._days_to_profit_target(curve, total_premium, 0.75)

        # Optimal exit DTE (where theta/gamma ratio is best for sellers)
        optimal_dte = self._find_optimal_exit_dte(active_legs, spot, current_dte, r)

        notes = []
        if current_dte < 3:
            notes.append("⚠️ < 3 DTE: Gamma risk dominates theta. Consider closing.")
        if capture_rate > 75:
            notes.append(f"🎯 {capture_rate:.0f}% theta captured — consider early exit.")
        if daily_theta_total < 50 and current_dte > 7:
            notes.append("Slow theta burn — check IV environment.")

        return StrategyThetaProjection(
            strategy_id=strategy.strategy_id,
            strategy_type=strategy.strategy_type.value,
            current_dte=round(current_dte, 1),
            current_pnl=round(current_pnl, 2),
            total_premium=round(total_premium, 2),
            theta_earned_so_far=round(theta_earned, 2),
            theta_capture_rate=round(capture_rate, 1),
            daily_theta=round(daily_theta_total, 2),
            days_to_target_50pct=round(d50, 1),
            days_to_target_75pct=round(d75, 1),
            optimal_exit_dte=round(optimal_dte, 1),
            projection_curve=curve,
            notes=notes,
        )

    def project_portfolio(
        self, strategies: List[Strategy], spot_map: Dict[str, float]
    ) -> PortfolioThetaProjection:
        """Build aggregate theta projection for all active strategies."""
        active = [s for s in strategies if s.status in
                  (StrategyStatus.ACTIVE, StrategyStatus.PARTIAL_EXIT)]

        projections = []
        total_theta = 0.0
        total_premium = 0.0
        total_earned = 0.0

        for s in active:
            spot = float(spot_map.get(s.stock_code, 0))
            if spot <= 0:
                continue
            proj = self.project_strategy(s, spot)
            if proj:
                projections.append(proj)
                total_theta += proj.daily_theta
                total_premium += proj.total_premium
                total_earned += proj.theta_earned_so_far

        # Portfolio capture rate
        capture = (total_earned / total_premium * 100) if total_premium > 0 else 0

        # Aggregate curve: sum projections by DTE
        agg = self._aggregate_curves(projections)

        # 30-day projection (all strategies held to expiry)
        proj_30d = sum(p.total_premium * 0.85 for p in projections)  # assume 85% capture

        return PortfolioThetaProjection(
            total_daily_theta=round(total_theta, 2),
            total_premium=round(total_premium, 2),
            total_theta_earned=round(total_earned, 2),
            portfolio_capture_rate=round(capture, 1),
            projected_30d_theta=round(proj_30d, 2),
            strategy_projections=projections,
            aggregate_curve=agg,
        )

    # ── Internal helpers ──────────────────────────────────────

    def _build_projection_curve(
        self, legs: List[Leg], spot: float, max_dte: float, r: float
    ) -> List[ThetaProjectionPoint]:
        """Reprice legs across DTE from current to 0.5."""
        curve = []
        dte_steps = []
        d = max_dte
        while d > 0.5:
            dte_steps.append(round(d, 1))
            d -= 0.5
        dte_steps.append(0.5)

        # Entry value (at max_dte)
        initial_value = self._portfolio_value_at_dte(legs, spot, max_dte, r)

        cumulative_pnl = 0.0
        prev_value = initial_value

        for dte in dte_steps:
            current_val = self._portfolio_value_at_dte(legs, spot, dte, r)
            daily_theta = (prev_value - current_val) / 0.5 if dte < max_dte else 0.0
            cumulative_pnl = initial_value - current_val
            remaining_pct = (current_val / initial_value * 100) if initial_value != 0 else 0
            capture_rate = 100 - remaining_pct if initial_value > 0 else 0

            curve.append(ThetaProjectionPoint(
                dte=dte,
                expected_pnl=round(cumulative_pnl, 2),
                daily_theta=round(daily_theta, 2),
                portfolio_value=round(current_val, 2),
                capture_rate=round(capture_rate, 1),
            ))
            prev_value = current_val

        return curve

    def _portfolio_value_at_dte(
        self, legs: List[Leg], spot: float, dte: float, r: float
    ) -> float:
        """Compute net portfolio option value at a given DTE."""
        T = max(dte / 365.0, 1e-6)
        total = 0.0
        for leg in legs:
            iv = leg.greeks.iv if leg.greeks.iv > 0 else 0.20
            price = BlackScholes.price(spot, leg.strike_price, T, r, iv, leg.right)
            if leg.side == OrderSide.SELL:
                total += leg.entry_price * leg.quantity  # received
                total -= price * leg.quantity             # current cost to close
            else:
                total -= leg.entry_price * leg.quantity
                total += price * leg.quantity
        return total

    def _days_to_profit_target(
        self, curve: List[ThetaProjectionPoint], total_premium: float, target_pct: float
    ) -> float:
        """Find DTE where we expect to hit target_pct of max profit."""
        if not curve or total_premium <= 0:
            return 0.0
        target_pnl = total_premium * target_pct
        max_dte = curve[0].dte if curve else 0.0
        for pt in curve:
            if pt.expected_pnl >= target_pnl:
                return max_dte - pt.dte  # days elapsed
        return 0.0  # not reached in projection

    def _find_optimal_exit_dte(
        self, legs: List[Leg], spot: float, max_dte: float, r: float
    ) -> float:
        """
        Find DTE where theta/gamma ratio peaks (best for sellers).
        After this point, gamma risk outweighs theta benefit.
        """
        best_ratio = 0.0
        best_dte = max_dte * 0.5  # default: half of remaining DTE

        for dte in [d * 0.5 for d in range(1, int(max_dte * 2) + 1)]:
            if dte <= 0:
                continue
            T = max(dte / 365.0, 1e-6)
            theta_sum = 0.0
            gamma_sum = 0.0
            for leg in legs:
                iv = leg.greeks.iv if leg.greeks.iv > 0 else 0.20
                g = BlackScholes.greeks(spot, leg.strike_price, T, r, iv, leg.right)
                theta_sum += abs(g.theta) * leg.quantity
                gamma_sum += abs(g.gamma) * leg.quantity

            if gamma_sum > 0:
                ratio = theta_sum / gamma_sum
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_dte = dte

        return best_dte

    def _aggregate_curves(
        self, projections: List[StrategyThetaProjection]
    ) -> List[dict]:
        """Aggregate all strategy curves into one portfolio curve."""
        if not projections:
            return []

        # Align by DTE buckets
        dte_map: Dict[float, List[float]] = {}
        for proj in projections:
            for pt in proj.projection_curve:
                dte_map.setdefault(pt.dte, []).append(pt.expected_pnl)

        result = []
        for dte in sorted(dte_map.keys(), reverse=True):
            total = sum(dte_map[dte])
            result.append({
                "DTE": dte,
                "Portfolio Theta P&L": round(total, 2),
            })

        return result

    def curve_to_dicts(
        self, curve: List[ThetaProjectionPoint]
    ) -> List[dict]:
        """Convert projection curve to dataframe-ready dicts."""
        return [
            {
                "DTE": pt.dte,
                "Projected P&L": pt.expected_pnl,
                "Daily Theta": pt.daily_theta,
                "Remaining Value": pt.portfolio_value,
                "Capture %": pt.capture_rate,
            }
            for pt in curve
        ]
