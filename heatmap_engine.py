"""
heatmap_engine.py  — NEW MODULE

P&L Heatmap Generator
  • 2D grid: Spot price vs DTE → P&L
  • 3D surface: Spot × DTE × IV → P&L
  • Breakeven zone visualization
  • Position-level decomposition heatmap
  • Theta bleed heatmap (P&L over time, spot fixed)

Zero breaking-changes: pure add-on.
"""

from __future__ import annotations

import math
from typing import List, Dict, Optional, Tuple

import numpy as np

from greeks_engine import BlackScholes, time_to_expiry
from models import Strategy, Leg, LegStatus, OrderSide, OptionRight


# ────────────────────────────────────────────────────────────────
# Heatmap Engine
# ────────────────────────────────────────────────────────────────

class HeatmapEngine:
    """
    Generates P&L heatmaps for option strategies.

    All methods return dict with:
        x_labels: List[str]   (spot prices or dates)
        y_labels: List[str]   (DTE or IV scenarios)
        z_matrix: List[List[float]]   (P&L values)
        breakeven_points: List[float]
        max_profit: float
        max_loss: float
    """

    def __init__(self, risk_free_rate: float = 0.07):
        self.r = risk_free_rate

    # ── Main: Spot × DTE Heatmap ──────────────────────────────

    def spot_vs_time(
        self,
        strategy: Strategy,
        current_spot: float,
        spot_range_pct: float = 10.0,
        spot_steps: int = 11,
        dte_steps: int = 8,
    ) -> dict:
        """
        Primary heatmap: P&L as spot moves ± range% over time.

        Returns a grid where:
            X = DTE remaining (from expiry to today)
            Y = Spot price scenarios
            Z = Strategy P&L
        """
        active_legs = [l for l in strategy.legs if l.status == LegStatus.ACTIVE]
        if not active_legs:
            return {}

        # Find max expiry DTE
        max_T = max(time_to_expiry(l.expiry_date) for l in active_legs)
        max_dte = max_T * 365.0

        # Spot range
        lo = current_spot * (1 - spot_range_pct / 100)
        hi = current_spot * (1 + spot_range_pct / 100)
        spot_grid = np.linspace(lo, hi, spot_steps)

        # DTE grid: from now to expiry
        dte_grid = np.linspace(0.1, max_dte, dte_steps)

        z_matrix: List[List[float]] = []
        y_labels: List[str] = []

        for test_spot in spot_grid:
            row_pnl = []
            for dte in dte_grid:
                T_remaining = dte / 365.0
                pnl = self._compute_strategy_pnl(active_legs, test_spot, T_remaining)
                row_pnl.append(round(pnl, 1))
            z_matrix.append(row_pnl)
            y_labels.append(f"₹{test_spot:,.0f}")

        x_labels = [f"{dte:.0f}d" for dte in dte_grid]

        # Breakeven analysis (at T=0)
        breakeven = self._find_breakeven(active_legs, spot_grid)

        return {
            "x_labels": x_labels,
            "y_labels": y_labels,
            "z_matrix": z_matrix,
            "breakeven_upper": round(breakeven[1], 0) if len(breakeven) > 1 else 0,
            "breakeven_lower": round(breakeven[0], 0) if breakeven else 0,
            "current_spot": current_spot,
            "max_profit": max(max(row) for row in z_matrix),
            "max_loss": min(min(row) for row in z_matrix),
            "title": f"P&L Heatmap — {strategy.strategy_type.value.replace('_', ' ').title()}",
        }

    # ── Spot × IV Heatmap ─────────────────────────────────────

    def spot_vs_iv(
        self,
        strategy: Strategy,
        current_spot: float,
        current_iv: float = 0.15,
        spot_range_pct: float = 8.0,
        iv_range_pct: float = 50.0,
        spot_steps: int = 9,
        iv_steps: int = 7,
        dte: float = 5.0,
    ) -> dict:
        """
        What-if: P&L as spot AND IV move simultaneously.
        Useful for understanding vega exposure.
        """
        active_legs = [l for l in strategy.legs if l.status == LegStatus.ACTIVE]
        if not active_legs or current_iv <= 0:
            return {}

        lo_s = current_spot * (1 - spot_range_pct / 100)
        hi_s = current_spot * (1 + spot_range_pct / 100)
        spot_grid = np.linspace(lo_s, hi_s, spot_steps)

        iv_lo = current_iv * (1 - iv_range_pct / 100)
        iv_hi = current_iv * (1 + iv_range_pct / 100)
        iv_grid = np.linspace(max(0.01, iv_lo), iv_hi, iv_steps)

        T = dte / 365.0
        z_matrix: List[List[float]] = []
        y_labels: List[str] = []

        for iv_shift in iv_grid:
            row_pnl = []
            for test_spot in spot_grid:
                pnl = self._compute_strategy_pnl(active_legs, test_spot, T, override_iv=iv_shift)
                row_pnl.append(round(pnl, 1))
            z_matrix.append(row_pnl)
            y_labels.append(f"IV={iv_shift*100:.0f}%")

        x_labels = [f"₹{s:,.0f}" for s in spot_grid]

        return {
            "x_labels": x_labels,
            "y_labels": y_labels,
            "z_matrix": z_matrix,
            "max_profit": max(max(row) for row in z_matrix),
            "max_loss": min(min(row) for row in z_matrix),
            "title": "P&L by Spot × IV",
        }

    # ── Theta Bleed Heatmap ───────────────────────────────────

    def theta_bleed(
        self,
        strategy: Strategy,
        current_spot: float,
        days: int = 10,
    ) -> dict:
        """
        Theta bleed: cumulative P&L over next N days at fixed spot.
        Useful for short sellers to visualize daily decay.
        """
        active_legs = [l for l in strategy.legs if l.status == LegStatus.ACTIVE]
        if not active_legs:
            return {}

        max_T = max(time_to_expiry(l.expiry_date) for l in active_legs)
        max_dte = max_T * 365.0
        days = min(days, int(max_dte))

        # ±5% spot scenarios
        spots = [
            current_spot * 0.95,
            current_spot * 0.97,
            current_spot * 0.99,
            current_spot,
            current_spot * 1.01,
            current_spot * 1.03,
            current_spot * 1.05,
        ]

        z_matrix: List[List[float]] = []
        y_labels = [f"₹{s:,.0f}" for s in spots]

        for test_spot in spots:
            row = []
            for day in range(days + 1):
                T_remaining = max(max_T - (day / 365.0), 1e-6)
                pnl = self._compute_strategy_pnl(active_legs, test_spot, T_remaining)
                row.append(round(pnl, 1))
            z_matrix.append(row)

        x_labels = [f"Day {i}" for i in range(days + 1)]

        return {
            "x_labels": x_labels,
            "y_labels": y_labels,
            "z_matrix": z_matrix,
            "max_profit": max(max(row) for row in z_matrix),
            "max_loss": min(min(row) for row in z_matrix),
            "title": f"Theta Bleed — Next {days} Days",
        }

    # ── Leg Decomposition ─────────────────────────────────────

    def leg_decomposition(
        self,
        strategy: Strategy,
        current_spot: float,
        spot_range_pct: float = 10.0,
        steps: int = 11,
    ) -> List[dict]:
        """
        Per-leg payoff at expiry (T=0).
        Returns one dict per leg with x_labels and pnl_values.
        """
        spot_grid = np.linspace(
            current_spot * (1 - spot_range_pct / 100),
            current_spot * (1 + spot_range_pct / 100),
            steps
        )

        decomposition = []
        for leg in strategy.legs:
            if leg.status != LegStatus.ACTIVE or leg.entry_price <= 0:
                continue

            pnl_values = []
            for test_spot in spot_grid:
                pnl = _leg_expiry_pnl(leg, test_spot)
                pnl_values.append(round(pnl, 1))

            decomposition.append({
                "leg_name": leg.display_name,
                "right": leg.right.value.upper(),
                "side": leg.side.value.upper(),
                "x_labels": [f"₹{s:,.0f}" for s in spot_grid],
                "pnl_values": pnl_values,
                "entry_price": leg.entry_price,
                "sl_price": leg.sl_price,
                "strike": int(leg.strike_price),
            })

        return decomposition

    # ── Summary Stats ─────────────────────────────────────────

    def compute_scenario_stats(
        self,
        strategy: Strategy,
        current_spot: float,
        spot_range_pct: float = 5.0,
    ) -> dict:
        """
        Quick scenario summary: bull/bear/flat/extreme scenarios.
        """
        active_legs = [l for l in strategy.legs if l.status == LegStatus.ACTIVE]
        if not active_legs:
            return {}

        T = max((time_to_expiry(l.expiry_date) for l in active_legs), default=0.0)

        scenarios = {
            "flat": current_spot,
            f"up_{spot_range_pct:.0f}%": current_spot * (1 + spot_range_pct / 100),
            f"dn_{spot_range_pct:.0f}%": current_spot * (1 - spot_range_pct / 100),
            "up_10%": current_spot * 1.10,
            "dn_10%": current_spot * 0.90,
        }

        result = {}
        for label, test_spot in scenarios.items():
            pnl_now = self._compute_strategy_pnl(active_legs, test_spot, T)
            pnl_expiry = self._compute_strategy_pnl(active_legs, test_spot, 1e-6)
            result[label] = {
                "spot": round(test_spot, 0),
                "pnl_today": round(pnl_now, 0),
                "pnl_expiry": round(pnl_expiry, 0),
            }

        return result

    # ── Private ───────────────────────────────────────────────

    def _compute_strategy_pnl(
        self,
        legs: List[Leg],
        spot: float,
        T: float,
        override_iv: Optional[float] = None,
    ) -> float:
        total = 0.0
        for leg in legs:
            if leg.entry_price <= 0:
                continue

            iv = override_iv if override_iv else (leg.greeks.iv if leg.greeks.iv > 0.01 else 0.20)

            try:
                model_price = BlackScholes.price(
                    spot, leg.strike_price, max(T, 1e-6),
                    self.r, iv, leg.right
                )
            except Exception:
                model_price = max(0.0, spot - leg.strike_price) if leg.right == OptionRight.CALL else max(0.0, leg.strike_price - spot)

            if leg.side == OrderSide.SELL:
                pnl = (leg.entry_price - model_price) * leg.quantity
            else:
                pnl = (model_price - leg.entry_price) * leg.quantity

            total += pnl

        return total

    def _find_breakeven(self, legs: List[Leg], spot_grid: np.ndarray) -> List[float]:
        """Find breakeven spots at expiry (T≈0)."""
        pnls = [self._compute_strategy_pnl(legs, s, 1e-6) for s in spot_grid]
        breakevens = []

        for i in range(len(pnls) - 1):
            if pnls[i] * pnls[i + 1] < 0:
                # Linear interpolation
                be = spot_grid[i] + (spot_grid[i + 1] - spot_grid[i]) * (-pnls[i] / (pnls[i + 1] - pnls[i]))
                breakevens.append(float(be))

        return breakevens


# ────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────

def _leg_expiry_pnl(leg: Leg, spot: float) -> float:
    """P&L at expiry for a single leg."""
    if leg.right == OptionRight.CALL:
        intrinsic = max(0.0, spot - leg.strike_price)
    else:
        intrinsic = max(0.0, leg.strike_price - spot)

    if leg.side == OrderSide.SELL:
        return (leg.entry_price - intrinsic) * leg.quantity
    else:
        return (intrinsic - leg.entry_price) * leg.quantity
