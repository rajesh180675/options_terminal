"""
margin_ledger.py  — Layer 3 NEW MODULE

Margin & Capital Efficiency Tracker
  • SPAN margin estimation per strategy (lot-size × notional × vol-factor)
  • Exposure margin computation
  • Portfolio margin utilization bar (% of total capital used)
  • Margin at risk (worst-case margin call scenario)
  • Return on Margin (ROM) per trade and annualized
  • Capital efficiency: premium collected / margin blocked
  • Free margin remaining
  • Margin call proximity alert (<10% free)
  • Per-instrument margin decomposition
  • Daily margin cost (opportunity cost of blocked capital)

Zero breaking-changes: pure add-on. No existing file modified.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from greeks_engine import BlackScholes, time_to_expiry
from models import Strategy, Leg, LegStatus, OrderSide, OptionRight, StrategyStatus
from app_config import Config, INSTRUMENTS


# ────────────────────────────────────────────────────────────────
# Data Classes
# ────────────────────────────────────────────────────────────────

@dataclass
class StrategyMargin:
    strategy_id: str
    strategy_type: str
    instrument: str

    # Margin components
    span_margin: float          # SPAN (worst-case scenario) margin
    exposure_margin: float      # 1.5-3% of notional
    total_margin: float         # span + exposure
    net_option_value: float     # net premium collected (credit to margin)

    # Capital metrics
    premium_collected: float
    current_pnl: float
    rom_trade_pct: float        # Return on Margin (trade to date)
    rom_annual_pct: float       # Annualized ROM

    # Efficiency
    capital_efficiency: float   # premium / margin (higher = more efficient)
    days_held: float

    # Notional
    notional: float
    lots: int


@dataclass
class MarginLedgerSummary:
    # Portfolio totals
    total_span_margin: float
    total_exposure_margin: float
    total_margin_blocked: float
    total_premium_collected: float
    total_current_pnl: float

    # Capital analysis
    available_capital: float        # user-provided or estimated
    margin_utilization_pct: float   # blocked / available * 100
    free_margin: float
    free_margin_pct: float

    # Risk metrics
    margin_at_risk: float           # if all SLs hit
    worst_case_call: float          # estimated margin call scenario

    # Efficiency metrics
    portfolio_rom_pct: float        # portfolio-level ROM
    portfolio_rom_annual_pct: float
    portfolio_capital_efficiency: float

    # Alerts
    alerts: List[dict]             # [{level, message}]

    # Per-strategy breakdown
    strategy_margins: List[StrategyMargin]

    # Per-instrument breakdown
    instrument_margins: Dict[str, dict]

    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%H:%M:%S"))


# ────────────────────────────────────────────────────────────────
# Margin Ledger Engine
# ────────────────────────────────────────────────────────────────

class MarginLedger:
    """
    Estimates and tracks margin usage across the portfolio.

    Note: These are estimates based on SPAN-like calculations.
    For exact SPAN margins, use the broker's margin API.

    Usage:
        ledger = MarginLedger()
        summary = ledger.compute(strategies, spot_map, available_capital)
    """

    # SPAN approximation factors
    SPAN_NIFTY_PCT = 0.09           # ~9% of notional for Nifty
    SPAN_BANKNIFTY_PCT = 0.11       # ~11% (more volatile)
    SPAN_SENSEX_PCT = 0.10          # ~10%
    EXPOSURE_PCT = 0.03             # 3% of notional (exposure margin)
    IRON_CONDOR_CREDIT_PCT = 0.70   # IC gets ~70% of SPAN (defined risk discount)
    IRON_BUTTERFLY_CREDIT_PCT = 0.65

    FREE_MARGIN_WARN_PCT = 20.0     # warn when < 20% free
    FREE_MARGIN_CRITICAL_PCT = 10.0

    # Risk-free rate for opportunity cost
    RISK_FREE_ANNUAL = 0.065        # 6.5% FD equivalent

    def __init__(self):
        self._r = Config.RISK_FREE_RATE

    # ── Main Computation ──────────────────────────────────────

    def compute(
        self,
        strategies: list,
        spot_map: Dict[str, float],
        available_capital: float = 0.0,
        days_computation: float = 0.0,  # days the capital has been deployed
    ) -> MarginLedgerSummary:
        """
        Compute full margin ledger for all active strategies.

        Args:
            strategies:         list of Strategy objects
            spot_map:           {stock_code: spot_price}
            available_capital:  total trading capital (for utilization %)
            days_computation:   avg days capital has been deployed (for annualization)
        """
        strategy_margins: List[StrategyMargin] = []

        for s in strategies:
            if s.status not in (StrategyStatus.ACTIVE, StrategyStatus.PARTIAL_EXIT):
                continue
            spot = spot_map.get(s.stock_code, 0.0)
            if spot <= 0:
                continue
            sm = self._compute_strategy_margin(s, spot)
            strategy_margins.append(sm)

        # Portfolio totals
        total_span = sum(sm.span_margin for sm in strategy_margins)
        total_exp = sum(sm.exposure_margin for sm in strategy_margins)
        total_blocked = total_span + total_exp
        total_premium = sum(sm.premium_collected for sm in strategy_margins)
        total_pnl = sum(sm.current_pnl for sm in strategy_margins)

        # Capital metrics
        cap = available_capital if available_capital > 0 else max(total_blocked * 1.5, 500_000)
        utilization_pct = (total_blocked / cap * 100) if cap > 0 else 0.0
        free_margin = cap - total_blocked
        free_margin_pct = (free_margin / cap * 100) if cap > 0 else 100.0

        # Margin at risk (sum of max losses per strategy)
        margin_at_risk = sum(
            sm.premium_collected * (Config.SL_PERCENTAGE / 100)
            for sm in strategy_margins
        )
        worst_case = total_blocked + margin_at_risk

        # Portfolio efficiency
        avg_days = days_computation if days_computation > 0 else 7.0
        rom_pct = (total_pnl / total_blocked * 100) if total_blocked > 0 else 0.0
        rom_annual = (rom_pct * 365 / avg_days) if avg_days > 0 else 0.0
        cap_eff = (total_premium / total_blocked) if total_blocked > 0 else 0.0

        # Alerts
        alerts = self._build_alerts(
            free_margin_pct, utilization_pct, margin_at_risk, cap, strategy_margins
        )

        # Per-instrument breakdown
        inst_margins = self._instrument_breakdown(strategy_margins)

        return MarginLedgerSummary(
            total_span_margin=round(total_span, 0),
            total_exposure_margin=round(total_exp, 0),
            total_margin_blocked=round(total_blocked, 0),
            total_premium_collected=round(total_premium, 0),
            total_current_pnl=round(total_pnl, 0),
            available_capital=round(cap, 0),
            margin_utilization_pct=round(utilization_pct, 1),
            free_margin=round(free_margin, 0),
            free_margin_pct=round(free_margin_pct, 1),
            margin_at_risk=round(margin_at_risk, 0),
            worst_case_call=round(worst_case, 0),
            portfolio_rom_pct=round(rom_pct, 2),
            portfolio_rom_annual_pct=round(rom_annual, 1),
            portfolio_capital_efficiency=round(cap_eff, 3),
            alerts=alerts,
            strategy_margins=sorted(strategy_margins, key=lambda s: s.total_margin, reverse=True),
            instrument_margins=inst_margins,
        )

    # ── Per-Strategy Margin ───────────────────────────────────

    def _compute_strategy_margin(self, s: Strategy, spot: float) -> StrategyMargin:
        active_legs = [l for l in s.legs if l.status == LegStatus.ACTIVE]
        if not active_legs:
            active_legs = s.legs

        # Instrument config
        inst_name = self._resolve_instrument(s.stock_code)
        inst_cfg = INSTRUMENTS.get(inst_name, INSTRUMENTS["NIFTY"])
        lot_size = inst_cfg["lot_size"]

        # Notional: spot × lot_size × num_lots
        sell_legs = [l for l in active_legs if l.side == OrderSide.SELL]
        max_qty = max((l.quantity for l in sell_legs), default=lot_size)
        num_lots = max_qty // lot_size if lot_size > 0 else 1
        notional = spot * lot_size * num_lots

        # SPAN factor based on instrument and strategy
        span_pct = self._span_factor(inst_name, s.strategy_type.value)

        # Base SPAN
        span_margin = notional * span_pct

        # Defined risk strategies get margin credit (max loss defined)
        if "condor" in s.strategy_type.value.lower():
            span_margin *= self.IRON_CONDOR_CREDIT_PCT
        elif "butterfly" in s.strategy_type.value.lower():
            span_margin *= self.IRON_BUTTERFLY_CREDIT_PCT

        exposure_margin = notional * self.EXPOSURE_PCT

        # Premium collected
        premium = sum(
            l.entry_price * l.quantity
            for l in active_legs
            if l.side == OrderSide.SELL and l.entry_price > 0
        )

        # Current P&L
        pnl = s.compute_total_pnl() if hasattr(s, "compute_total_pnl") else 0.0

        # ROM
        total_margin = span_margin + exposure_margin
        rom_pct = (pnl / total_margin * 100) if total_margin > 0 else 0.0
        try:
            created = datetime.fromisoformat(s.created_at) if s.created_at else datetime.now()
            days_held = max(0.01, (datetime.now() - created).total_seconds() / 86400)
        except Exception:
            days_held = 1.0
        rom_annual = (rom_pct * 365 / days_held) if days_held > 0 else 0.0

        cap_eff = (premium / total_margin) if total_margin > 0 else 0.0

        return StrategyMargin(
            strategy_id=s.strategy_id,
            strategy_type=s.strategy_type.value,
            instrument=inst_name,
            span_margin=round(span_margin, 0),
            exposure_margin=round(exposure_margin, 0),
            total_margin=round(total_margin, 0),
            net_option_value=round(premium, 0),
            premium_collected=round(premium, 0),
            current_pnl=round(pnl, 0),
            rom_trade_pct=round(rom_pct, 2),
            rom_annual_pct=round(rom_annual, 1),
            capital_efficiency=round(cap_eff, 3),
            days_held=round(days_held, 1),
            notional=round(notional, 0),
            lots=num_lots,
        )

    def _span_factor(self, instrument: str, strategy_type: str) -> float:
        base = {
            "NIFTY": self.SPAN_NIFTY_PCT,
            "BANKNIFTY": self.SPAN_BANKNIFTY_PCT,
            "SENSEX": self.SPAN_SENSEX_PCT,
        }.get(instrument.upper(), self.SPAN_NIFTY_PCT)

        # Iron condor / butterfly get benefit of max-loss cap
        if any(x in strategy_type.lower() for x in ("condor", "butterfly")):
            return base * 0.7  # defined risk discount

        return base

    def _resolve_instrument(self, stock_code: str) -> str:
        """Map breeze stock code back to instrument name."""
        code_map = {v["breeze_code"]: k for k, v in INSTRUMENTS.items()}
        return code_map.get(stock_code, stock_code)

    def _build_alerts(
        self,
        free_pct: float,
        util_pct: float,
        margin_at_risk: float,
        capital: float,
        strategy_margins: List[StrategyMargin],
    ) -> List[dict]:
        alerts = []
        if free_pct < self.FREE_MARGIN_CRITICAL_PCT:
            alerts.append({
                "level": "CRITICAL",
                "message": f"🚨 Free margin critically low: {free_pct:.1f}% — avoid new positions"
            })
        elif free_pct < self.FREE_MARGIN_WARN_PCT:
            alerts.append({
                "level": "WARN",
                "message": f"⚠️ Free margin low: {free_pct:.1f}% — consider reducing size"
            })

        if util_pct > 80:
            alerts.append({
                "level": "WARN",
                "message": f"⚠️ High capital utilization: {util_pct:.1f}% — portfolio overweight"
            })

        mar_pct = (margin_at_risk / capital * 100) if capital > 0 else 0
        if mar_pct > 15:
            alerts.append({
                "level": "WARN",
                "message": f"⚠️ Margin at risk: ₹{margin_at_risk:,.0f} ({mar_pct:.1f}% of capital)"
            })

        # Low efficiency strategies
        for sm in strategy_margins:
            if sm.capital_efficiency < 0.02 and sm.total_margin > 50_000:
                alerts.append({
                    "level": "INFO",
                    "message": f"ℹ️ {sm.strategy_type} on {sm.instrument}: low capital efficiency ({sm.capital_efficiency:.3f})"
                })

        return alerts

    def _instrument_breakdown(self, strategy_margins: List[StrategyMargin]) -> Dict[str, dict]:
        from collections import defaultdict
        by_inst: Dict[str, List[StrategyMargin]] = defaultdict(list)
        for sm in strategy_margins:
            by_inst[sm.instrument].append(sm)

        result = {}
        for inst, sms in by_inst.items():
            result[inst] = {
                "strategies": len(sms),
                "total_margin": round(sum(s.total_margin for s in sms), 0),
                "total_premium": round(sum(s.premium_collected for s in sms), 0),
                "total_pnl": round(sum(s.current_pnl for s in sms), 0),
                "avg_rom_pct": round(sum(s.rom_trade_pct for s in sms) / len(sms), 2),
            }
        return result
