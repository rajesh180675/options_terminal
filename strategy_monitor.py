"""
strategy_monitor.py  — Layer 3 NEW MODULE

Live Strategy Health Monitor
  • Per-strategy health score (0-100)
  • SL proximity alert (% to stop loss)
  • Delta drift monitoring (current vs entry delta)
  • DTE risk scoring (gamma acceleration zone)
  • Profit capture % (how much of premium captured)
  • Time-in-trade tracking
  • Recommended action engine (HOLD, CLOSE, ADJUST, ROLL)
  • Portfolio-level risk heatmap
  • Alert escalation (INFO → WARN → CRITICAL)

Zero breaking-changes: pure add-on. No existing file modified.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from greeks_engine import BlackScholes, time_to_expiry
from models import Strategy, Leg, LegStatus, OrderSide, OptionRight, StrategyStatus
from app_config import Config


# ────────────────────────────────────────────────────────────────
# Health Score Components
# ────────────────────────────────────────────────────────────────

@dataclass
class HealthComponent:
    name: str
    score: float            # 0-100 (100 = perfect)
    weight: float           # contribution weight
    status: str             # OK, WARN, CRITICAL
    detail: str


@dataclass
class StrategyHealth:
    strategy_id: str
    strategy_type: str
    instrument: str
    health_score: float         # 0-100 composite
    health_label: str           # EXCELLENT, GOOD, CAUTION, DANGER, CRITICAL
    health_color: str           # green, yellow, orange, red

    # Key metrics
    pnl: float
    pnl_pct: float              # % of premium collected
    profit_capture_pct: float   # % of max theoretical profit captured
    sl_proximity_pct: float     # % distance to nearest SL (lower = closer to SL)
    delta_drift: float          # current net delta vs target
    dte: float                  # min DTE across legs
    time_in_trade_hrs: float

    # Components
    components: List[HealthComponent]

    # Recommended action
    action: str                 # HOLD, CLOSE_50, CLOSE_ALL, ADJUST_DELTA, ROLL, MONITOR
    action_reason: str
    action_urgency: str         # LOW, MEDIUM, HIGH, CRITICAL

    # Leg details
    legs_summary: List[dict]    # brief per-leg info

    strategy_ref: Optional[object] = None   # ref to original Strategy object


@dataclass
class MonitorSummary:
    strategies: List[StrategyHealth]
    total_pnl: float
    portfolio_health: float     # avg health score
    critical_count: int
    warn_count: int
    ok_count: int
    total_premium: float
    total_profit_capture_pct: float
    max_risk_strategy: Optional[StrategyHealth]
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%H:%M:%S"))


# ────────────────────────────────────────────────────────────────
# Strategy Monitor
# ────────────────────────────────────────────────────────────────

class StrategyMonitor:
    """
    Real-time strategy health scoring and action recommendation.

    Usage:
        monitor = StrategyMonitor()
        summary = monitor.evaluate(strategies, spot_map)
    """

    # Scoring thresholds
    SL_DANGER_PCT = 20.0        # SL proximity < 20% = CRITICAL
    SL_WARN_PCT = 40.0          # < 40% = WARN
    DELTA_DRIFT_WARN = 25.0     # net delta > 25 = warn
    DELTA_DRIFT_CRITICAL = 50.0
    DTE_GAMMA_WARN = 3.0        # DTE < 3 = gamma zone
    DTE_GAMMA_CRITICAL = 1.0
    PROFIT_CAPTURE_CLOSE = 75.0 # > 75% captured = consider closing
    LOSS_LIMIT_PCT = -50.0      # > 50% loss of premium = bad
    MAX_HOLD_HOURS = 168.0      # 7 days = consider rolling

    # Component weights
    WEIGHTS = {
        "sl_proximity": 0.30,
        "pnl_health": 0.25,
        "delta_drift": 0.20,
        "dte_risk": 0.15,
        "time_risk": 0.10,
    }

    def __init__(self):
        self._r = Config.RISK_FREE_RATE

    # ── Main Evaluation ───────────────────────────────────────

    def evaluate(
        self,
        strategies: list,
        spot_map: Dict[str, float],
    ) -> MonitorSummary:
        """Score all active strategies and generate health summary."""
        health_list: List[StrategyHealth] = []

        for s in strategies:
            if not hasattr(s, "status"):
                continue
            if s.status not in (StrategyStatus.ACTIVE, StrategyStatus.PARTIAL_EXIT):
                continue
            spot = spot_map.get(s.stock_code, 0.0)
            if spot <= 0:
                continue
            h = self._evaluate_strategy(s, spot)
            health_list.append(h)

        # Sort: worst first
        health_list.sort(key=lambda h: h.health_score)

        total_pnl = sum(h.pnl for h in health_list)
        total_premium = sum(
            abs(h.pnl / h.pnl_pct * 100) if h.pnl_pct != 0 else 0
            for h in health_list
        )
        avg_health = (
            sum(h.health_score for h in health_list) / len(health_list)
            if health_list else 100.0
        )
        critical = [h for h in health_list if h.health_label in ("CRITICAL", "DANGER")]
        warns = [h for h in health_list if h.health_label == "CAUTION"]
        oks = [h for h in health_list if h.health_label in ("GOOD", "EXCELLENT")]

        avg_profit_capture = (
            sum(h.profit_capture_pct for h in health_list) / len(health_list)
            if health_list else 0.0
        )

        return MonitorSummary(
            strategies=health_list,
            total_pnl=round(total_pnl, 2),
            portfolio_health=round(avg_health, 1),
            critical_count=len(critical),
            warn_count=len(warns),
            ok_count=len(oks),
            total_premium=round(total_premium, 0),
            total_profit_capture_pct=round(avg_profit_capture, 1),
            max_risk_strategy=health_list[0] if health_list else None,
        )

    def evaluate_single(self, strategy, spot: float) -> StrategyHealth:
        return self._evaluate_strategy(strategy, spot)

    # ── Per-Strategy Scoring ──────────────────────────────────

    def _evaluate_strategy(self, s: Strategy, spot: float) -> StrategyHealth:
        active_legs = [l for l in s.legs if l.status == LegStatus.ACTIVE]
        if not active_legs:
            return self._dead_health(s)

        # ── Compute base metrics ──────────────────────────────

        # PnL
        pnl = s.compute_total_pnl() if hasattr(s, "compute_total_pnl") else 0.0
        premium_collected = sum(
            l.entry_price * l.quantity
            for l in active_legs
            if l.side == OrderSide.SELL and l.entry_price > 0
        )
        pnl_pct = (pnl / premium_collected * 100) if premium_collected > 0 else 0.0
        profit_capture_pct = max(0.0, pnl_pct)

        # SL proximity
        sl_proximities = []
        for l in active_legs:
            if l.sl_price > 0 and l.entry_price > 0 and l.current_price > 0:
                if l.side == OrderSide.SELL:
                    dist_to_sl = (l.sl_price - l.current_price)
                    range_from_entry = (l.sl_price - l.entry_price)
                    if range_from_entry > 0:
                        prox_pct = (dist_to_sl / range_from_entry) * 100
                        sl_proximities.append(max(0.0, min(100.0, prox_pct)))
        sl_proximity = min(sl_proximities) if sl_proximities else 100.0

        # Delta drift
        net_delta = 0.0
        for l in active_legs:
            T = time_to_expiry(l.expiry_date)
            if T > 0 and l.current_price > 0:
                try:
                    iv = l.greeks.iv if l.greeks.iv > 0 else BlackScholes.implied_vol(
                        l.current_price, spot, l.strike_price, T, self._r, l.right
                    )
                    g = BlackScholes.greeks(spot, l.strike_price, T, self._r, iv, l.right)
                    sign = -1.0 if l.side == OrderSide.SELL else 1.0
                    net_delta += g.delta * sign * l.quantity
                except Exception:
                    pass
        delta_drift = abs(net_delta)

        # DTE
        dtes = []
        for l in active_legs:
            T = time_to_expiry(l.expiry_date)
            dtes.append(T * 365)
        dte = min(dtes) if dtes else 0.0

        # Time in trade
        try:
            created = datetime.fromisoformat(s.created_at) if s.created_at else datetime.now()
            time_in = (datetime.now() - created).total_seconds() / 3600
        except Exception:
            time_in = 0.0

        # ── Compute component scores ──────────────────────────

        components = [
            self._score_sl_proximity(sl_proximity),
            self._score_pnl(pnl_pct),
            self._score_delta_drift(delta_drift),
            self._score_dte(dte),
            self._score_time(time_in),
        ]

        # Weighted composite
        health_score = sum(c.score * self.WEIGHTS[c.name.lower().replace(" ", "_").replace("-", "_")[:15]]
                           for c in components
                           if c.name.lower().replace(" ", "_").replace("-", "_")[:15] in self.WEIGHTS)

        # Fix: compute manually with key matching
        w_map = {
            "sl_proximity": self.WEIGHTS["sl_proximity"],
            "pnl_health": self.WEIGHTS["pnl_health"],
            "delta_drift": self.WEIGHTS["delta_drift"],
            "dte_risk": self.WEIGHTS["dte_risk"],
            "time_risk": self.WEIGHTS["time_risk"],
        }
        comp_key_map = {
            "SL Proximity": "sl_proximity",
            "P&L Health": "pnl_health",
            "Delta Drift": "delta_drift",
            "DTE Risk": "dte_risk",
            "Time Risk": "time_risk",
        }
        health_score = sum(
            c.score * w_map.get(comp_key_map.get(c.name, ""), 0)
            for c in components
        )
        health_score = max(0.0, min(100.0, health_score))

        label, color = self._health_label(health_score)
        action, reason, urgency = self._recommend_action(
            health_score, sl_proximity, pnl_pct, delta_drift, dte, time_in
        )

        # Leg summary
        legs_summary = []
        for l in active_legs:
            sl_pct_str = ""
            if l.sl_price > 0 and l.current_price > 0:
                sl_pct_str = f"{((l.sl_price - l.current_price) / l.current_price * 100):+.1f}%"
            legs_summary.append({
                "name": f"{l.right.value.upper()} {int(l.strike_price)}",
                "side": l.side.value,
                "entry": round(l.entry_price, 2),
                "current": round(l.current_price, 2),
                "sl": round(l.sl_price, 2),
                "sl_dist": sl_pct_str,
                "pnl": round((l.entry_price - l.current_price) * l.quantity if l.side == OrderSide.SELL
                             else (l.current_price - l.entry_price) * l.quantity, 0),
            })

        return StrategyHealth(
            strategy_id=s.strategy_id,
            strategy_type=s.strategy_type.value,
            instrument=s.stock_code,
            health_score=round(health_score, 1),
            health_label=label,
            health_color=color,
            pnl=round(pnl, 2),
            pnl_pct=round(pnl_pct, 1),
            profit_capture_pct=round(profit_capture_pct, 1),
            sl_proximity_pct=round(sl_proximity, 1),
            delta_drift=round(delta_drift, 1),
            dte=round(dte, 2),
            time_in_trade_hrs=round(time_in, 1),
            components=components,
            action=action,
            action_reason=reason,
            action_urgency=urgency,
            legs_summary=legs_summary,
            strategy_ref=s,
        )

    # ── Component Scorers ─────────────────────────────────────

    def _score_sl_proximity(self, prox_pct: float) -> HealthComponent:
        if prox_pct >= 60:
            return HealthComponent("SL Proximity", 100, self.WEIGHTS["sl_proximity"], "OK",
                                   f"SL far ({prox_pct:.0f}% buffer)")
        elif prox_pct >= self.SL_WARN_PCT:
            score = 50 + (prox_pct - self.SL_WARN_PCT) / (60 - self.SL_WARN_PCT) * 50
            return HealthComponent("SL Proximity", score, self.WEIGHTS["sl_proximity"], "WARN",
                                   f"SL moderately close ({prox_pct:.0f}% buffer)")
        elif prox_pct >= self.SL_DANGER_PCT:
            score = 20 + (prox_pct - self.SL_DANGER_PCT) / (self.SL_WARN_PCT - self.SL_DANGER_PCT) * 30
            return HealthComponent("SL Proximity", score, self.WEIGHTS["sl_proximity"], "WARN",
                                   f"⚠️ SL close ({prox_pct:.0f}% buffer)")
        else:
            return HealthComponent("SL Proximity", max(0, prox_pct), self.WEIGHTS["sl_proximity"], "CRITICAL",
                                   f"🚨 SL imminent ({prox_pct:.0f}% buffer)")

    def _score_pnl(self, pnl_pct: float) -> HealthComponent:
        if pnl_pct >= 50:
            return HealthComponent("P&L Health", 100, self.WEIGHTS["pnl_health"], "OK",
                                   f"Strong profit ({pnl_pct:.0f}% captured)")
        elif pnl_pct >= 20:
            score = 70 + (pnl_pct - 20) / 30 * 30
            return HealthComponent("P&L Health", score, self.WEIGHTS["pnl_health"], "OK",
                                   f"Profitable ({pnl_pct:.0f}% captured)")
        elif pnl_pct >= 0:
            score = 50 + pnl_pct / 20 * 20
            return HealthComponent("P&L Health", score, self.WEIGHTS["pnl_health"], "OK",
                                   f"Modest profit ({pnl_pct:.0f}% captured)")
        elif pnl_pct >= -25:
            score = max(30, 50 + pnl_pct)
            return HealthComponent("P&L Health", score, self.WEIGHTS["pnl_health"], "WARN",
                                   f"Small loss ({pnl_pct:.0f}%)")
        else:
            score = max(0, 30 + pnl_pct)
            return HealthComponent("P&L Health", score, self.WEIGHTS["pnl_health"], "CRITICAL",
                                   f"🚨 Large loss ({pnl_pct:.0f}%)")

    def _score_delta_drift(self, delta: float) -> HealthComponent:
        if delta <= 10:
            return HealthComponent("Delta Drift", 100, self.WEIGHTS["delta_drift"], "OK",
                                   f"Near-neutral (Δ={delta:.1f})")
        elif delta <= self.DELTA_DRIFT_WARN:
            score = 60 + (self.DELTA_DRIFT_WARN - delta) / (self.DELTA_DRIFT_WARN - 10) * 40
            return HealthComponent("Delta Drift", score, self.WEIGHTS["delta_drift"], "OK",
                                   f"Moderate drift (Δ={delta:.1f})")
        elif delta <= self.DELTA_DRIFT_CRITICAL:
            score = 30 + (self.DELTA_DRIFT_CRITICAL - delta) / (self.DELTA_DRIFT_CRITICAL - self.DELTA_DRIFT_WARN) * 30
            return HealthComponent("Delta Drift", score, self.WEIGHTS["delta_drift"], "WARN",
                                   f"⚠️ Delta drifted (Δ={delta:.1f})")
        else:
            return HealthComponent("Delta Drift", max(0, 30 - delta + self.DELTA_DRIFT_CRITICAL),
                                   self.WEIGHTS["delta_drift"], "CRITICAL",
                                   f"🚨 Heavy delta exposure (Δ={delta:.1f})")

    def _score_dte(self, dte: float) -> HealthComponent:
        if dte > 7:
            return HealthComponent("DTE Risk", 100, self.WEIGHTS["dte_risk"], "OK",
                                   f"Ample time ({dte:.1f}d)")
        elif dte > self.DTE_GAMMA_WARN:
            score = 70 + (dte - self.DTE_GAMMA_WARN) / (7 - self.DTE_GAMMA_WARN) * 30
            return HealthComponent("DTE Risk", score, self.WEIGHTS["dte_risk"], "OK",
                                   f"{dte:.1f}d to expiry")
        elif dte > self.DTE_GAMMA_CRITICAL:
            score = 30 + (dte - self.DTE_GAMMA_CRITICAL) / (self.DTE_GAMMA_WARN - self.DTE_GAMMA_CRITICAL) * 40
            return HealthComponent("DTE Risk", score, self.WEIGHTS["dte_risk"], "WARN",
                                   f"⚠️ Gamma zone ({dte:.1f}d)")
        else:
            return HealthComponent("DTE Risk", max(0, dte * 20),
                                   self.WEIGHTS["dte_risk"], "CRITICAL",
                                   f"🚨 Near expiry ({dte:.2f}d) — extreme gamma risk")

    def _score_time(self, hours: float) -> HealthComponent:
        if hours <= 48:
            return HealthComponent("Time Risk", 100, self.WEIGHTS["time_risk"], "OK",
                                   f"Trade is fresh ({hours:.0f}h)")
        elif hours <= 96:
            return HealthComponent("Time Risk", 80, self.WEIGHTS["time_risk"], "OK",
                                   f"Normal hold time ({hours:.0f}h)")
        elif hours <= self.MAX_HOLD_HOURS:
            return HealthComponent("Time Risk", 60, self.WEIGHTS["time_risk"], "WARN",
                                   f"Extended hold ({hours:.0f}h)")
        else:
            return HealthComponent("Time Risk", 40, self.WEIGHTS["time_risk"], "WARN",
                                   f"⚠️ Long-running trade ({hours:.0f}h) — consider roll")

    # ── Health Label ──────────────────────────────────────────

    def _health_label(self, score: float) -> Tuple[str, str]:
        if score >= 80:
            return "EXCELLENT", "green"
        elif score >= 65:
            return "GOOD", "lightgreen"
        elif score >= 45:
            return "CAUTION", "yellow"
        elif score >= 25:
            return "DANGER", "orange"
        else:
            return "CRITICAL", "red"

    # ── Action Recommendation ─────────────────────────────────

    def _recommend_action(
        self, score: float, sl_prox: float, pnl_pct: float,
        delta: float, dte: float, hours: float
    ) -> Tuple[str, str, str]:
        """Returns (action, reason, urgency)."""

        if sl_prox < self.SL_DANGER_PCT:
            return "CLOSE_ALL", f"SL imminent ({sl_prox:.0f}% buffer)", "CRITICAL"

        if pnl_pct <= self.LOSS_LIMIT_PCT:
            return "CLOSE_ALL", f"Max loss exceeded ({pnl_pct:.0f}%)", "CRITICAL"

        if dte <= self.DTE_GAMMA_CRITICAL:
            return "ROLL", f"Near expiry ({dte:.2f}d) — gamma risk extreme", "CRITICAL"

        if pnl_pct >= self.PROFIT_CAPTURE_CLOSE:
            return "CLOSE_50", f"75%+ profit captured ({pnl_pct:.0f}%) — lock in gains", "MEDIUM"

        if delta > self.DELTA_DRIFT_CRITICAL:
            return "ADJUST_DELTA", f"Large delta drift ({delta:.0f}) — hedge required", "HIGH"

        if dte <= self.DTE_GAMMA_WARN:
            return "MONITOR", f"Entering gamma zone ({dte:.1f}d) — watch closely", "HIGH"

        if sl_prox < self.SL_WARN_PCT:
            return "MONITOR", f"SL relatively close ({sl_prox:.0f}% buffer)", "MEDIUM"

        if delta > self.DELTA_DRIFT_WARN:
            return "ADJUST_DELTA", f"Delta drifting ({delta:.0f}) — consider minor hedge", "MEDIUM"

        if hours > self.MAX_HOLD_HOURS:
            return "ROLL", f"Trade open {hours:.0f}h — consider rolling", "LOW"

        if score >= 70:
            return "HOLD", "Strategy healthy — no action needed", "LOW"

        return "MONITOR", f"Score {score:.0f}/100 — watch for deterioration", "LOW"

    # ── Helpers ───────────────────────────────────────────────

    def _dead_health(self, s: Strategy) -> StrategyHealth:
        return StrategyHealth(
            strategy_id=s.strategy_id,
            strategy_type=s.strategy_type.value,
            instrument=s.stock_code,
            health_score=0.0, health_label="CRITICAL", health_color="red",
            pnl=0.0, pnl_pct=0.0, profit_capture_pct=0.0,
            sl_proximity_pct=0.0, delta_drift=0.0, dte=0.0, time_in_trade_hrs=0.0,
            components=[], action="CLOSE_ALL", action_reason="No active legs",
            action_urgency="CRITICAL", legs_summary=[],
        )
