# ═══════════════════════════════════════════════════════════════
# FILE: adjustment_engine.py  (NEW — professional delta management)
# ═══════════════════════════════════════════════════════════════
"""
Portfolio adjustment engine:
  • Monitors net portfolio delta continuously
  • Generates adjustment suggestions when delta drifts
  • Can auto-execute hedges if enabled
  • Logs all decisions (audit trail)
"""

import threading
from typing import List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from models import Strategy, Leg, StrategyStatus, LegStatus, OptionRight, OrderSide, Greeks
from app_config import Config
from shared_state import SharedState
from utils import LOG


@dataclass
class AdjustmentSuggestion:
    timestamp: str
    net_delta: float
    suggestion: str
    hedge_action: str      # "sell_call" / "sell_put" / "none"
    hedge_strike: float
    hedge_quantity: int
    urgency: str           # "low" / "medium" / "high"
    executed: bool = False


class AdjustmentEngine:
    """
    Professional delta management.
    TastyTrade recommends adjusting at 2x initial delta.
    We use configurable threshold (default: 30 net delta).
    """

    def __init__(self, state: SharedState, notifier=None):
        self.state = state
        self.notifier = notifier
        self._suggestions: List[AdjustmentSuggestion] = []
        self._lock = threading.Lock()
        self._last_alert_time: float = 0

    def check_and_suggest(self, strategies: List[Strategy],
                          spot: float) -> Optional[AdjustmentSuggestion]:
        """
        Called every monitor cycle.
        Evaluates portfolio delta and generates suggestions.
        """
        threshold = float(Config._get("ADJUSTMENT_THRESHOLD", "30"))
        if threshold <= 0:
            return None

        # Compute net portfolio delta
        net_delta = 0.0
        active_instrument = None
        for s in strategies:
            if s.status not in (StrategyStatus.ACTIVE, StrategyStatus.PARTIAL_EXIT):
                continue
            for leg in s.legs:
                if leg.status != LegStatus.ACTIVE:
                    continue
                if not active_instrument:
                    active_instrument = leg.stock_code
                sign = -1.0 if leg.side == OrderSide.SELL else 1.0
                net_delta += leg.greeks.delta * sign * leg.quantity

        if abs(net_delta) < threshold:
            return None

        # Generate suggestion
        import time as _time
        now = _time.time()
        if now - self._last_alert_time < 60:
            return None  # Don't spam — max 1 alert per minute
        self._last_alert_time = now

        if net_delta > 0:
            # Portfolio too bullish — need to sell calls or buy puts
            urgency = "high" if abs(net_delta) > threshold * 2 else "medium"
            suggestion = AdjustmentSuggestion(
                timestamp=datetime.now().strftime("%H:%M:%S"),
                net_delta=round(net_delta, 1),
                suggestion=(
                    f"Portfolio delta +{net_delta:.0f} exceeds threshold ±{threshold:.0f}. "
                    f"Consider selling OTM calls or buying puts to reduce bullish exposure."
                ),
                hedge_action="sell_call",
                hedge_strike=spot * 1.02 if spot > 0 else 0,
                hedge_quantity=Config.lot_size(active_instrument or "NIFTY"),
                urgency=urgency,
            )
        else:
            urgency = "high" if abs(net_delta) > threshold * 2 else "medium"
            suggestion = AdjustmentSuggestion(
                timestamp=datetime.now().strftime("%H:%M:%S"),
                net_delta=round(net_delta, 1),
                suggestion=(
                    f"Portfolio delta {net_delta:.0f} exceeds threshold ±{threshold:.0f}. "
                    f"Consider selling OTM puts or buying calls to reduce bearish exposure."
                ),
                hedge_action="sell_put",
                hedge_strike=spot * 0.98 if spot > 0 else 0,
                hedge_quantity=Config.lot_size(active_instrument or "NIFTY"),
                urgency=urgency,
            )

        with self._lock:
            self._suggestions.append(suggestion)
            if len(self._suggestions) > 100:
                self._suggestions = self._suggestions[-50:]

        self.state.add_log(
            "WARN", "Adjust",
            f"Δ drift: {net_delta:+.0f} | {suggestion.hedge_action} suggested"
        )

        # Notify via Telegram
        if self.notifier:
            self.notifier.send(
                f"⚠️ Delta Drift Alert\n"
                f"Net Δ: {net_delta:+.1f}\n"
                f"Action: {suggestion.suggestion}",
                level="WARN"
            )

        return suggestion

    def get_suggestions(self) -> List[AdjustmentSuggestion]:
        with self._lock:
            return list(self._suggestions)
