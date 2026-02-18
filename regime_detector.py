"""
regime_detector.py  — NEW MODULE

Market Regime Detection
  • Volatility regime (low / normal / high / extreme)
  • IV vs HV comparison (overpriced / fair / underpriced vol)
  • Trend vs mean-reversion identification
  • Optimal strategy for current regime
  • IV term structure shape (contango / backwardation / flat)

Zero breaking-changes: pure add-on.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import List, Optional, Dict, Tuple


# ────────────────────────────────────────────────────────────────
# Regime Enums
# ────────────────────────────────────────────────────────────────

class VolRegime(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    EXTREME = "extreme"

class TrendRegime(str, Enum):
    STRONG_BULL = "strong_bull"
    BULL = "bull"
    NEUTRAL = "neutral"
    BEAR = "bear"
    STRONG_BEAR = "strong_bear"

class IVState(str, Enum):
    CHEAP = "vol_underpriced"
    FAIR = "vol_fair"
    RICH = "vol_overpriced"
    VERY_RICH = "vol_very_overpriced"

class TSShape(str, Enum):
    CONTANGO = "contango"       # near < far (normal)
    FLAT = "flat"
    BACKWARDATION = "backwardation"  # near > far (stressed)


# ────────────────────────────────────────────────────────────────
# Regime Result
# ────────────────────────────────────────────────────────────────

@dataclass
class RegimeResult:
    # Computed at
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%H:%M:%S"))

    # Regime classifications
    vol_regime: VolRegime = VolRegime.NORMAL
    trend_regime: TrendRegime = TrendRegime.NEUTRAL
    iv_state: IVState = IVState.FAIR
    ts_shape: TSShape = TSShape.CONTANGO

    # Raw metrics
    current_iv: float = 0.0
    realized_vol_10d: float = 0.0
    realized_vol_30d: float = 0.0
    iv_rv_ratio: float = 1.0
    iv_percentile: float = 50.0
    iv_rank: float = 50.0

    # Trend metrics
    spot_return_5d: float = 0.0
    spot_return_20d: float = 0.0
    trend_strength: float = 0.0

    # Recommendations
    preferred_strategies: List[str] = field(default_factory=list)
    avoid_strategies: List[str] = field(default_factory=list)
    regime_notes: List[str] = field(default_factory=list)
    confidence: float = 0.5   # 0–1

    def to_dict(self) -> dict:
        return {
            "vol_regime": self.vol_regime.value,
            "trend_regime": self.trend_regime.value,
            "iv_state": self.iv_state.value,
            "ts_shape": self.ts_shape.value,
            "current_iv": self.current_iv,
            "hv_10d": self.realized_vol_10d,
            "hv_30d": self.realized_vol_30d,
            "iv_rv_ratio": round(self.iv_rv_ratio, 2),
            "ivp": round(self.iv_percentile, 1),
            "ivr": round(self.iv_rank, 1),
            "return_5d": round(self.spot_return_5d, 2),
            "return_20d": round(self.spot_return_20d, 2),
            "preferred": self.preferred_strategies,
            "avoid": self.avoid_strategies,
            "notes": self.regime_notes,
            "confidence": round(self.confidence, 2),
        }


# ────────────────────────────────────────────────────────────────
# Regime Detector
# ────────────────────────────────────────────────────────────────

class RegimeDetector:
    """
    Detects current market regime from spot history and option chain IVs.

    Usage:
        detector = RegimeDetector()
        regime = detector.analyze(
            current_iv=0.15,
            spot_history=[24000, 23900, ...],  # daily closes, newest last
            term_structure=[{"dte": 7, "atm_iv": 15.2}, {"dte": 30, "atm_iv": 14.5}],
            iv_history=[14.0, 15.2, ...]       # 252 trading days of ATM IV
        )
    """

    # Thresholds (can be tuned)
    VOL_LOW = 0.12       # Below 12% = low vol regime
    VOL_HIGH = 0.22      # Above 22% = high vol regime
    VOL_EXTREME = 0.35   # Above 35% = extreme vol
    IV_CHEAP_RATIO = 0.85  # IV < 85% of HV = cheap
    IV_RICH_RATIO = 1.20   # IV > 120% of HV = rich
    IV_VERY_RICH_RATIO = 1.50

    def analyze(
        self,
        current_iv: float,
        spot_history: Optional[List[float]] = None,
        term_structure: Optional[List[dict]] = None,
        iv_history: Optional[List[float]] = None,
    ) -> RegimeResult:
        result = RegimeResult(current_iv=current_iv)

        if current_iv <= 0:
            result.regime_notes.append("No IV data available")
            return result

        # ── 1. Vol regime ──────────────────────────────────────
        result.vol_regime = self._classify_vol_regime(current_iv)

        # ── 2. Realized vol ────────────────────────────────────
        if spot_history and len(spot_history) >= 5:
            result.realized_vol_10d = self._historical_vol(spot_history, 10)
            result.realized_vol_30d = self._historical_vol(spot_history, 30)

            if result.realized_vol_10d > 0:
                result.iv_rv_ratio = current_iv / result.realized_vol_10d
                result.iv_state = self._classify_iv_state(result.iv_rv_ratio)

            # Trend
            result.spot_return_5d = self._return(spot_history, 5)
            result.spot_return_20d = self._return(spot_history, 20)
            result.trend_regime = self._classify_trend(
                result.spot_return_5d, result.spot_return_20d
            )
            result.trend_strength = self._trend_strength(spot_history)

        # ── 3. IV percentile / rank ────────────────────────────
        if iv_history and len(iv_history) >= 20:
            ivp, ivr = self._iv_percentile_rank(current_iv * 100, iv_history)
            result.iv_percentile = ivp
            result.iv_rank = ivr

        # ── 4. Term structure ──────────────────────────────────
        if term_structure:
            result.ts_shape = self._classify_ts(term_structure)

        # ── 5. Strategy recommendations ───────────────────────
        self._build_recommendations(result)

        # ── 6. Confidence ─────────────────────────────────────
        data_quality = sum([
            bool(spot_history and len(spot_history) >= 10),
            bool(iv_history and len(iv_history) >= 50),
            bool(term_structure),
        ])
        result.confidence = 0.4 + data_quality * 0.2

        return result

    # ── Classifiers ───────────────────────────────────────────

    def _classify_vol_regime(self, iv: float) -> VolRegime:
        if iv < self.VOL_LOW:
            return VolRegime.LOW
        if iv < self.VOL_HIGH:
            return VolRegime.NORMAL
        if iv < self.VOL_EXTREME:
            return VolRegime.HIGH
        return VolRegime.EXTREME

    def _classify_iv_state(self, ratio: float) -> IVState:
        if ratio < self.IV_CHEAP_RATIO:
            return IVState.CHEAP
        if ratio < self.IV_RICH_RATIO:
            return IVState.FAIR
        if ratio < self.IV_VERY_RICH_RATIO:
            return IVState.RICH
        return IVState.VERY_RICH

    def _classify_trend(self, ret_5d: float, ret_20d: float) -> TrendRegime:
        """Combined 5d and 20d return for trend classification."""
        score = ret_5d * 0.6 + ret_20d * 0.4  # weighted

        if score > 3.0:
            return TrendRegime.STRONG_BULL
        if score > 1.0:
            return TrendRegime.BULL
        if score < -3.0:
            return TrendRegime.STRONG_BEAR
        if score < -1.0:
            return TrendRegime.BEAR
        return TrendRegime.NEUTRAL

    def _classify_ts(self, term_structure: List[dict]) -> TSShape:
        """
        Classify term structure shape.
        term_structure: [{dte: float, atm_iv: float}, ...]
        """
        sorted_ts = sorted(term_structure, key=lambda x: x.get("dte", 0))
        if len(sorted_ts) < 2:
            return TSShape.FLAT

        near_iv = sorted_ts[0].get("atm_iv", 0)
        far_iv = sorted_ts[-1].get("atm_iv", 0)

        if near_iv <= 0 or far_iv <= 0:
            return TSShape.FLAT

        slope = (far_iv - near_iv) / far_iv

        if slope > 0.05:
            return TSShape.CONTANGO
        if slope < -0.05:
            return TSShape.BACKWARDATION
        return TSShape.FLAT

    # ── Strategy Recommendations ──────────────────────────────

    def _build_recommendations(self, r: RegimeResult) -> None:
        preferred = []
        avoid = []
        notes = []

        # Vol regime → strategy bias
        if r.vol_regime == VolRegime.LOW:
            avoid.extend(["short_straddle", "short_strangle"])
            preferred.extend(["long_straddle", "calendar_spread"])
            notes.append("Low vol: premium selling has poor risk/reward")

        elif r.vol_regime == VolRegime.NORMAL:
            preferred.extend(["short_strangle", "iron_condor"])
            notes.append("Normal vol: range-bound strategies favored")

        elif r.vol_regime == VolRegime.HIGH:
            preferred.extend(["short_straddle", "short_strangle"])
            notes.append("High vol: premium selling attractive; use tighter SL")

        elif r.vol_regime == VolRegime.EXTREME:
            preferred.extend(["iron_butterfly", "defined_risk"])
            avoid.append("naked_short")
            notes.append("Extreme vol: use defined-risk strategies only")

        # IV state adjustment
        if r.iv_state == IVState.CHEAP:
            preferred = [s for s in preferred if "short" not in s]
            preferred.extend(["long_straddle", "long_strangle"])
            notes.append(f"IV cheap vs HV (ratio={r.iv_rv_ratio:.2f}): favor buying vol")

        elif r.iv_state in (IVState.RICH, IVState.VERY_RICH):
            notes.append(f"IV rich vs HV (ratio={r.iv_rv_ratio:.2f}): vol selling attractive")

        # Trend regime
        if r.trend_regime in (TrendRegime.STRONG_BULL, TrendRegime.STRONG_BEAR):
            avoid.append("short_straddle")
            preferred.append("directional_spread")
            notes.append("Strong trend: delta-neutral strategies risky")

        elif r.trend_regime == TrendRegime.NEUTRAL:
            notes.append("Neutral trend: good for market-neutral strategies")

        # Term structure
        if r.ts_shape == TSShape.BACKWARDATION:
            preferred.append("calendar_spread")
            notes.append("Backwardation in TS: calendar spreads have edge (sell near, buy far)")
        elif r.ts_shape == TSShape.CONTANGO:
            notes.append("Normal contango TS: standard weekly/monthly premium selling")

        # IV percentile
        if r.iv_percentile > 80:
            notes.append(f"IVP={r.iv_percentile:.0f}%: historically high vol — premium selling favored")
        elif r.iv_percentile < 20:
            notes.append(f"IVP={r.iv_percentile:.0f}%: historically low vol — avoid short premium")

        # Deduplicate
        r.preferred_strategies = list(dict.fromkeys(preferred))
        r.avoid_strategies = list(dict.fromkeys(avoid))
        r.regime_notes = notes

    # ── Statistical Helpers ───────────────────────────────────

    def _historical_vol(self, prices: List[float], window: int) -> float:
        """Annualized historical volatility from closing prices."""
        if len(prices) < window + 1:
            window = len(prices) - 1
        if window < 2:
            return 0.0

        recent = prices[-window-1:]
        log_returns = [math.log(recent[i+1] / recent[i]) for i in range(len(recent)-1) if recent[i] > 0]

        if len(log_returns) < 2:
            return 0.0

        mean = sum(log_returns) / len(log_returns)
        variance = sum((x - mean)**2 for x in log_returns) / (len(log_returns) - 1)
        return math.sqrt(variance * 252)

    def _return(self, prices: List[float], window: int) -> float:
        """Percentage return over last N days."""
        if len(prices) < window + 1:
            return 0.0
        old = prices[-window - 1]
        new = prices[-1]
        if old <= 0:
            return 0.0
        return (new / old - 1) * 100

    def _trend_strength(self, prices: List[float], window: int = 20) -> float:
        """
        Linear regression slope normalized by price.
        Positive = uptrend; Negative = downtrend.
        """
        if len(prices) < window:
            window = len(prices)
        if window < 3:
            return 0.0

        pts = prices[-window:]
        n = len(pts)
        xs = list(range(n))
        mx = n / 2
        my = sum(pts) / n

        num = sum((xs[i] - mx) * (pts[i] - my) for i in range(n))
        den = sum((xs[i] - mx)**2 for i in range(n))

        if den == 0:
            return 0.0

        slope = num / den
        return slope / (my / n) * 100 if my > 0 else 0.0

    def _iv_percentile_rank(
        self,
        current_iv_pct: float,
        history_pct: List[float],
    ) -> Tuple[float, float]:
        """
        Returns (IVP, IVR) where history is in percent (e.g., 15.2 for 15.2%).
        """
        if not history_pct:
            return 50.0, 50.0

        arr = sorted(history_pct)
        n = len(arr)
        below = sum(1 for x in arr if x < current_iv_pct)
        ivp = (below / n) * 100

        lo, hi = arr[0], arr[-1]
        ivr = ((current_iv_pct - lo) / (hi - lo)) * 100 if hi > lo else 50.0

        return ivp, ivr
