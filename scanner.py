"""
scanner.py  — NEW MODULE

Options Market Scanner
  • High IV scanner (instruments trading above IV rank threshold)
  • Unusual OI buildup / OI change analysis
  • Put-call skew outliers
  • Expected move breach alerts
  • Largest premium decay opportunities
  • Gamma squeeze / high gamma concentration zones

Zero breaking-changes: pure add-on.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Tuple

from greeks_engine import BlackScholes, time_to_expiry
from models import OptionRight
from utils import LOG, atm_strike, safe_float


# ────────────────────────────────────────────────────────────────
# Scan Result
# ────────────────────────────────────────────────────────────────

@dataclass
class ScanResult:
    scan_type: str
    instrument: str
    strike: float = 0.0
    expiry: str = ""
    right: str = ""
    value: float = 0.0
    description: str = ""
    severity: str = "INFO"     # INFO, WARN, ALERT
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%H:%M:%S"))

    def to_dict(self) -> dict:
        return {
            "scan_type": self.scan_type,
            "instrument": self.instrument,
            "strike": int(self.strike) if self.strike else 0,
            "expiry": self.expiry,
            "right": self.right,
            "value": round(self.value, 2),
            "description": self.description,
            "severity": self.severity,
            "time": self.timestamp,
        }


# ────────────────────────────────────────────────────────────────
# Scanner Engine
# ────────────────────────────────────────────────────────────────

class MarketScanner:
    """
    Scans option chain data and returns actionable signals.

    All methods accept chain (list of dicts) and spot price.
    They are independent — call any subset you need.
    """

    # ── Configuration ─────────────────────────────────────────
    HIGH_IV_THRESHOLD = 30.0          # IV% above this = high IV alert
    IV_RANK_ALERT = 80.0              # IVR above this = extreme
    UNUSUAL_OI_MULTIPLE = 2.5         # OI > 2.5x avg = unusual
    GAMMA_CONCENTRATION_PCT = 20.0    # % OI at a single strike = gamma squeeze
    THETA_DECAY_MIN = 5.0             # Min daily theta for decay scanner (₹)
    MAX_PAIN_DEVIATION_PCT = 1.0      # Alert if spot deviates > 1% from max pain
    VOL_SPREAD_MIN = 2.0              # Min bid-ask spread in IV% to flag illiquid

    def __init__(self):
        self._cache: Dict[str, Tuple[float, List[ScanResult]]] = {}

    def run_all(self, chain: List[dict], spot: float,
                instrument: str = "", expiry: str = "") -> List[ScanResult]:
        """Run all scanners and return combined results."""
        results: List[ScanResult] = []
        if not chain or spot <= 0:
            return results

        results.extend(self.scan_high_iv(chain, spot, instrument))
        results.extend(self.scan_unusual_oi(chain, spot, instrument))
        results.extend(self.scan_gamma_concentration(chain, spot, instrument))
        results.extend(self.scan_theta_rich(chain, spot, instrument, expiry))
        results.extend(self.scan_put_call_skew(chain, spot, instrument))
        results.extend(self.scan_max_pain_deviation(chain, spot, instrument))
        results.extend(self.scan_liquidity(chain, instrument))

        # Sort by severity
        _order = {"ALERT": 0, "WARN": 1, "INFO": 2}
        results.sort(key=lambda r: _order.get(r.severity, 2))
        return results

    # ── Individual Scanners ───────────────────────────────────

    def scan_high_iv(self, chain: List[dict], spot: float,
                     instrument: str = "") -> List[ScanResult]:
        """Flag options with unusually high IV."""
        results = []
        ivs = [float(r.get("iv", 0)) for r in chain if float(r.get("iv", 0)) > 0]
        if not ivs:
            return results

        avg_iv = sum(ivs) / len(ivs)

        for row in chain:
            iv = float(row.get("iv", 0))
            strike = float(row.get("strike", 0))
            right = str(row.get("right", ""))

            if iv <= 0 or strike <= 0:
                continue

            # Flag if IV > threshold AND significantly above chain average
            if iv > self.HIGH_IV_THRESHOLD and iv > avg_iv * 1.5:
                severity = "ALERT" if iv > avg_iv * 2.0 else "WARN"
                results.append(ScanResult(
                    scan_type="HIGH_IV",
                    instrument=instrument,
                    strike=strike,
                    expiry=str(row.get("expiry", "")),
                    right=right,
                    value=iv,
                    description=f"{int(strike)} {right} IV={iv:.1f}% (chain avg={avg_iv:.1f}%)",
                    severity=severity,
                ))

        return results[:5]  # top 5

    def scan_unusual_oi(self, chain: List[dict], spot: float,
                        instrument: str = "") -> List[ScanResult]:
        """Detect unusual OI buildup compared to chain average."""
        results = []
        ois = [float(r.get("oi", 0)) for r in chain if float(r.get("oi", 0)) > 0]
        if len(ois) < 3:
            return results

        avg_oi = sum(ois) / len(ois)
        threshold = avg_oi * self.UNUSUAL_OI_MULTIPLE

        for row in chain:
            oi = float(row.get("oi", 0))
            strike = float(row.get("strike", 0))
            right = str(row.get("right", ""))

            if oi < threshold or strike <= 0:
                continue

            ratio = oi / avg_oi
            moneyness = abs(strike - spot) / spot * 100

            # More interesting if OTM with large OI
            if moneyness > 0.5:
                severity = "ALERT" if ratio > 4.0 else "WARN"
                results.append(ScanResult(
                    scan_type="UNUSUAL_OI",
                    instrument=instrument,
                    strike=strike,
                    expiry=str(row.get("expiry", "")),
                    right=right,
                    value=oi,
                    description=f"{int(strike)} {right} OI={oi:,.0f} ({ratio:.1f}x avg), moneyness={moneyness:.1f}%",
                    severity=severity,
                ))

        results.sort(key=lambda r: r.value, reverse=True)
        return results[:5]

    def scan_gamma_concentration(self, chain: List[dict], spot: float,
                                 instrument: str = "") -> List[ScanResult]:
        """
        Detect high gamma concentration near spot (gamma squeeze risk).
        High call OI near spot = resistance; high put OI near spot = support.
        """
        results = []
        total_oi = sum(float(r.get("oi", 0)) for r in chain)
        if total_oi <= 0:
            return results

        # Look at strikes within 1% of spot
        near_strikes = [
            r for r in chain
            if abs(float(r.get("strike", 0)) - spot) / spot < 0.01
            and float(r.get("oi", 0)) > 0
        ]

        for row in near_strikes:
            oi = float(row.get("oi", 0))
            pct = (oi / total_oi) * 100
            strike = float(row.get("strike", 0))
            right = str(row.get("right", ""))

            if pct >= self.GAMMA_CONCENTRATION_PCT:
                gamma = float(row.get("gamma", 0))
                results.append(ScanResult(
                    scan_type="GAMMA_CONCENTRATION",
                    instrument=instrument,
                    strike=strike,
                    expiry=str(row.get("expiry", "")),
                    right=right,
                    value=pct,
                    description=(f"Γ-squeeze risk: {int(strike)} {right} "
                                 f"has {pct:.0f}% of total OI near spot | γ={gamma:.5f}"),
                    severity="WARN",
                ))

        return results

    def scan_theta_rich(self, chain: List[dict], spot: float,
                        instrument: str = "", expiry: str = "") -> List[ScanResult]:
        """
        Find ATM/NTM options with richest theta (best premium decay candidates).
        Ideal for short-selling: high IV + high theta + near ATM.
        """
        results = []
        T = time_to_expiry(expiry) if expiry else 0.0
        if T <= 0:
            T = 5.0 / 365.0  # fallback 5 DTE

        candidates = []
        for row in chain:
            strike = float(row.get("strike", 0))
            ltp = float(row.get("ltp", 0))
            theta = float(row.get("theta", 0))
            iv = float(row.get("iv", 0))
            right = str(row.get("right", ""))

            if strike <= 0 or ltp <= 0:
                continue

            # Theta should be negative for long; positive theta shown in chain = per-day decay in ₹
            theta_daily = abs(theta) if theta != 0 else 0.0

            # Near ATM check (within 2%)
            moneyness_pct = abs(strike - spot) / spot * 100

            if moneyness_pct < 2.0 and iv > 10 and theta_daily >= self.THETA_DECAY_MIN:
                # Score: theta / ltp = decay rate
                decay_rate = theta_daily / ltp if ltp > 0 else 0
                candidates.append((decay_rate, theta_daily, row))

        candidates.sort(reverse=True)
        for rank, (decay_rate, theta_daily, row) in enumerate(candidates[:5]):
            strike = float(row.get("strike", 0))
            right = str(row.get("right", ""))
            iv = float(row.get("iv", 0))
            ltp = float(row.get("ltp", 0))
            results.append(ScanResult(
                scan_type="THETA_RICH",
                instrument=instrument,
                strike=strike,
                expiry=str(row.get("expiry", "")),
                right=right,
                value=theta_daily,
                description=(f"{int(strike)} {right} LTP={ltp:.1f} "
                             f"Θ={theta_daily:.1f}/day ({decay_rate*100:.1f}% decay) IV={iv:.1f}%"),
                severity="INFO",
            ))

        return results

    def scan_put_call_skew(self, chain: List[dict], spot: float,
                           instrument: str = "") -> List[ScanResult]:
        """Detect extreme put-call skew at each strike."""
        results = []
        # Build paired CE/PE rows
        by_strike: Dict[float, Dict[str, dict]] = {}
        for row in chain:
            strike = float(row.get("strike", 0))
            right = str(row.get("right", "")).upper()
            if strike > 0 and right in ("CALL", "PUT"):
                by_strike.setdefault(strike, {})[right] = row

        for strike, sides in by_strike.items():
            if "CALL" not in sides or "PUT" not in sides:
                continue
            ce_iv = float(sides["CALL"].get("iv", 0))
            pe_iv = float(sides["PUT"].get("iv", 0))
            if ce_iv <= 0 or pe_iv <= 0:
                continue

            skew = pe_iv - ce_iv
            if abs(skew) > 5.0:
                moneyness_pct = abs(strike - spot) / spot * 100
                if moneyness_pct < 5.0:
                    severity = "ALERT" if abs(skew) > 10 else "WARN"
                    direction = "PUT SKEW" if skew > 0 else "CALL SKEW"
                    results.append(ScanResult(
                        scan_type="SKEW_OUTLIER",
                        instrument=instrument,
                        strike=strike,
                        value=skew,
                        description=(f"{int(strike)} {direction}: "
                                     f"CE={ce_iv:.1f}% PE={pe_iv:.1f}% skew={skew:+.1f}%"),
                        severity=severity,
                    ))

        results.sort(key=lambda r: abs(r.value), reverse=True)
        return results[:3]

    def scan_max_pain_deviation(self, chain: List[dict], spot: float,
                                instrument: str = "") -> List[ScanResult]:
        """Alert if spot price deviates significantly from max pain."""
        if not chain or spot <= 0:
            return []

        max_pain = _compute_max_pain_quick(chain)
        if max_pain <= 0:
            return []

        deviation_pct = (spot - max_pain) / max_pain * 100

        results = []
        if abs(deviation_pct) > self.MAX_PAIN_DEVIATION_PCT:
            direction = "above" if deviation_pct > 0 else "below"
            severity = "ALERT" if abs(deviation_pct) > 2.0 else "WARN"
            results.append(ScanResult(
                scan_type="MAX_PAIN_DEVIATION",
                instrument=instrument,
                value=deviation_pct,
                description=(f"Spot {spot:,.0f} is {abs(deviation_pct):.1f}% {direction} "
                             f"max pain {max_pain:,.0f} — reversion pressure expected"),
                severity=severity,
            ))
        return results

    def scan_liquidity(self, chain: List[dict],
                       instrument: str = "") -> List[ScanResult]:
        """Detect illiquid options (wide spreads) to avoid."""
        results = []
        for row in chain:
            bid = float(row.get("bid", 0))
            ask = float(row.get("ask", 0))
            ltp = float(row.get("ltp", 0))
            strike = float(row.get("strike", 0))
            right = str(row.get("right", ""))

            if bid <= 0 or ask <= 0 or ltp <= 0:
                continue

            spread_pct = (ask - bid) / ltp * 100 if ltp > 0 else 0

            if spread_pct > 15:  # >15% spread = very illiquid
                results.append(ScanResult(
                    scan_type="ILLIQUID",
                    instrument=instrument,
                    strike=strike,
                    right=right,
                    value=spread_pct,
                    description=(f"{int(strike)} {right} spread={spread_pct:.0f}% "
                                 f"(bid={bid:.1f} ask={ask:.1f}) — avoid!"),
                    severity="WARN",
                ))

        results.sort(key=lambda r: r.value, reverse=True)
        return results[:3]

    def oi_change_analysis(self, current_chain: List[dict],
                           prev_chain: List[dict],
                           instrument: str = "") -> List[ScanResult]:
        """
        Compare OI between two chain snapshots.
        Large OI buildup = new positions; large OI drop = unwinding.
        """
        results = []
        if not current_chain or not prev_chain:
            return results

        prev_map: Dict[Tuple, float] = {}
        for row in prev_chain:
            key = (float(row.get("strike", 0)), str(row.get("right", "")))
            prev_map[key] = float(row.get("oi", 0))

        for row in current_chain:
            strike = float(row.get("strike", 0))
            right = str(row.get("right", ""))
            curr_oi = float(row.get("oi", 0))
            key = (strike, right)
            prev_oi = prev_map.get(key, curr_oi)

            if prev_oi <= 0:
                continue

            change = curr_oi - prev_oi
            change_pct = change / prev_oi * 100

            if abs(change_pct) > 20 and abs(change) > 5000:
                direction = "BUILDUP" if change > 0 else "UNWINDING"
                severity = "ALERT" if abs(change_pct) > 50 else "WARN"
                results.append(ScanResult(
                    scan_type=f"OI_{direction}",
                    instrument=instrument,
                    strike=strike,
                    right=right,
                    value=change_pct,
                    description=(f"{int(strike)} {right} OI {direction}: "
                                 f"{prev_oi:,.0f} → {curr_oi:,.0f} ({change_pct:+.0f}%)"),
                    severity=severity,
                ))

        results.sort(key=lambda r: abs(r.value), reverse=True)
        return results[:5]


# ────────────────────────────────────────────────────────────────
# Helper
# ────────────────────────────────────────────────────────────────

def _compute_max_pain_quick(chain: List[dict]) -> float:
    """Quick max pain calculation."""
    strikes = sorted(set(float(r["strike"]) for r in chain if float(r.get("strike", 0)) > 0))
    if not strikes:
        return 0.0

    min_pain = float("inf")
    pain_strike = strikes[0]

    for test_strike in strikes:
        total_pain = 0.0
        for row in chain:
            k = float(row.get("strike", 0))
            oi = float(row.get("oi", 0))
            right = str(row.get("right", "")).upper()

            if oi <= 0 or k <= 0:
                continue

            if right == "CALL":
                pain = max(0.0, test_strike - k) * oi
            else:
                pain = max(0.0, k - test_strike) * oi

            total_pain += pain

        if total_pain < min_pain:
            min_pain = total_pain
            pain_strike = test_strike

    return pain_strike
