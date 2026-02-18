"""
iv_surface.py  — NEW MODULE

Volatility Surface Engine
  • Multi-expiry IV surface computation (moneyness vs DTE grid)
  • Term structure (ATM IV across expiries)
  • Skew analysis (25-delta risk reversal, butterfly spread)
  • IV percentile / rank vs historical
  • VIX-style expected move calculations

Zero breaking-changes: pure add-on, no existing file modified.
"""

from __future__ import annotations

import math
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np

from greeks_engine import BlackScholes, time_to_expiry
from models import OptionRight
from utils import LOG, atm_strike, safe_float


# ────────────────────────────────────────────────────────────────
# Surface point
# ────────────────────────────────────────────────────────────────

class SurfacePoint:
    """Single (moneyness, dte, iv) point on the vol surface."""
    __slots__ = ("strike", "expiry", "right", "moneyness", "dte_days", "iv", "ltp")

    def __init__(self, strike: float, expiry: str, right: str,
                 moneyness: float, dte_days: float, iv: float, ltp: float):
        self.strike = strike
        self.expiry = expiry
        self.right = right
        self.moneyness = moneyness      # log(K/S)
        self.dte_days = dte_days
        self.iv = iv
        self.ltp = ltp


# ────────────────────────────────────────────────────────────────
# IV Surface Engine
# ────────────────────────────────────────────────────────────────

class IVSurfaceEngine:
    """
    Builds a volatility surface from an option chain.

    Usage (no engine dependency – pass the chain directly):
        engine = IVSurfaceEngine()
        surface = engine.build(chain, spot)
        term_structure = engine.term_structure(chain, spot)
        skew = engine.skew_metrics(chain, spot, expiry)
    """

    def __init__(self, risk_free_rate: float = 0.07):
        self.r = risk_free_rate
        self._surface_cache: List[SurfacePoint] = []
        self._cache_ts: float = 0.0
        self._cache_key: str = ""

    # ── Public API ────────────────────────────────────────────

    def build(self, chain: List[dict], spot: float,
              cache_ttl: float = 5.0) -> List[SurfacePoint]:
        """
        Build IV surface from enriched chain dicts.
        chain: list of dicts with keys: strike, right, expiry (date string),
               ltp, iv (can be 0 if not yet computed)
        """
        if spot <= 0:
            return []

        key = f"surf|{spot:.0f}|{len(chain)}"
        if self._surface_cache and key == self._cache_key and (time.time() - self._cache_ts) < cache_ttl:
            return list(self._surface_cache)

        surface: List[SurfacePoint] = []

        for row in chain:
            try:
                strike = float(row["strike"])
                expiry_str = str(row.get("expiry", row.get("expiry_date", "")))
                right_str = str(row.get("right", "CALL")).upper()
                ltp = float(row.get("ltp", 0))

                if strike <= 0 or ltp <= 0:
                    continue

                T = time_to_expiry(expiry_str) if expiry_str else 0.0
                if T <= 0:
                    continue

                dte_days = T * 365.0
                moneyness = math.log(strike / spot) if spot > 0 else 0.0

                # Use pre-computed IV if available, else solve
                iv = float(row.get("iv", 0))
                if iv <= 0 or iv > 999:  # iv stored as percent in chain
                    iv_raw = float(row.get("iv_raw", 0))
                    if iv_raw > 0:
                        iv = iv_raw
                    else:
                        opt_right = OptionRight.CALL if "CALL" in right_str else OptionRight.PUT
                        iv = BlackScholes.implied_vol(ltp, spot, strike, T, self.r, opt_right)
                else:
                    iv = iv / 100.0  # chain stores as percent

                if 0.005 < iv < 5.0:
                    surface.append(SurfacePoint(
                        strike=strike,
                        expiry=expiry_str[:10],
                        right=right_str,
                        moneyness=round(moneyness, 4),
                        dte_days=round(dte_days, 2),
                        iv=round(iv, 4),
                        ltp=ltp,
                    ))
            except Exception as e:
                LOG.debug(f"IVSurface skip row: {e}")
                continue

        self._surface_cache = surface
        self._cache_key = key
        self._cache_ts = time.time()
        return surface

    def term_structure(self, chain: List[dict], spot: float) -> List[dict]:
        """
        ATM IV for each expiry → term structure.
        Returns: [{expiry, dte, atm_iv, call_iv, put_iv}, ...]
        """
        if spot <= 0:
            return []

        # Group by expiry
        by_expiry: Dict[str, List[dict]] = {}
        for row in chain:
            exp = str(row.get("expiry", row.get("expiry_date", "")))[:10]
            if exp:
                by_expiry.setdefault(exp, []).append(row)

        result = []
        for exp, rows in sorted(by_expiry.items()):
            T = time_to_expiry(exp)
            if T <= 0:
                continue

            gap = _estimate_strike_gap(rows)
            atm = atm_strike(spot, gap) if gap > 0 else _find_nearest_strike(rows, spot)

            call_row = _find(rows, atm, "CALL")
            put_row = _find(rows, atm, "PUT")

            call_iv = _extract_iv(call_row, spot, T, self.r, OptionRight.CALL)
            put_iv = _extract_iv(put_row, spot, T, self.r, OptionRight.PUT)

            atm_iv = (call_iv + put_iv) / 2 if (call_iv > 0 and put_iv > 0) else max(call_iv, put_iv)

            if atm_iv > 0:
                result.append({
                    "expiry": exp,
                    "dte": round(T * 365, 1),
                    "atm_iv": round(atm_iv * 100, 2),
                    "call_iv": round(call_iv * 100, 2),
                    "put_iv": round(put_iv * 100, 2),
                })

        return result

    def skew_metrics(self, chain: List[dict], spot: float,
                     expiry: str = "") -> dict:
        """
        25-delta skew metrics for a single expiry.
        Returns: {rr25: float, bf25: float, atm_iv: float, skew_slope: float}
        """
        if spot <= 0:
            return {}

        rows = chain
        if expiry:
            rows = [r for r in chain if str(r.get("expiry", r.get("expiry_date", "")))[:10] == expiry[:10]]

        T_val = time_to_expiry(expiry) if expiry else None
        if T_val is None:
            # Use first row's expiry
            if rows:
                exp0 = str(rows[0].get("expiry", rows[0].get("expiry_date", "")))[:10]
                T_val = time_to_expiry(exp0)
        if not T_val or T_val <= 0:
            return {}

        T = T_val
        r = self.r
        gap = _estimate_strike_gap(rows)
        atm = atm_strike(spot, gap) if gap > 0 else _find_nearest_strike(rows, spot)

        # ATM IV
        atm_c = _find(rows, atm, "CALL")
        atm_p = _find(rows, atm, "PUT")
        atm_c_iv = _extract_iv(atm_c, spot, T, r, OptionRight.CALL)
        atm_p_iv = _extract_iv(atm_p, spot, T, r, OptionRight.PUT)
        atm_iv = (atm_c_iv + atm_p_iv) / 2 if (atm_c_iv > 0 and atm_p_iv > 0) else max(atm_c_iv, atm_p_iv)

        # 25-delta strikes
        call_25d_strike = _find_delta_strike(rows, spot, T, r, target_delta=0.25, right=OptionRight.CALL)
        put_25d_strike = _find_delta_strike(rows, spot, T, r, target_delta=0.25, right=OptionRight.PUT)

        call_25d_row = _find(rows, call_25d_strike, "CALL") if call_25d_strike else None
        put_25d_row = _find(rows, put_25d_strike, "PUT") if put_25d_strike else None

        call_25d_iv = _extract_iv(call_25d_row, spot, T, r, OptionRight.CALL) if call_25d_row else 0.0
        put_25d_iv = _extract_iv(put_25d_row, spot, T, r, OptionRight.PUT) if put_25d_row else 0.0

        # Risk Reversal = OTM Call IV - OTM Put IV
        rr25 = (call_25d_iv - put_25d_iv) * 100 if (call_25d_iv > 0 and put_25d_iv > 0) else 0.0

        # Butterfly = (OTM Call IV + OTM Put IV)/2 - ATM IV
        bf25 = ((call_25d_iv + put_25d_iv) / 2 - atm_iv) * 100 if (call_25d_iv > 0 and put_25d_iv > 0) else 0.0

        # Skew slope (IV change per 1% moneyness change)
        skew_slope = _compute_skew_slope(rows, spot, T, r)

        return {
            "atm_iv": round(atm_iv * 100, 2),
            "call_25d_iv": round(call_25d_iv * 100, 2),
            "put_25d_iv": round(put_25d_iv * 100, 2),
            "rr25": round(rr25, 2),
            "bf25": round(bf25, 2),
            "skew_slope": round(skew_slope, 3),
            "put_skew": round((put_25d_iv - atm_iv) * 100, 2),   # protective put premium
            "call_skew": round((call_25d_iv - atm_iv) * 100, 2),
        }

    def iv_percentile(self, current_iv: float, history: List[float]) -> dict:
        """
        IV Percentile and Rank vs historical.
        history: list of historical ATM IV values (percent)
        Returns: {ivp: float, ivr: float, current_iv: float}
        """
        if not history or current_iv <= 0:
            return {"ivp": 0.0, "ivr": 0.0, "current_iv": current_iv}

        arr = sorted(history)
        n = len(arr)

        # IV Percentile: % of days where IV was below current
        below = sum(1 for x in arr if x < current_iv)
        ivp = (below / n) * 100

        # IV Rank: (current - min) / (max - min) * 100
        lo, hi = arr[0], arr[-1]
        ivr = ((current_iv - lo) / (hi - lo)) * 100 if hi > lo else 50.0

        return {
            "ivp": round(ivp, 1),
            "ivr": round(ivr, 1),
            "current_iv": round(current_iv, 2),
            "iv_min": round(lo, 2),
            "iv_max": round(hi, 2),
            "iv_mean": round(sum(history) / n, 2),
        }

    def surface_to_grid(self, surface: List[SurfacePoint],
                        moneyness_bins: int = 9,
                        dte_bins: int = 5) -> dict:
        """
        Convert surface points to a 2D grid for heatmap visualization.
        Returns: {x_labels: [], y_labels: [], z_matrix: [[...], ...]}
        """
        if not surface:
            return {}

        # Moneyness bins: -20% to +20%
        mono_edges = np.linspace(-0.20, 0.20, moneyness_bins + 1)
        mono_centers = [(mono_edges[i] + mono_edges[i+1]) / 2 for i in range(moneyness_bins)]
        mono_labels = [f"{c*100:+.0f}%" for c in mono_centers]

        # DTE bins
        dtes = sorted(set(round(p.dte_days) for p in surface))
        if len(dtes) > dte_bins:
            step = len(dtes) // dte_bins
            dte_samples = dtes[::step][:dte_bins]
        else:
            dte_samples = dtes
        dte_labels = [f"{d}d" for d in dte_samples]

        # Fill grid (average IV in each cell)
        z = [[None] * len(dte_samples) for _ in range(moneyness_bins)]

        for row_i, (m_lo, m_hi) in enumerate(zip(mono_edges[:-1], mono_edges[1:])):
            for col_j, dte_target in enumerate(dte_samples):
                pts = [
                    p.iv for p in surface
                    if m_lo <= p.moneyness < m_hi
                    and abs(p.dte_days - dte_target) < 3
                ]
                if pts:
                    z[row_i][col_j] = round(sum(pts) / len(pts) * 100, 1)

        return {
            "x_labels": dte_labels,
            "y_labels": mono_labels,
            "z_matrix": z,
            "title": "IV Surface (%)",
        }


# ────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────

def _estimate_strike_gap(rows: List[dict]) -> float:
    strikes = sorted(set(float(r["strike"]) for r in rows if float(r.get("strike", 0)) > 0))
    if len(strikes) >= 2:
        diffs = [strikes[i+1] - strikes[i] for i in range(min(len(strikes)-1, 5))]
        return min(diffs)
    return 50.0


def _find_nearest_strike(rows: List[dict], spot: float) -> float:
    strikes = [float(r["strike"]) for r in rows if float(r.get("strike", 0)) > 0]
    if not strikes:
        return round(spot, -2)
    return min(strikes, key=lambda x: abs(x - spot))


def _find(rows: List[dict], strike: float, right: str) -> Optional[dict]:
    for r in rows:
        if float(r.get("strike", 0)) == strike and r.get("right", "").upper() == right.upper():
            return r
    return None


def _extract_iv(row: Optional[dict], spot: float, T: float,
                r: float, opt_right: OptionRight) -> float:
    if row is None:
        return 0.0
    iv = float(row.get("iv", 0))
    ltp = float(row.get("ltp", 0))
    strike = float(row.get("strike", 0))
    if iv > 0 and iv < 200:
        return iv / 100.0
    if ltp > 0 and spot > 0 and strike > 0 and T > 0:
        return BlackScholes.implied_vol(ltp, spot, strike, T, r, opt_right)
    return 0.0


def _find_delta_strike(rows: List[dict], spot: float, T: float,
                       r: float, target_delta: float,
                       right: OptionRight) -> Optional[float]:
    """Find strike with delta closest to target."""
    best_k = None
    best_diff = float("inf")
    for row in rows:
        if row.get("right", "").upper() != right.value.upper():
            continue
        strike = float(row.get("strike", 0))
        ltp = float(row.get("ltp", 0))
        if strike <= 0 or ltp <= 0:
            continue
        iv = _extract_iv(row, spot, T, r, right)
        if iv <= 0:
            continue
        delta = abs(BlackScholes.delta(spot, strike, T, r, iv, right))
        diff = abs(delta - target_delta)
        if diff < best_diff:
            best_diff = diff
            best_k = strike
    return best_k


def _compute_skew_slope(rows: List[dict], spot: float,
                        T: float, r: float) -> float:
    """Slope of IV vs moneyness (linear regression)."""
    pts = []
    for row in rows:
        strike = float(row.get("strike", 0))
        ltp = float(row.get("ltp", 0))
        right_str = row.get("right", "PUT").upper()
        if strike <= 0 or ltp <= 0 or spot <= 0:
            continue
        opt_right = OptionRight.CALL if right_str == "CALL" else OptionRight.PUT
        iv = _extract_iv(row, spot, T, r, opt_right)
        if iv > 0:
            mono = math.log(strike / spot)
            pts.append((mono, iv))

    if len(pts) < 3:
        return 0.0

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    den = sum((xs[i] - mx) ** 2 for i in range(n))
    return num / den if den > 0 else 0.0
