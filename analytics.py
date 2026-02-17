# ═══════════════════════════════════════════════════════════════
# FILE: analytics.py  (NEW)
# ═══════════════════════════════════════════════════════════════
"""
Market analytics engine:
  • Max Pain from OI data
  • IV Percentile from historical realized vol
  • Expected Move from ATM straddle
  • Probability of Profit for active strategies
  • Put-Call Ratio
  • Volatility Skew data
  • What-If scenario analysis
  • Greeks P&L attribution
  • Payoff diagram generation
"""

import math
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from scipy.stats import norm

from models import Strategy, Leg, LegStatus, OrderSide, OptionRight, Greeks
from greeks_engine import BlackScholes, time_to_expiry
from app_config import Config
from utils import safe_float, LOG


# ── Max Pain ─────────────────────────────────────────────────

def calculate_max_pain(chain: List[dict]) -> Tuple[float, List[dict]]:
    """
    Max Pain = strike where total ITM value (pain to writers) is minimized.

    For each candidate settlement price K:
      Call pain = sum of CE_OI * max(0, K - CE_strike) for all calls
      Put pain  = sum of PE_OI * max(0, PE_strike - K) for all puts
      Total pain = call_pain + put_pain

    Returns: (max_pain_strike, pain_curve_data)
    """
    calls = [(c["strike"], c.get("oi", 0)) for c in chain if c["right"] == "CALL"]
    puts = [(c["strike"], c.get("oi", 0)) for c in chain if c["right"] == "PUT"]

    strikes = sorted(set(c["strike"] for c in chain))
    if not strikes:
        return 0, []

    pain_curve = []
    min_pain = float("inf")
    mp_strike = strikes[len(strikes) // 2]

    for K in strikes:
        call_pain = sum(oi * max(0, K - s) for s, oi in calls)
        put_pain = sum(oi * max(0, s - K) for s, oi in puts)
        total = call_pain + put_pain

        pain_curve.append({"strike": K, "call_pain": call_pain,
                           "put_pain": put_pain, "total": total})
        if total < min_pain:
            min_pain = total
            mp_strike = K

    return mp_strike, pain_curve


# ── PCR ──────────────────────────────────────────────────────

def calculate_pcr(chain: List[dict]) -> Dict[str, float]:
    """Put-Call Ratio from OI and volume."""
    total_ce_oi = sum(c.get("oi", 0) for c in chain if c["right"] == "CALL")
    total_pe_oi = sum(c.get("oi", 0) for c in chain if c["right"] == "PUT")
    total_ce_vol = sum(c.get("volume", 0) for c in chain if c["right"] == "CALL")
    total_pe_vol = sum(c.get("volume", 0) for c in chain if c["right"] == "PUT")

    return {
        "pcr_oi": total_pe_oi / max(total_ce_oi, 1),
        "pcr_vol": total_pe_vol / max(total_ce_vol, 1),
        "total_ce_oi": total_ce_oi,
        "total_pe_oi": total_pe_oi,
    }


# ── Expected Move ────────────────────────────────────────────

def calculate_expected_move(spot: float, atm_ce_price: float,
                            atm_pe_price: float, dte: float) -> Dict:
    """
    Expected Move = ATM_straddle_price * 0.85  (empirical)
    Also: EM = spot * IV * sqrt(DTE/365)
    """
    straddle_price = atm_ce_price + atm_pe_price
    em_empirical = straddle_price * 0.85
    upper = spot + em_empirical
    lower = spot - em_empirical

    # IV-based calculation
    if spot > 0 and straddle_price > 0 and dte > 0:
        implied_em_pct = straddle_price / spot * 100
    else:
        implied_em_pct = 0

    return {
        "expected_move": round(em_empirical, 2),
        "upper_range": round(upper, 2),
        "lower_range": round(lower, 2),
        "straddle_price": round(straddle_price, 2),
        "em_percent": round(implied_em_pct, 2),
    }


# ── IV Percentile ────────────────────────────────────────────

def calculate_iv_percentile(historical_closes: List[float],
                            current_iv: float,
                            window: int = 20) -> Dict:
    """
    Compute realized volatility from historical closes,
    then rank current IV against the realized vol distribution.

    IV Percentile = % of past RV readings below current IV
    IV Rank = (current - min) / (max - min)
    """
    if len(historical_closes) < window + 5:
        return {"iv_percentile": 50, "iv_rank": 0.5,
                "current_iv": current_iv, "rv_20d": 0, "data_points": 0}

    closes = np.array(historical_closes, dtype=float)
    log_returns = np.diff(np.log(closes))

    # Rolling realized vol
    rv_series = []
    for i in range(window, len(log_returns)):
        chunk = log_returns[i - window:i]
        rv = np.std(chunk) * math.sqrt(252)  # annualize
        rv_series.append(rv)

    if not rv_series:
        return {"iv_percentile": 50, "iv_rank": 0.5,
                "current_iv": current_iv, "rv_20d": 0, "data_points": 0}

    rv_arr = np.array(rv_series)
    current_rv = rv_arr[-1]
    percentile = float(np.sum(rv_arr < current_iv) / len(rv_arr) * 100)
    rv_min, rv_max = float(np.min(rv_arr)), float(np.max(rv_arr))
    rank = (current_iv - rv_min) / max(rv_max - rv_min, 0.001)

    return {
        "iv_percentile": round(percentile, 1),
        "iv_rank": round(min(max(rank, 0), 1), 3),
        "current_iv": round(current_iv * 100, 2),
        "rv_20d": round(current_rv * 100, 2),
        "rv_min": round(rv_min * 100, 2),
        "rv_max": round(rv_max * 100, 2),
        "data_points": len(rv_series),
    }


# ── Probability of Profit ────────────────────────────────────

def calculate_pop(strategy: Strategy, spot: float) -> float:
    """
    Probability of Profit at expiry using Black-Scholes.
    For short straddle/strangle: probability that spot stays
    between the lower and upper breakeven points.
    """
    if not strategy.legs or spot <= 0:
        return 0

    active = [l for l in strategy.legs if l.status in
              (LegStatus.ACTIVE, LegStatus.ENTERING)]
    if not active:
        return 0

    # Find breakeven points
    total_premium = sum(l.entry_price * l.quantity for l in active
                        if l.side == OrderSide.SELL)
    total_premium -= sum(l.entry_price * l.quantity for l in active
                         if l.side == OrderSide.BUY)

    premium_per_lot = total_premium / max(
        sum(l.quantity for l in active if l.side == OrderSide.SELL), 1
    )

    # Identify strikes
    ce_strikes = [l.strike_price for l in active
                  if l.right == OptionRight.CALL and l.side == OrderSide.SELL]
    pe_strikes = [l.strike_price for l in active
                  if l.right == OptionRight.PUT and l.side == OrderSide.SELL]

    if not ce_strikes or not pe_strikes:
        return 0

    upper_be = max(ce_strikes) + premium_per_lot
    lower_be = min(pe_strikes) - premium_per_lot

    # Get IV and time
    exp = active[0].expiry_date
    T = time_to_expiry(exp)
    avg_iv = np.mean([l.greeks.iv for l in active if l.greeks.iv > 0])
    if avg_iv <= 0:
        avg_iv = 0.15

    if T <= 0:
        return 0

    r = Config.RISK_FREE_RATE
    d_upper = (math.log(upper_be / spot) - (r - 0.5 * avg_iv ** 2) * T) / (
        avg_iv * math.sqrt(T))
    d_lower = (math.log(lower_be / spot) - (r - 0.5 * avg_iv ** 2) * T) / (
        avg_iv * math.sqrt(T))

    pop = float(norm.cdf(d_upper) - norm.cdf(d_lower))
    return round(pop * 100, 1)


# ── Volatility Skew ──────────────────────────────────────────

def calculate_skew(chain: List[dict], spot: float) -> List[dict]:
    """Extract IV vs strike for skew visualization."""
    skew_data = []
    for c in chain:
        if c.get("iv", 0) > 0:
            moneyness = c["strike"] / spot if spot > 0 else 1
            skew_data.append({
                "strike": c["strike"],
                "iv": c["iv"],
                "right": c["right"],
                "moneyness": round(moneyness, 4),
            })
    return sorted(skew_data, key=lambda x: x["strike"])


# ── Payoff Diagram ───────────────────────────────────────────

def generate_payoff(strategy: Strategy, spot: float,
                    num_points: int = 80) -> List[dict]:
    """
    Generate (spot_at_expiry, pnl) data for charting.
    Works for ANY combination of legs (straddle, strangle,
    iron condor, iron butterfly, custom).
    """
    active = [l for l in strategy.legs if l.status in
              (LegStatus.ACTIVE, LegStatus.ENTERING)]
    if not active or spot <= 0:
        return []

    gap = Config.strike_gap(strategy.stock_code)
    strikes = [l.strike_price for l in active]
    center = sum(strikes) / len(strikes)
    spread = max(center * 0.08, gap * 10)
    low = center - spread
    high = center + spread

    points = np.linspace(low, high, num_points)
    payoff = []

    for S in points:
        pnl = 0.0
        for leg in active:
            if leg.right == OptionRight.CALL:
                intrinsic = max(0, S - leg.strike_price)
            else:
                intrinsic = max(0, leg.strike_price - S)

            if leg.side == OrderSide.SELL:
                leg_pnl = (leg.entry_price - intrinsic) * leg.quantity
            else:
                leg_pnl = (intrinsic - leg.entry_price) * leg.quantity
            pnl += leg_pnl

        payoff.append({"spot": round(S, 1), "pnl": round(pnl, 2)})

    return payoff


# ── What-If Scenario Analysis ────────────────────────────────

def what_if_analysis(strategy: Strategy, spot: float,
                     scenarios_pct: List[float] = None) -> List[dict]:
    """
    Show P&L at CURRENT time for various spot price scenarios.
    Unlike payoff (at expiry), this uses live Greeks.
    """
    if scenarios_pct is None:
        scenarios_pct = [-5, -4, -3, -2, -1, -0.5, 0, 0.5, 1, 2, 3, 4, 5]

    active = [l for l in strategy.legs if l.status == LegStatus.ACTIVE]
    if not active or spot <= 0:
        return []

    r = Config.RISK_FREE_RATE
    results = []

    for pct in scenarios_pct:
        scenario_spot = spot * (1 + pct / 100)
        total_pnl = 0

        for leg in active:
            T = time_to_expiry(leg.expiry_date)
            iv = leg.greeks.iv if leg.greeks.iv > 0 else 0.15

            new_price = BlackScholes.price(
                scenario_spot, leg.strike_price, T, r, iv, leg.right
            )
            new_price = max(new_price, 0.01)

            if leg.side == OrderSide.SELL:
                pnl = (leg.entry_price - new_price) * leg.quantity
            else:
                pnl = (new_price - leg.entry_price) * leg.quantity
            total_pnl += pnl

        results.append({
            "scenario": f"{pct:+.1f}%",
            "spot": round(scenario_spot, 1),
            "pnl": round(total_pnl, 2),
        })

    return results


# ── Greeks P&L Attribution ───────────────────────────────────

def greeks_pnl_attribution(strategy: Strategy, spot_change: float,
                            iv_change: float = 0, days: float = 1
                            ) -> Dict[str, float]:
    """
    Decompose strategy P&L into Greek components:
      Delta P&L  = Σ (leg_delta * sign * qty) * spot_change
      Gamma P&L  = 0.5 * Σ (leg_gamma * sign * qty) * spot_change²
      Theta P&L  = Σ (leg_theta * sign * qty) * days
      Vega P&L   = Σ (leg_vega * sign * qty) * iv_change
    """
    ng = strategy.net_greeks
    delta_pnl = ng.delta * spot_change
    gamma_pnl = 0.5 * ng.gamma * spot_change ** 2
    theta_pnl = ng.theta * days
    vega_pnl = ng.vega * iv_change * 100

    return {
        "delta_pnl": round(delta_pnl, 2),
        "gamma_pnl": round(gamma_pnl, 2),
        "theta_pnl": round(theta_pnl, 2),
        "vega_pnl": round(vega_pnl, 2),
        "total_attributed": round(delta_pnl + gamma_pnl + theta_pnl + vega_pnl, 2),
    }
