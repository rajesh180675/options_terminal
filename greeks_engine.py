# ═══════════════════════════════════════════════════════════════
# FILE: greeks_engine.py
# ═══════════════════════════════════════════════════════════════
"""
Local Black-Scholes + Newton-Raphson IV solver.
Time-to-expiry uses 15:30 IST (market close on expiry day).
"""

import math
from datetime import datetime
from scipy.stats import norm
from models import Greeks, OptionRight
from app_config import Config

_CAL = 365.0


class BlackScholes:

    @staticmethod
    def _d1d2(S, K, T, r, s):
        if T <= 1e-10 or s <= 1e-10:
            return 0.0, 0.0
        sqT = math.sqrt(T)
        d1 = (math.log(S / K) + (r + 0.5 * s * s) * T) / (s * sqT)
        return d1, d1 - s * sqT

    @classmethod
    def price(cls, S, K, T, r, s, right: OptionRight) -> float:
        if T <= 1e-10:
            return max(S - K, 0.0) if right == OptionRight.CALL else max(K - S, 0.0)
        d1, d2 = cls._d1d2(S, K, T, r, s)
        if right == OptionRight.CALL:
            return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
        return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    @classmethod
    def delta(cls, S, K, T, r, s, right: OptionRight) -> float:
        if T <= 1e-10 or s <= 1e-10:
            if right == OptionRight.CALL:
                return 1.0 if S > K else 0.0
            return -1.0 if S < K else 0.0
        d1, _ = cls._d1d2(S, K, T, r, s)
        return norm.cdf(d1) if right == OptionRight.CALL else norm.cdf(d1) - 1.0

    @classmethod
    def gamma(cls, S, K, T, r, s) -> float:
        if T <= 1e-10 or s <= 1e-10:
            return 0.0
        d1, _ = cls._d1d2(S, K, T, r, s)
        return norm.pdf(d1) / (S * s * math.sqrt(T))

    @classmethod
    def theta(cls, S, K, T, r, s, right: OptionRight) -> float:
        if T <= 1e-10 or s <= 1e-10:
            return 0.0
        d1, d2 = cls._d1d2(S, K, T, r, s)
        t1 = -(S * norm.pdf(d1) * s) / (2.0 * math.sqrt(T))
        if right == OptionRight.CALL:
            return (t1 - r * K * math.exp(-r * T) * norm.cdf(d2)) / _CAL
        return (t1 + r * K * math.exp(-r * T) * norm.cdf(-d2)) / _CAL

    @classmethod
    def vega(cls, S, K, T, r, s) -> float:
        if T <= 1e-10 or s <= 1e-10:
            return 0.0
        d1, _ = cls._d1d2(S, K, T, r, s)
        return S * norm.pdf(d1) * math.sqrt(T) / 100.0

    @classmethod
    def greeks(cls, S, K, T, r, s, right: OptionRight) -> Greeks:
        return Greeks(
            delta=cls.delta(S, K, T, r, s, right),
            gamma=cls.gamma(S, K, T, r, s),
            theta=cls.theta(S, K, T, r, s, right),
            vega=cls.vega(S, K, T, r, s),
            iv=s,
        )

    @classmethod
    def implied_vol(cls, mkt, S, K, T, r, right: OptionRight,
                    guess=0.20, tol=1e-6, maxiter=80) -> float:
        if mkt <= 0.01 or T <= 1e-10:
            return guess
        intr = max(S - K, 0) if right == OptionRight.CALL else max(K - S, 0)
        if mkt < intr:
            mkt = intr + 0.01
        s = guess
        for _ in range(maxiter):
            p = cls.price(S, K, T, r, s, right)
            diff = p - mkt
            if abs(diff) < tol:
                return max(s, 0.001)
            v = cls.vega(S, K, T, r, s) * 100.0
            if abs(v) < 1e-12:
                break
            s -= diff / v
            s = max(0.001, min(s, 5.0))
        try:
            from scipy.optimize import brentq
            s = brentq(lambda x: cls.price(S, K, T, r, x, right) - mkt,
                       0.01, 5.0, xtol=tol)
        except Exception:
            s = guess
        return max(s, 0.001)


def time_to_expiry(expiry_str: str) -> float:
    """T in years. Expiry = 15:30 IST on expiry day."""
    if not expiry_str:
        return 1.0 / _CAL
    ds = expiry_str[:10]
    for fmt in ["%Y-%m-%d", "%d-%b-%Y"]:
        try:
            exp = datetime.strptime(ds, fmt)
            break
        except ValueError:
            continue
    else:
        return 1.0 / _CAL
    exp = exp.replace(hour=15, minute=30)
    diff = (exp - datetime.now()).total_seconds()
    return max(diff / (_CAL * 24 * 3600), 1e-6)
