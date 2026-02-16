# ═══════════════════════════════════════════════════════════════
# FILE: greeks_engine.py
# ═══════════════════════════════════════════════════════════════
"""
Local Black-Scholes engine with Newton-Raphson IV solver.
Corrected: time-to-expiry now accounts for IST market close at 15:30.
"""

import math
from datetime import datetime
from scipy.stats import norm
from models import Greeks, OptionRight
from config import Config


_CALENDAR_DAYS = 365.0


class BlackScholes:

    @staticmethod
    def _d1d2(S: float, K: float, T: float, r: float, sigma: float):
        if T <= 1e-10 or sigma <= 1e-10:
            return 0.0, 0.0
        sqrt_T = math.sqrt(T)
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T
        return d1, d2

    @classmethod
    def price(cls, S, K, T, r, sigma, right: OptionRight) -> float:
        if T <= 1e-10:
            if right == OptionRight.CALL:
                return max(S - K, 0.0)
            return max(K - S, 0.0)
        d1, d2 = cls._d1d2(S, K, T, r, sigma)
        if right == OptionRight.CALL:
            return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
        return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    @classmethod
    def delta(cls, S, K, T, r, sigma, right: OptionRight) -> float:
        if T <= 1e-10 or sigma <= 1e-10:
            if right == OptionRight.CALL:
                return 1.0 if S > K else (0.5 if abs(S - K) < 0.01 else 0.0)
            return -1.0 if S < K else (-0.5 if abs(S - K) < 0.01 else 0.0)
        d1, _ = cls._d1d2(S, K, T, r, sigma)
        if right == OptionRight.CALL:
            return norm.cdf(d1)
        return norm.cdf(d1) - 1.0

    @classmethod
    def gamma(cls, S, K, T, r, sigma) -> float:
        if T <= 1e-10 or sigma <= 1e-10:
            return 0.0
        d1, _ = cls._d1d2(S, K, T, r, sigma)
        return norm.pdf(d1) / (S * sigma * math.sqrt(T))

    @classmethod
    def theta(cls, S, K, T, r, sigma, right: OptionRight) -> float:
        if T <= 1e-10 or sigma <= 1e-10:
            return 0.0
        d1, d2 = cls._d1d2(S, K, T, r, sigma)
        term1 = -(S * norm.pdf(d1) * sigma) / (2.0 * math.sqrt(T))
        if right == OptionRight.CALL:
            return (term1 - r * K * math.exp(-r * T) * norm.cdf(d2)) / _CALENDAR_DAYS
        return (term1 + r * K * math.exp(-r * T) * norm.cdf(-d2)) / _CALENDAR_DAYS

    @classmethod
    def vega(cls, S, K, T, r, sigma) -> float:
        if T <= 1e-10 or sigma <= 1e-10:
            return 0.0
        d1, _ = cls._d1d2(S, K, T, r, sigma)
        return S * norm.pdf(d1) * math.sqrt(T) / 100.0

    @classmethod
    def greeks(cls, S, K, T, r, sigma, right: OptionRight) -> Greeks:
        return Greeks(
            delta=cls.delta(S, K, T, r, sigma, right),
            gamma=cls.gamma(S, K, T, r, sigma),
            theta=cls.theta(S, K, T, r, sigma, right),
            vega=cls.vega(S, K, T, r, sigma),
            iv=sigma,
        )

    @classmethod
    def implied_vol(
        cls,
        market_price: float,
        S: float,
        K: float,
        T: float,
        r: float,
        right: OptionRight,
        initial_guess: float = 0.20,
        tol: float = 1e-6,
        max_iter: int = 80,
    ) -> float:
        if market_price <= 0.01 or T <= 1e-10:
            return initial_guess

        # Intrinsic check
        if right == OptionRight.CALL:
            intrinsic = max(S - K, 0.0)
        else:
            intrinsic = max(K - S, 0.0)
        if market_price < intrinsic:
            market_price = intrinsic + 0.01

        sigma = initial_guess
        for _ in range(max_iter):
            bs_price = cls.price(S, K, T, r, sigma, right)
            diff = bs_price - market_price
            if abs(diff) < tol:
                return max(sigma, 0.001)
            v = cls.vega(S, K, T, r, sigma) * 100.0
            if abs(v) < 1e-12:
                break
            sigma -= diff / v
            sigma = max(0.001, min(sigma, 5.0))

        # Brent fallback
        try:
            from scipy.optimize import brentq
            def obj(s):
                return cls.price(S, K, T, r, s, right) - market_price
            sigma = brentq(obj, 0.01, 5.0, xtol=tol)
        except Exception:
            sigma = initial_guess
        return max(sigma, 0.001)


def compute_time_to_expiry(expiry_date_str: str) -> float:
    """
    Compute T in years. Uses IST 15:30 as expiry time on the expiry date.
    This fixes the theta calculation error from using midnight.
    """
    if not expiry_date_str:
        return 1.0 / _CALENDAR_DAYS

    # Parse the date portion
    date_str = expiry_date_str[:10]
    for fmt in ["%Y-%m-%d", "%d-%b-%Y"]:
        try:
            expiry = datetime.strptime(date_str, fmt)
            break
        except ValueError:
            continue
    else:
        return 1.0 / _CALENDAR_DAYS

    # Set expiry to 15:30 IST on expiry day
    expiry = expiry.replace(hour=15, minute=30, second=0, microsecond=0)
    now = datetime.now()
    diff_seconds = (expiry - now).total_seconds()
    if diff_seconds <= 0:
        return 1e-6  # essentially expired
    return diff_seconds / (_CALENDAR_DAYS * 24 * 3600)
