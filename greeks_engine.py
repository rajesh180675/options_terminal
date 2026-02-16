# ═══════════════════════════════════════════════════════════════
# FILE: greeks_engine.py
# ═══════════════════════════════════════════════════════════════
"""
Self-contained Black-Scholes engine.
Computes Delta, Gamma, Theta, Vega, and solves for Implied Volatility.
All math is local — zero external API calls.
"""

import math
from scipy.stats import norm
from models import Greeks, OptionRight
from config import Config
from utils import LOG


_SQRT_2PI = math.sqrt(2.0 * math.pi)
_TRADING_DAYS = 365.0  # calendar days for Indian markets (convention)


class BlackScholes:
    """Vectorisable Black-Scholes calculator."""

    @staticmethod
    def _d1d2(
        S: float, K: float, T: float, r: float, sigma: float
    ) -> tuple[float, float]:
        if T <= 0 or sigma <= 0:
            return 0.0, 0.0
        sqrt_T = math.sqrt(T)
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T
        return d1, d2

    @classmethod
    def price(
        cls,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        right: OptionRight,
    ) -> float:
        if T <= 0:
            if right == OptionRight.CALL:
                return max(S - K, 0.0)
            return max(K - S, 0.0)
        d1, d2 = cls._d1d2(S, K, T, r, sigma)
        if right == OptionRight.CALL:
            return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
        return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    @classmethod
    def delta(cls, S, K, T, r, sigma, right: OptionRight) -> float:
        if T <= 0 or sigma <= 0:
            if right == OptionRight.CALL:
                return 1.0 if S > K else 0.0
            return -1.0 if S < K else 0.0
        d1, _ = cls._d1d2(S, K, T, r, sigma)
        if right == OptionRight.CALL:
            return norm.cdf(d1)
        return norm.cdf(d1) - 1.0

    @classmethod
    def gamma(cls, S, K, T, r, sigma) -> float:
        if T <= 0 or sigma <= 0:
            return 0.0
        d1, _ = cls._d1d2(S, K, T, r, sigma)
        return norm.pdf(d1) / (S * sigma * math.sqrt(T))

    @classmethod
    def theta(cls, S, K, T, r, sigma, right: OptionRight) -> float:
        if T <= 0 or sigma <= 0:
            return 0.0
        d1, d2 = cls._d1d2(S, K, T, r, sigma)
        common = -(S * norm.pdf(d1) * sigma) / (2.0 * math.sqrt(T))
        if right == OptionRight.CALL:
            return (common - r * K * math.exp(-r * T) * norm.cdf(d2)) / _TRADING_DAYS
        return (common + r * K * math.exp(-r * T) * norm.cdf(-d2)) / _TRADING_DAYS

    @classmethod
    def vega(cls, S, K, T, r, sigma) -> float:
        if T <= 0 or sigma <= 0:
            return 0.0
        d1, _ = cls._d1d2(S, K, T, r, sigma)
        return S * norm.pdf(d1) * math.sqrt(T) / 100.0  # per 1% move

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
        max_iter: int = 100,
    ) -> float:
        """Newton-Raphson IV solver with Brent fallback."""
        if market_price <= 0 or T <= 0:
            return initial_guess

        sigma = initial_guess
        for _ in range(max_iter):
            bs_price = cls.price(S, K, T, r, sigma, right)
            diff = bs_price - market_price
            v = cls.vega(S, K, T, r, sigma) * 100.0  # undo the /100
            if abs(diff) < tol:
                return max(sigma, 0.001)
            if abs(v) < 1e-12:
                break
            sigma -= diff / v
            sigma = max(sigma, 0.001)
            sigma = min(sigma, 5.0)

        # Brent fallback between 1% and 500% vol
        try:
            from scipy.optimize import brentq

            def objective(s):
                return cls.price(S, K, T, r, s, right) - market_price

            sigma = brentq(objective, 0.01, 5.0, xtol=tol)
        except Exception:
            sigma = initial_guess
        return max(sigma, 0.001)


def compute_time_to_expiry(expiry_date_str: str) -> float:
    """Return T in years from now to expiry (calendar-day basis)."""
    from datetime import datetime

    try:
        # Try ISO format first: "2024-01-25T06:00:00.000Z"
        expiry = datetime.strptime(expiry_date_str[:10], "%Y-%m-%d")
    except ValueError:
        try:
            # Try "25-Jan-2024"
            expiry = datetime.strptime(expiry_date_str[:11].strip(), "%d-%b-%Y")
        except ValueError:
            LOG.warning(f"Cannot parse expiry date: {expiry_date_str}")
            return 1.0 / _TRADING_DAYS  # default 1 day

    now = datetime.now()
    diff = (expiry - now).total_seconds()
    if diff <= 0:
        return 0.0001  # avoid zero
    return diff / (365.25 * 24 * 3600)
