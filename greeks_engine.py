"""
greeks_engine.py — Local Black-Scholes Greeks calculator with Newton-Raphson IV solver.
No external API calls for Greeks computation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


@dataclass
class Greeks:
    price: float  # theoretical BS price
    iv: float
    delta: float
    gamma: float
    theta: float
    vega: float


class BlackScholes:
    """Optimised Black-Scholes engine for European options on indices."""

    @staticmethod
    def d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
        if T <= 0 or sigma <= 0:
            return 0.0
        return (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))

    @staticmethod
    def d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
        if T <= 0 or sigma <= 0:
            return 0.0
        return BlackScholes.d1(S, K, T, r, sigma) - sigma * math.sqrt(T)

    @staticmethod
    def call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
        if T <= 0:
            return max(S - K, 0.0)
        d_1 = BlackScholes.d1(S, K, T, r, sigma)
        d_2 = BlackScholes.d2(S, K, T, r, sigma)
        return S * norm.cdf(d_1) - K * math.exp(-r * T) * norm.cdf(d_2)

    @staticmethod
    def put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
        if T <= 0:
            return max(K - S, 0.0)
        d_1 = BlackScholes.d1(S, K, T, r, sigma)
        d_2 = BlackScholes.d2(S, K, T, r, sigma)
        return K * math.exp(-r * T) * norm.cdf(-d_2) - S * norm.cdf(-d_1)

    @staticmethod
    def delta(S: float, K: float, T: float, r: float, sigma: float,
              right: str) -> float:
        if T <= 0 or sigma <= 0:
            if right == "call":
                return 1.0 if S > K else 0.0
            return -1.0 if S < K else 0.0
        d_1 = BlackScholes.d1(S, K, T, r, sigma)
        if right == "call":
            return norm.cdf(d_1)
        return norm.cdf(d_1) - 1.0

    @staticmethod
    def gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
        if T <= 0 or sigma <= 0 or S <= 0:
            return 0.0
        d_1 = BlackScholes.d1(S, K, T, r, sigma)
        return norm.pdf(d_1) / (S * sigma * math.sqrt(T))

    @staticmethod
    def theta(S: float, K: float, T: float, r: float, sigma: float,
              right: str) -> float:
        """Returns theta per calendar day (divide annual by 365)."""
        if T <= 0 or sigma <= 0:
            return 0.0
        d_1 = BlackScholes.d1(S, K, T, r, sigma)
        d_2 = BlackScholes.d2(S, K, T, r, sigma)
        sqrt_T = math.sqrt(T)
        term1 = -(S * norm.pdf(d_1) * sigma) / (2 * sqrt_T)
        if right == "call":
            term2 = -r * K * math.exp(-r * T) * norm.cdf(d_2)
        else:
            term2 = r * K * math.exp(-r * T) * norm.cdf(-d_2)
        return (term1 + term2) / 365.0

    @staticmethod
    def vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Returns vega for a 1% (0.01) change in IV."""
        if T <= 0 or sigma <= 0:
            return 0.0
        d_1 = BlackScholes.d1(S, K, T, r, sigma)
        return S * norm.pdf(d_1) * math.sqrt(T) * 0.01

    @staticmethod
    def implied_volatility(market_price: float, S: float, K: float, T: float,
                           r: float, right: str,
                           tol: float = 1e-6, max_iter: int = 100) -> float:
        """Newton-Raphson IV solver."""
        if T <= 0 or market_price <= 0:
            return 0.0

        intrinsic = max(S - K, 0.0) if right == "call" else max(K - S, 0.0)
        if market_price < intrinsic:
            return 0.001

        sigma = 0.25  # initial guess

        for _ in range(max_iter):
            if right == "call":
                price = BlackScholes.call_price(S, K, T, r, sigma)
            else:
                price = BlackScholes.put_price(S, K, T, r, sigma)

            d_1 = BlackScholes.d1(S, K, T, r, sigma)
            vega_val = S * norm.pdf(d_1) * math.sqrt(T)

            if vega_val < 1e-12:
                break

            diff = price - market_price
            sigma -= diff / vega_val

            if sigma <= 0.001:
                sigma = 0.001
            if sigma > 5.0:
                sigma = 5.0

            if abs(diff) < tol:
                break

        return sigma

    @staticmethod
    def compute_greeks(S: float, K: float, T: float, r: float,
                       market_price: float, right: str) -> Greeks:
        """Full Greeks computation: solves IV then computes all sensitivities."""
        iv = BlackScholes.implied_volatility(market_price, S, K, T, r, right)

        if right == "call":
            theo_price = BlackScholes.call_price(S, K, T, r, iv)
        else:
            theo_price = BlackScholes.put_price(S, K, T, r, iv)

        return Greeks(
            price=theo_price,
            iv=iv,
            delta=BlackScholes.delta(S, K, T, r, iv, right),
            gamma=BlackScholes.gamma(S, K, T, r, iv),
            theta=BlackScholes.theta(S, K, T, r, iv, right),
            vega=BlackScholes.vega(S, K, T, r, iv),
        )


class GreeksEngine:
    """
    Convenience wrapper that computes Greeks for an entire option chain
    given the spot price and risk-free rate.
    """

    def __init__(self, risk_free_rate: float = 0.07):
        self.r = risk_free_rate
        self.bs = BlackScholes

    def compute_for_strike(self, spot: float, strike: float,
                           tte_years: float, market_price: float,
                           right: str) -> Greeks:
        return self.bs.compute_greeks(spot, strike, tte_years, self.r,
                                      market_price, right)

    def find_strike_by_delta(self, spot: float, strikes: list[float],
                             tte_years: float, prices: dict[float, float],
                             target_delta: float, right: str) -> float:
        """
        Find the strike whose absolute delta is closest to target_delta.
        `prices` maps strike -> LTP for the given right.
        """
        best_strike = strikes[0]
        best_diff = float("inf")

        for strike in strikes:
            mkt_price = prices.get(strike, 0.0)
            if mkt_price <= 0:
                continue
            greeks = self.compute_for_strike(spot, strike, tte_years,
                                             mkt_price, right)
            diff = abs(abs(greeks.delta) - target_delta)
            if diff < best_diff:
                best_diff = diff
                best_strike = strike

        return best_strike

    @staticmethod
    def time_to_expiry_years(expiry_date_str: str) -> float:
        """
        Calculate time to expiry in years from an ISO date string.
        Minimum returned is 1 minute expressed in years.
        """
        from datetime import datetime, timezone
        try:
            if "T" in expiry_date_str:
                expiry = datetime.fromisoformat(
                    expiry_date_str.replace("Z", "+00:00")
                )
            else:
                expiry = datetime.strptime(expiry_date_str, "%Y-%m-%d")
                expiry = expiry.replace(
                    hour=15, minute=30,
                    tzinfo=timezone.utc
                )
            now = datetime.now(timezone.utc)
            diff = (expiry - now).total_seconds()
            return max(diff / (365.25 * 24 * 3600), 1 / (365.25 * 24 * 60))
        except Exception:
            return 1 / 365.25  # default 1 day
