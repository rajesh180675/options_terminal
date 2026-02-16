# ═══════════════════════════════════════════════════════════════
# FILE: strategy_engine.py
# ═══════════════════════════════════════════════════════════════
"""
Strategy deployer: straddle, strangle, and manual limit sells.

FIXES:
  1. Option chain response parsing uses same field names for mock & live
  2. Manual single-leg sell method added
  3. Chain caching to avoid redundant API calls
"""

import time
from datetime import datetime
from typing import Optional, Tuple, List

from models import (
    Strategy, Leg, StrategyType, StrategyStatus,
    LegStatus, OrderSide, OptionRight, Greeks,
)
from greeks_engine import BlackScholes, compute_time_to_expiry
from connection_manager import SessionManager
from order_manager import OrderManager
from database import Database
from shared_state import SharedState
from config import Config
from utils import LOG, atm_strike, breeze_expiry_format, next_weekly_expiry, safe_float


class StrategyEngine:

    def __init__(self, session: SessionManager, order_mgr: OrderManager,
                 db: Database, state: SharedState):
        self.session = session
        self.order_mgr = order_mgr
        self.db = db
        self.state = state

    # ── Short Straddle ───────────────────────────────────────

    def deploy_short_straddle(
        self,
        stock_code: str = "NIFTY",
        expiry_date: str = "",
        lots: int = 1,
        sl_percentage: float = 0.0,
    ) -> Optional[Strategy]:
        if not expiry_date:
            expiry_date = breeze_expiry_format(next_weekly_expiry(stock_code))

        spot = self.session.get_spot_price(stock_code)
        if spot <= 0:
            self.state.add_log("ERROR", "Strategy", "Cannot get spot price")
            return None

        gap = Config.strike_gap(stock_code)
        atm = atm_strike(spot, gap)
        quantity = Config.lot_size(stock_code) * lots
        sl_pct = sl_percentage if sl_percentage > 0 else Config.SL_PERCENTAGE

        self.state.add_log("INFO", "Strategy",
                           f"SHORT STRADDLE: {stock_code} ATM={int(atm)} "
                           f"qty={quantity} exp={expiry_date[:10]}")

        strategy = Strategy(
            strategy_type=StrategyType.SHORT_STRADDLE,
            stock_code=stock_code,
            target_delta=0.5,
            status=StrategyStatus.DEPLOYING,
        )

        ce_leg = self._make_leg(strategy.strategy_id, stock_code, atm,
                                 OptionRight.CALL, expiry_date, quantity, sl_pct)
        pe_leg = self._make_leg(strategy.strategy_id, stock_code, atm,
                                 OptionRight.PUT, expiry_date, quantity, sl_pct)

        strategy.legs = [ce_leg, pe_leg]
        self.db.save_strategy(strategy)
        self.db.save_leg(ce_leg)
        self.db.save_leg(pe_leg)

        # Subscribe feeds
        self.session.subscribe_option(stock_code, atm, "call", expiry_date)
        self.session.subscribe_option(stock_code, atm, "put", expiry_date)
        time.sleep(0.5)

        ok_ce = self.order_mgr.execute_sell_with_chase(ce_leg)
        ok_pe = self.order_mgr.execute_sell_with_chase(pe_leg)

        strategy.status = (
            StrategyStatus.ACTIVE if (ok_ce and ok_pe)
            else StrategyStatus.ACTIVE if (ok_ce or ok_pe)
            else StrategyStatus.ERROR
        )
        self.db.save_strategy(strategy)
        return strategy

    # ── Short Strangle ───────────────────────────────────────

    def deploy_short_strangle(
        self,
        stock_code: str = "NIFTY",
        target_delta: float = 0.15,
        expiry_date: str = "",
        lots: int = 1,
        sl_percentage: float = 0.0,
    ) -> Optional[Strategy]:
        if not expiry_date:
            expiry_date = breeze_expiry_format(next_weekly_expiry(stock_code))

        spot = self.session.get_spot_price(stock_code)
        if spot <= 0:
            self.state.add_log("ERROR", "Strategy", "Cannot get spot price")
            return None

        self.state.add_log("INFO", "Strategy",
                           f"Scanning for {target_delta:.2f}Δ strangle…")

        ce_strike, pe_strike = self._find_delta_strikes(
            stock_code, spot, expiry_date, target_delta
        )
        if not ce_strike or not pe_strike:
            self.state.add_log("ERROR", "Strategy", "Delta strike scan failed")
            return None

        quantity = Config.lot_size(stock_code) * lots
        sl_pct = sl_percentage if sl_percentage > 0 else Config.SL_PERCENTAGE

        strategy = Strategy(
            strategy_type=StrategyType.SHORT_STRANGLE,
            stock_code=stock_code,
            target_delta=target_delta,
            status=StrategyStatus.DEPLOYING,
        )

        ce_leg = self._make_leg(strategy.strategy_id, stock_code, ce_strike,
                                 OptionRight.CALL, expiry_date, quantity, sl_pct)
        pe_leg = self._make_leg(strategy.strategy_id, stock_code, pe_strike,
                                 OptionRight.PUT, expiry_date, quantity, sl_pct)

        strategy.legs = [ce_leg, pe_leg]
        self.db.save_strategy(strategy)
        self.db.save_leg(ce_leg)
        self.db.save_leg(pe_leg)

        self.session.subscribe_option(stock_code, ce_strike, "call", expiry_date)
        self.session.subscribe_option(stock_code, pe_strike, "put", expiry_date)
        time.sleep(0.5)

        ok_ce = self.order_mgr.execute_sell_with_chase(ce_leg)
        ok_pe = self.order_mgr.execute_sell_with_chase(pe_leg)

        strategy.status = (
            StrategyStatus.ACTIVE if (ok_ce and ok_pe)
            else StrategyStatus.ACTIVE if (ok_ce or ok_pe)
            else StrategyStatus.ERROR
        )
        self.db.save_strategy(strategy)
        return strategy

    # ── Manual single-leg limit sell ─────────────────────────

    def sell_single_leg_limit(
        self,
        stock_code: str,
        strike_price: float,
        right: OptionRight,
        expiry_date: str,
        lots: int,
        limit_price: float,
        sl_percentage: float = 0.0,
    ) -> Optional[Strategy]:
        """
        Manual limit order sell — used by the Limit Order Screen.
        Creates a strategy with a single leg.
        """
        quantity = Config.lot_size(stock_code) * lots
        sl_pct = sl_percentage if sl_percentage > 0 else Config.SL_PERCENTAGE

        strategy = Strategy(
            strategy_type=StrategyType.MANUAL_SELL,
            stock_code=stock_code,
            target_delta=0,
            status=StrategyStatus.DEPLOYING,
        )

        leg = self._make_leg(strategy.strategy_id, stock_code, strike_price,
                              right, expiry_date, quantity, sl_pct)

        strategy.legs = [leg]
        self.db.save_strategy(strategy)
        self.db.save_leg(leg)

        # Subscribe for price updates
        self.session.subscribe_option(stock_code, strike_price, right.value, expiry_date)
        time.sleep(0.3)

        ok = self.order_mgr.execute_limit_sell(leg, limit_price)
        strategy.status = StrategyStatus.ACTIVE if ok else StrategyStatus.ERROR
        self.db.save_strategy(strategy)
        return strategy

    # ── Helpers ──────────────────────────────────────────────

    def _make_leg(self, strategy_id, stock_code, strike, right,
                   expiry_date, quantity, sl_pct) -> Leg:
        return Leg(
            strategy_id=strategy_id,
            stock_code=stock_code,
            strike_price=strike,
            right=right,
            expiry_date=expiry_date,
            side=OrderSide.SELL,
            quantity=quantity,
            sl_percentage=sl_pct,
            status=LegStatus.ENTERING,
        )

    def _find_delta_strikes(self, stock_code, spot, expiry_date, target_delta):
        result = self.session.get_option_chain(stock_code, expiry_date, "others")
        if not result or result.get("Status") != 200 or not result.get("Success"):
            return None, None

        chain = result["Success"]
        T = compute_time_to_expiry(expiry_date)
        r = Config.RISK_FREE_RATE

        best_ce, best_ce_diff = None, float("inf")
        best_pe, best_pe_diff = None, float("inf")

        for item in chain:
            strike = safe_float(item.get("strike_price", 0))
            right_str = str(item.get("right", "")).lower()
            ltp = safe_float(item.get("ltp", 0))

            if strike <= 0 or ltp <= 0:
                continue

            if "call" in right_str:
                right = OptionRight.CALL
                iv = BlackScholes.implied_vol(ltp, spot, strike, T, r, right)
                delta = abs(BlackScholes.delta(spot, strike, T, r, iv, right))
                diff = abs(delta - target_delta)
                if diff < best_ce_diff and strike > spot:
                    best_ce_diff = diff
                    best_ce = strike
            elif "put" in right_str:
                right = OptionRight.PUT
                iv = BlackScholes.implied_vol(ltp, spot, strike, T, r, right)
                delta = abs(BlackScholes.delta(spot, strike, T, r, iv, right))
                diff = abs(delta - target_delta)
                if diff < best_pe_diff and strike < spot:
                    best_pe_diff = diff
                    best_pe = strike

        if best_ce and best_pe:
            self.state.add_log("INFO", "Strategy",
                               f"Delta scan: CE={int(best_ce)} (diff={best_ce_diff:.3f}), "
                               f"PE={int(best_pe)} (diff={best_pe_diff:.3f})")
        return best_ce, best_pe

    # ── Option chain with Greeks (for UI) ────────────────────

    def get_option_chain_with_greeks(
        self, stock_code: str = "NIFTY", expiry_date: str = "",
    ) -> List[dict]:
        if not expiry_date:
            expiry_date = breeze_expiry_format(next_weekly_expiry(stock_code))

        # Check cache (avoid hitting rate limit)
        cached, cache_time = self.state.get_chain_cache()
        if cached and (time.time() - cache_time) < 3.0:
            return cached

        spot = self.session.get_spot_price(stock_code)
        result = self.session.get_option_chain(stock_code, expiry_date, "others")
        if not result or not result.get("Success"):
            return cached if cached else []

        chain = result["Success"]
        T = compute_time_to_expiry(expiry_date)
        r = Config.RISK_FREE_RATE
        enriched = []

        for item in chain:
            strike = safe_float(item.get("strike_price", 0))
            right_str = str(item.get("right", "")).strip().lower()
            ltp = safe_float(item.get("ltp", 0))

            if strike <= 0:
                continue

            right = OptionRight.CALL if "call" in right_str else OptionRight.PUT

            if ltp > 0 and T > 0 and spot > 0:
                iv = BlackScholes.implied_vol(ltp, spot, strike, T, r, right)
                greeks = BlackScholes.greeks(spot, strike, T, r, iv, right)
            else:
                iv = 0.0
                greeks = Greeks()

            enriched.append({
                "strike": int(strike),
                "right": "CALL" if right == OptionRight.CALL else "PUT",
                "ltp": round(ltp, 2),
                "bid": safe_float(item.get("best_bid_price", 0)),
                "ask": safe_float(item.get("best_offer_price", 0)),
                "iv": round(iv * 100, 2),
                "delta": round(greeks.delta, 4),
                "gamma": round(greeks.gamma, 6),
                "theta": round(greeks.theta, 2),
                "vega": round(greeks.vega, 2),
                "oi": safe_float(item.get("open_interest", 0)),
                "volume": safe_float(item.get("volume",
                                    item.get("total_quantity_traded", 0))),
            })

        enriched.sort(key=lambda x: (x["strike"], x["right"]))
        self.state.set_chain_cache(enriched, time.time())
        return enriched
