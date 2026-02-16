# ═══════════════════════════════════════════════════════════════
# FILE: strategy_engine.py
# ═══════════════════════════════════════════════════════════════
"""
Strategy deployment engine.
  • Short Straddle: sells ATM CE + PE
  • Short Strangle: sells OTM CE + PE targeting a specific delta
"""

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
from utils import LOG, atm_strike, breeze_date, next_weekly_expiry, safe_float


class StrategyEngine:
    """Autonomous strategy deployer."""

    def __init__(
        self,
        session: SessionManager,
        order_mgr: OrderManager,
        db: Database,
        state: SharedState,
    ):
        self.session = session
        self.order_mgr = order_mgr
        self.db = db
        self.state = state

    # ── Public deployment methods ────────────────────────────

    def deploy_short_straddle(
        self,
        stock_code: str = "NIFTY",
        expiry_date: str = "",
        lots: int = 1,
        sl_percentage: float = 0.0,
    ) -> Optional[Strategy]:
        """Sell ATM CE + ATM PE."""
        if not expiry_date:
            expiry_date = breeze_date(next_weekly_expiry(stock_code))

        spot = self.session.get_spot_price(stock_code)
        if spot <= 0:
            self.state.add_log("ERROR", "Strategy", "Cannot get spot price")
            return None

        gap = Config.strike_gap(stock_code)
        atm = atm_strike(spot, gap)
        lot_size = Config.lot_size(stock_code)
        quantity = lot_size * lots
        sl_pct = sl_percentage if sl_percentage > 0 else Config.SL_PERCENTAGE

        self.state.add_log("INFO", "Strategy",
                           f"Deploying SHORT STRADDLE: {stock_code} ATM={atm} "
                           f"qty={quantity} expiry={expiry_date[:10]}")

        strategy = Strategy(
            strategy_type=StrategyType.SHORT_STRADDLE,
            stock_code=stock_code,
            target_delta=0.5,
            status=StrategyStatus.DEPLOYING,
        )

        # CE Leg
        ce_leg = Leg(
            strategy_id=strategy.strategy_id,
            stock_code=stock_code,
            strike_price=atm,
            right=OptionRight.CALL,
            expiry_date=expiry_date,
            side=OrderSide.SELL,
            quantity=quantity,
            sl_percentage=sl_pct,
            status=LegStatus.ENTERING,
        )

        # PE Leg
        pe_leg = Leg(
            strategy_id=strategy.strategy_id,
            stock_code=stock_code,
            strike_price=atm,
            right=OptionRight.PUT,
            expiry_date=expiry_date,
            side=OrderSide.SELL,
            quantity=quantity,
            sl_percentage=sl_pct,
            status=LegStatus.ENTERING,
        )

        strategy.legs = [ce_leg, pe_leg]
        self.db.save_strategy(strategy)
        self.db.save_leg(ce_leg)
        self.db.save_leg(pe_leg)

        # Subscribe to feeds for both legs
        self.session.subscribe_option(stock_code, atm, "call", expiry_date)
        self.session.subscribe_option(stock_code, atm, "put", expiry_date)
        import time; time.sleep(0.5)  # Brief pause to receive initial ticks

        # Execute both legs
        success_ce = self.order_mgr.execute_sell_with_chase(ce_leg)
        success_pe = self.order_mgr.execute_sell_with_chase(pe_leg)

        if success_ce and success_pe:
            strategy.status = StrategyStatus.ACTIVE
            self.state.add_log("INFO", "Strategy", "Short Straddle ACTIVE")
        elif success_ce or success_pe:
            strategy.status = StrategyStatus.ACTIVE  # partial but active
            self.state.add_log("WARN", "Strategy",
                               "Short Straddle partially filled")
        else:
            strategy.status = StrategyStatus.ERROR
            self.state.add_log("ERROR", "Strategy", "Short Straddle FAILED")

        self.db.save_strategy(strategy)
        return strategy

    def deploy_short_strangle(
        self,
        stock_code: str = "NIFTY",
        target_delta: float = 0.15,
        expiry_date: str = "",
        lots: int = 1,
        sl_percentage: float = 0.0,
    ) -> Optional[Strategy]:
        """Sell OTM CE + OTM PE at the specified delta level."""
        if not expiry_date:
            expiry_date = breeze_date(next_weekly_expiry(stock_code))

        spot = self.session.get_spot_price(stock_code)
        if spot <= 0:
            self.state.add_log("ERROR", "Strategy", "Cannot get spot price")
            return None

        self.state.add_log("INFO", "Strategy",
                           f"Scanning chain for {target_delta:.2f}Δ strangle…")

        # Find the best strikes via option chain + local Greeks
        ce_strike, pe_strike = self._find_delta_strikes(
            stock_code, spot, expiry_date, target_delta
        )

        if ce_strike is None or pe_strike is None:
            self.state.add_log("ERROR", "Strategy", "Could not find delta strikes")
            return None

        lot_size = Config.lot_size(stock_code)
        quantity = lot_size * lots
        sl_pct = sl_percentage if sl_percentage > 0 else Config.SL_PERCENTAGE

        self.state.add_log("INFO", "Strategy",
                           f"Deploying SHORT STRANGLE: CE={ce_strike} PE={pe_strike} "
                           f"target_delta={target_delta}")

        strategy = Strategy(
            strategy_type=StrategyType.SHORT_STRANGLE,
            stock_code=stock_code,
            target_delta=target_delta,
            status=StrategyStatus.DEPLOYING,
        )

        ce_leg = Leg(
            strategy_id=strategy.strategy_id,
            stock_code=stock_code,
            strike_price=ce_strike,
            right=OptionRight.CALL,
            expiry_date=expiry_date,
            side=OrderSide.SELL,
            quantity=quantity,
            sl_percentage=sl_pct,
            status=LegStatus.ENTERING,
        )

        pe_leg = Leg(
            strategy_id=strategy.strategy_id,
            stock_code=stock_code,
            strike_price=pe_strike,
            right=OptionRight.PUT,
            expiry_date=expiry_date,
            side=OrderSide.SELL,
            quantity=quantity,
            sl_percentage=sl_pct,
            status=LegStatus.ENTERING,
        )

        strategy.legs = [ce_leg, pe_leg]
        self.db.save_strategy(strategy)
        self.db.save_leg(ce_leg)
        self.db.save_leg(pe_leg)

        # Subscribe to feeds
        self.session.subscribe_option(stock_code, ce_strike, "call", expiry_date)
        self.session.subscribe_option(stock_code, pe_strike, "put", expiry_date)
        import time; time.sleep(0.5)

        # Execute
        success_ce = self.order_mgr.execute_sell_with_chase(ce_leg)
        success_pe = self.order_mgr.execute_sell_with_chase(pe_leg)

        if success_ce and success_pe:
            strategy.status = StrategyStatus.ACTIVE
        elif success_ce or success_pe:
            strategy.status = StrategyStatus.ACTIVE
        else:
            strategy.status = StrategyStatus.ERROR

        self.db.save_strategy(strategy)
        return strategy

    # ── Strike selection via Greeks ──────────────────────────

    def _find_delta_strikes(
        self,
        stock_code: str,
        spot: float,
        expiry_date: str,
        target_delta: float,
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Scan the option chain, compute local BS delta for each strike,
        and return (CE_strike, PE_strike) closest to ±target_delta.
        """
        result = self.session.get_option_chain(stock_code, expiry_date, "others")
        if not result or result.get("Status") != 200 or not result.get("Success"):
            LOG.error("Option chain fetch failed")
            return None, None

        chain = result["Success"]
        T = compute_time_to_expiry(expiry_date)
        r = Config.RISK_FREE_RATE

        best_ce_strike = None
        best_ce_delta_diff = float("inf")
        best_pe_strike = None
        best_pe_delta_diff = float("inf")

        for item in chain:
            strike = safe_float(item.get("strike_price", 0))
            right_str = item.get("right", "").lower()
            ltp = safe_float(item.get("ltp", 0))

            if strike <= 0 or ltp <= 0:
                continue

            if "call" in right_str:
                right = OptionRight.CALL
                # Compute IV then delta
                iv = BlackScholes.implied_vol(ltp, spot, strike, T, r, right)
                delta = abs(BlackScholes.delta(spot, strike, T, r, iv, right))
                diff = abs(delta - target_delta)
                if diff < best_ce_delta_diff and strike > spot:
                    best_ce_delta_diff = diff
                    best_ce_strike = strike
            elif "put" in right_str:
                right = OptionRight.PUT
                iv = BlackScholes.implied_vol(ltp, spot, strike, T, r, right)
                delta = abs(BlackScholes.delta(spot, strike, T, r, iv, right))
                diff = abs(delta - target_delta)
                if diff < best_pe_delta_diff and strike < spot:
                    best_pe_delta_diff = diff
                    best_pe_strike = strike

        if best_ce_strike and best_pe_strike:
            self.state.add_log("INFO", "Strategy",
                               f"Delta scan: CE={best_ce_strike} (Δ diff={best_ce_delta_diff:.3f}), "
                               f"PE={best_pe_strike} (Δ diff={best_pe_delta_diff:.3f})")
        return best_ce_strike, best_pe_strike

    # ── Chain snapshot (for UI) ──────────────────────────────

    def get_option_chain_with_greeks(
        self,
        stock_code: str = "NIFTY",
        expiry_date: str = "",
    ) -> List[dict]:
        """Return enriched option chain with locally computed Greeks."""
        if not expiry_date:
            expiry_date = breeze_date(next_weekly_expiry(stock_code))

        spot = self.session.get_spot_price(stock_code)
        result = self.session.get_option_chain(stock_code, expiry_date, "others")
        if not result or not result.get("Success"):
            return []

        chain = result["Success"]
        T = compute_time_to_expiry(expiry_date)
        r = Config.RISK_FREE_RATE
        enriched = []

        for item in chain:
            strike = safe_float(item.get("strike_price", 0))
            right_str = item.get("right", "").lower()
            ltp = safe_float(item.get("ltp", 0))

            if strike <= 0:
                continue

            right = OptionRight.CALL if "call" in right_str else OptionRight.PUT

            if ltp > 0 and T > 0:
                iv = BlackScholes.implied_vol(ltp, spot, strike, T, r, right)
                greeks = BlackScholes.greeks(spot, strike, T, r, iv, right)
            else:
                iv = 0.0
                greeks = Greeks()

            enriched.append({
                "strike": strike,
                "right": right_str.upper(),
                "ltp": ltp,
                "bid": safe_float(item.get("best_bid_price", 0)),
                "ask": safe_float(item.get("best_offer_price", 0)),
                "iv": round(iv * 100, 2),
                "delta": round(greeks.delta, 4),
                "gamma": round(greeks.gamma, 6),
                "theta": round(greeks.theta, 2),
                "vega": round(greeks.vega, 2),
                "oi": safe_float(item.get("open_interest", 0)),
                "volume": safe_float(item.get("volume", 0)),
            })

        return sorted(enriched, key=lambda x: (x["strike"], x["right"]))
