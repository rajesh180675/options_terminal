# ═══════════════════════════════════════════════════════════════
# FILE: multi_leg.py  (NEW — Iron Condor + Iron Butterfly builder)
# ═══════════════════════════════════════════════════════════════
"""
Multi-leg strategy builder:
  • Iron Condor (4 legs — defined risk short strangle)
  • Iron Butterfly (4 legs — defined risk short straddle)

These are what make Sensibull's strategy builder professional.
We deploy all legs as one Strategy with 4 Leg entries.
"""

import time
from typing import Optional, List
from models import (Strategy, Leg, StrategyType, StrategyStatus,
                    LegStatus, OrderSide, OptionRight)
from greeks_engine import BlackScholes, time_to_expiry
from connection_manager import SessionManager
from order_manager import OrderManager
from database import Database
from shared_state import SharedState
from app_config import Config
from utils import LOG, atm_strike, breeze_expiry, next_weekly_expiry, safe_float


# Extend StrategyType (we keep backward compatibility by checking string)
IRON_CONDOR = "iron_condor"
IRON_BUTTERFLY = "iron_butterfly"


class MultiLegBuilder:
    """Builds and deploys multi-leg strategies."""

    def __init__(self, session: SessionManager, omgr: OrderManager,
                 db: Database, state: SharedState):
        self.session = session
        self.omgr = omgr
        self.db = db
        self.state = state

    def deploy_iron_condor(
        self, instrument: str, target_delta: float = 0.15,
        wing_width: int = 2, expiry: str = "", lots: int = 1,
        sl_pct: float = 0.0,
    ) -> Optional[Strategy]:
        """
        Iron Condor:
          Sell OTM CE at ~target_delta
          Sell OTM PE at ~target_delta
          Buy further OTM CE (wing)
          Buy further OTM PE (wing)
        """
        inst = Config.instrument(instrument)
        bc, exc = inst["breeze_code"], inst["exchange"]
        gap = inst["strike_gap"]
        wing_pts = wing_width * gap
        if not expiry:
            expiry = breeze_expiry(next_weekly_expiry(instrument))

        spot = self.session.get_spot_price(bc)
        if spot <= 0:
            self.state.add_log("ERROR", "MultiLeg", "No spot")
            return None

        # Find delta strikes from chain
        sell_ce, sell_pe = self._find_delta_strikes(bc, exc, spot, expiry, target_delta)
        if not sell_ce or not sell_pe:
            self.state.add_log("ERROR", "MultiLeg", "Delta scan failed")
            return None

        buy_ce = sell_ce + wing_pts
        buy_pe = sell_pe - wing_pts
        qty = inst["lot_size"] * lots
        sl = sl_pct if sl_pct > 0 else Config.SL_PERCENTAGE

        self.state.add_log("INFO", "MultiLeg",
            f"IRON CONDOR: Sell {int(sell_ce)}C/{int(sell_pe)}P "
            f"Buy {int(buy_ce)}C/{int(buy_pe)}P")

        # Register as MANUAL_SELL type (closest existing type for multi-leg)
        strategy = Strategy(
            strategy_type=StrategyType.MANUAL_SELL,  # We'll display as "iron_condor"
            stock_code=bc, target_delta=target_delta,
            status=StrategyStatus.DEPLOYING,
        )

        legs = [
            self._leg(strategy.strategy_id, bc, exc, sell_ce, OptionRight.CALL,
                      expiry, qty, sl, OrderSide.SELL),
            self._leg(strategy.strategy_id, bc, exc, sell_pe, OptionRight.PUT,
                      expiry, qty, sl, OrderSide.SELL),
            self._leg(strategy.strategy_id, bc, exc, buy_ce, OptionRight.CALL,
                      expiry, qty, 0, OrderSide.BUY),
            self._leg(strategy.strategy_id, bc, exc, buy_pe, OptionRight.PUT,
                      expiry, qty, 0, OrderSide.BUY),
        ]
        strategy.legs = legs
        self.db.save_strategy(strategy)
        for l in legs:
            self.db.save_leg(l)

        # Subscribe all strikes
        for l in legs:
            self.session.subscribe_option(bc, l.strike_price, l.right.value, expiry, exc)
        time.sleep(0.5)

        # Execute: sell legs first (premium), then buy legs (hedge)
        ok_all = True
        for l in legs:
            if l.side == OrderSide.SELL:
                ok = self.omgr.sell_chase(l)
            else:
                ok = self.omgr.buy_market(l)  # Buy wings at market for speed
                if ok:
                    # For buys, mark as active with entry = fill price
                    l.status = LegStatus.ACTIVE
                    self.db.save_leg(l)
            ok_all = ok_all and ok

        strategy.status = StrategyStatus.ACTIVE if ok_all else StrategyStatus.ERROR
        self.db.save_strategy(strategy)
        return strategy

    def deploy_iron_butterfly(
        self, instrument: str, wing_width: int = 3,
        expiry: str = "", lots: int = 1, sl_pct: float = 0.0,
    ) -> Optional[Strategy]:
        """
        Iron Butterfly:
          Sell ATM CE + Sell ATM PE + Buy OTM CE + Buy OTM PE
        """
        inst = Config.instrument(instrument)
        bc, exc = inst["breeze_code"], inst["exchange"]
        gap = inst["strike_gap"]
        wing_pts = wing_width * gap
        if not expiry:
            expiry = breeze_expiry(next_weekly_expiry(instrument))

        spot = self.session.get_spot_price(bc)
        if spot <= 0:
            self.state.add_log("ERROR", "MultiLeg", "No spot")
            return None

        atm = atm_strike(spot, gap)
        buy_ce = atm + wing_pts
        buy_pe = atm - wing_pts
        qty = inst["lot_size"] * lots
        sl = sl_pct if sl_pct > 0 else Config.SL_PERCENTAGE

        self.state.add_log("INFO", "MultiLeg",
            f"IRON BUTTERFLY: Sell ATM={int(atm)} Buy {int(buy_ce)}C/{int(buy_pe)}P")

        strategy = Strategy(
            strategy_type=StrategyType.MANUAL_SELL,
            stock_code=bc, target_delta=0.5,
            status=StrategyStatus.DEPLOYING,
        )

        legs = [
            self._leg(strategy.strategy_id, bc, exc, atm, OptionRight.CALL,
                      expiry, qty, sl, OrderSide.SELL),
            self._leg(strategy.strategy_id, bc, exc, atm, OptionRight.PUT,
                      expiry, qty, sl, OrderSide.SELL),
            self._leg(strategy.strategy_id, bc, exc, buy_ce, OptionRight.CALL,
                      expiry, qty, 0, OrderSide.BUY),
            self._leg(strategy.strategy_id, bc, exc, buy_pe, OptionRight.PUT,
                      expiry, qty, 0, OrderSide.BUY),
        ]
        strategy.legs = legs
        self.db.save_strategy(strategy)
        for l in legs:
            self.db.save_leg(l)
        for l in legs:
            self.session.subscribe_option(bc, l.strike_price, l.right.value, expiry, exc)
        time.sleep(0.5)

        ok_all = True
        for l in legs:
            ok = self.omgr.sell_chase(l) if l.side == OrderSide.SELL else self.omgr.buy_market(l)
            if ok and l.side == OrderSide.BUY:
                l.status = LegStatus.ACTIVE
                self.db.save_leg(l)
            ok_all = ok_all and ok

        strategy.status = StrategyStatus.ACTIVE if ok_all else StrategyStatus.ERROR
        self.db.save_strategy(strategy)
        return strategy

    def _find_delta_strikes(self, bc, exc, spot, expiry, target):
        result = self.session.get_option_chain(bc, expiry, "others", exc)
        if not result or not result.get("Success"):
            return None, None
        T = time_to_expiry(expiry)
        r = Config.RISK_FREE_RATE
        best_ce, bcd = None, 99.0
        best_pe, bpd = None, 99.0
        for item in result["Success"]:
            strike = safe_float(item.get("strike_price", 0))
            rs = str(item.get("right", "")).lower()
            ltp = safe_float(item.get("ltp", 0))
            if strike <= 0 or ltp <= 0:
                continue
            if "call" in rs and strike > spot:
                iv = BlackScholes.implied_vol(ltp, spot, strike, T, r, OptionRight.CALL)
                d = abs(abs(BlackScholes.delta(spot, strike, T, r, iv, OptionRight.CALL)) - target)
                if d < bcd:
                    bcd, best_ce = d, strike
            elif "put" in rs and strike < spot:
                iv = BlackScholes.implied_vol(ltp, spot, strike, T, r, OptionRight.PUT)
                d = abs(abs(BlackScholes.delta(spot, strike, T, r, iv, OptionRight.PUT)) - target)
                if d < bpd:
                    bpd, best_pe = d, strike
        return best_ce, best_pe

    def _leg(self, sid, bc, exc, strike, right, expiry, qty, sl, side):
        return Leg(
            strategy_id=sid, stock_code=bc, exchange_code=exc,
            strike_price=strike, right=right, expiry_date=expiry,
            side=side, quantity=qty, sl_percentage=sl,
            status=LegStatus.ENTERING,
        )
