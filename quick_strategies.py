"""
quick_strategies.py  — NEW MODULE

Additional strategy templates not in original strategy_engine.py:
  • Bull Put Spread  (directional bullish, defined-risk)
  • Bear Call Spread (directional bearish, defined-risk)
  • Calendar Spread  (sell near expiry, buy far expiry at same strike)
  • Jade Lizard     (short put + short call spread: no upside risk)
  • Put Ratio Spread (sell 1 ATM put, buy 2 OTM puts: directional with leverage)
  • Synthetic Short Futures (buy ATM put, sell ATM call: ultra-low premium)

All methods follow the same interface as StrategyEngine and are called from
the TradingEngine via a thin adapter (QuickStrategyAdapter).
No existing file is modified.
"""

from __future__ import annotations

import time
from typing import Optional, Tuple, List

from app_config import Config
from connection_manager import SessionManager
from database import Database
from greeks_engine import BlackScholes, time_to_expiry
from models import (
    Strategy, StrategyType, StrategyStatus,
    Leg, LegStatus, OrderSide, OptionRight
)
from order_manager import OrderManager
from shared_state import SharedState
from utils import LOG, breeze_expiry, next_weekly_expiry, atm_strike, safe_float


# ── Strategy Type Extensions ──────────────────────────────────
# We reuse StrategyType.MANUAL_SELL for custom types and store
# the real name in strategy metadata field that we treat as tag.

STRATEGY_TAG = {
    "BULL_PUT_SPREAD": "bull_put_spread",
    "BEAR_CALL_SPREAD": "bear_call_spread",
    "CALENDAR_SPREAD": "calendar_spread",
    "JADE_LIZARD": "jade_lizard",
    "PUT_RATIO_SPREAD": "put_ratio_spread",
    "SYNTHETIC_SHORT": "synthetic_short",
}


class QuickStrategyEngine:
    """
    Extra strategies that complement StrategyEngine.
    Accepts same constructor signature for drop-in addition to TradingEngine.
    """

    def __init__(self, session: SessionManager, order_mgr: OrderManager,
                 db: Database, state: SharedState):
        self.session = session
        self.omgr = order_mgr
        self.db = db
        self.state = state

    # ── Bull Put Spread ───────────────────────────────────────

    def deploy_bull_put_spread(
        self,
        instrument: str,
        expiry: str = "",
        lots: int = 1,
        sl_pct: float = 0.0,
        short_delta: float = 0.30,
        wing_width: Optional[int] = None,
    ) -> Optional[Strategy]:
        """
        Sell an OTM put + buy a further OTM put (protective wing).
        Max profit = net credit. Max loss = wing_width - credit.
        Ideal when: bullish/neutral + high IV.
        """
        inst = Config.instrument(instrument)
        stock_code = inst["breeze_code"]
        exchange = inst["exchange"]
        gap = inst["strike_gap"]
        qty = inst["lot_size"] * lots
        sl = sl_pct if sl_pct > 0 else Config.SL_PERCENTAGE

        if not expiry:
            expiry = breeze_expiry(next_weekly_expiry(instrument))
        if wing_width is None:
            wing_width = gap * 2

        spot = float(self.session.get_spot_price(stock_code))
        if spot <= 0:
            self.state.add_log("ERROR", "QStrat", "No spot for bull put spread")
            return None

        short_strike = self._find_put_strike_by_delta(stock_code, exchange, spot, expiry, short_delta)
        if not short_strike:
            self.state.add_log("ERROR", "QStrat", "Could not find short put strike")
            return None

        long_strike = short_strike - wing_width

        s = Strategy(
            strategy_type=StrategyType.MANUAL_SELL,
            stock_code=stock_code,
            target_delta=short_delta,
            status=StrategyStatus.DEPLOYING,
        )

        short_pe = self._make_leg(s.strategy_id, stock_code, exchange, short_strike,
                                  OptionRight.PUT, expiry, qty, OrderSide.SELL, sl)
        long_pe = self._make_leg(s.strategy_id, stock_code, exchange, long_strike,
                                 OptionRight.PUT, expiry, qty, OrderSide.BUY, 0.0)
        s.legs = [short_pe, long_pe]

        self._save_and_subscribe(s)
        time.sleep(0.5)

        ok1 = self._sell_chase(short_pe)
        ok2 = self._buy_market(long_pe) if ok1 else False

        s.status = StrategyStatus.ACTIVE if ok1 else StrategyStatus.ERROR
        self.db.save_strategy(s)

        self.state.add_log("INFO", "QStrat", f"Bull Put Spread: {ok1}/{ok2}")
        return s

    # ── Bear Call Spread ──────────────────────────────────────

    def deploy_bear_call_spread(
        self,
        instrument: str,
        expiry: str = "",
        lots: int = 1,
        sl_pct: float = 0.0,
        short_delta: float = 0.30,
        wing_width: Optional[int] = None,
    ) -> Optional[Strategy]:
        """
        Sell an OTM call + buy a further OTM call.
        Max profit = net credit. Max loss = wing_width - credit.
        Ideal when: bearish/neutral + high IV.
        """
        inst = Config.instrument(instrument)
        stock_code = inst["breeze_code"]
        exchange = inst["exchange"]
        gap = inst["strike_gap"]
        qty = inst["lot_size"] * lots
        sl = sl_pct if sl_pct > 0 else Config.SL_PERCENTAGE

        if not expiry:
            expiry = breeze_expiry(next_weekly_expiry(instrument))
        if wing_width is None:
            wing_width = gap * 2

        spot = float(self.session.get_spot_price(stock_code))
        if spot <= 0:
            self.state.add_log("ERROR", "QStrat", "No spot for bear call spread")
            return None

        short_strike = self._find_call_strike_by_delta(stock_code, exchange, spot, expiry, short_delta)
        if not short_strike:
            self.state.add_log("ERROR", "QStrat", "Could not find short call strike")
            return None

        long_strike = short_strike + wing_width

        s = Strategy(
            strategy_type=StrategyType.MANUAL_SELL,
            stock_code=stock_code,
            target_delta=short_delta,
            status=StrategyStatus.DEPLOYING,
        )

        short_ce = self._make_leg(s.strategy_id, stock_code, exchange, short_strike,
                                  OptionRight.CALL, expiry, qty, OrderSide.SELL, sl)
        long_ce = self._make_leg(s.strategy_id, stock_code, exchange, long_strike,
                                 OptionRight.CALL, expiry, qty, OrderSide.BUY, 0.0)
        s.legs = [short_ce, long_ce]

        self._save_and_subscribe(s)
        time.sleep(0.5)

        ok1 = self._sell_chase(short_ce)
        ok2 = self._buy_market(long_ce) if ok1 else False

        s.status = StrategyStatus.ACTIVE if ok1 else StrategyStatus.ERROR
        self.db.save_strategy(s)

        self.state.add_log("INFO", "QStrat", f"Bear Call Spread: {ok1}/{ok2}")
        return s

    # ── Jade Lizard ───────────────────────────────────────────

    def deploy_jade_lizard(
        self,
        instrument: str,
        expiry: str = "",
        lots: int = 1,
        sl_pct: float = 0.0,
        put_delta: float = 0.30,
        call_spread_delta: float = 0.20,
        call_wing_width: Optional[int] = None,
    ) -> Optional[Strategy]:
        """
        Jade Lizard = short OTM put + short OTM call spread (bear call spread).
        Structure ensures NO upside risk (call credit covers wing width).
        Only downside risk below the short put breakeven.
        Ideal when: slight bullish bias, high IV, no risk of upside move.

        Legs:
          - Short OTM put (put_delta ≈ 0.30)
          - Short OTM call (call_spread_delta ≈ 0.20)
          - Long further OTM call (wing)
        """
        inst = Config.instrument(instrument)
        stock_code = inst["breeze_code"]
        exchange = inst["exchange"]
        gap = inst["strike_gap"]
        qty = inst["lot_size"] * lots
        sl = sl_pct if sl_pct > 0 else Config.SL_PERCENTAGE

        if not expiry:
            expiry = breeze_expiry(next_weekly_expiry(instrument))
        if call_wing_width is None:
            call_wing_width = gap * 2

        spot = float(self.session.get_spot_price(stock_code))
        if spot <= 0:
            self.state.add_log("ERROR", "QStrat", "No spot for jade lizard")
            return None

        put_strike = self._find_put_strike_by_delta(stock_code, exchange, spot, expiry, put_delta)
        call_short_strike = self._find_call_strike_by_delta(stock_code, exchange, spot, expiry, call_spread_delta)

        if not put_strike or not call_short_strike:
            self.state.add_log("ERROR", "QStrat", "Could not find jade lizard strikes")
            return None

        call_long_strike = call_short_strike + call_wing_width

        s = Strategy(
            strategy_type=StrategyType.MANUAL_SELL,
            stock_code=stock_code,
            target_delta=put_delta,
            status=StrategyStatus.DEPLOYING,
        )

        short_put = self._make_leg(s.strategy_id, stock_code, exchange, put_strike,
                                   OptionRight.PUT, expiry, qty, OrderSide.SELL, sl)
        short_call = self._make_leg(s.strategy_id, stock_code, exchange, call_short_strike,
                                    OptionRight.CALL, expiry, qty, OrderSide.SELL, sl)
        long_call = self._make_leg(s.strategy_id, stock_code, exchange, call_long_strike,
                                   OptionRight.CALL, expiry, qty, OrderSide.BUY, 0.0)
        s.legs = [short_put, short_call, long_call]

        self._save_and_subscribe(s)
        time.sleep(0.5)

        ok1 = self._sell_chase(short_put)
        ok2 = self._sell_chase(short_call)
        ok3 = self._buy_market(long_call) if (ok1 or ok2) else False

        s.status = StrategyStatus.ACTIVE if (ok1 or ok2) else StrategyStatus.ERROR
        self.db.save_strategy(s)

        self.state.add_log("INFO", "QStrat", f"Jade Lizard: put={ok1} call={ok2} wing={ok3}")
        return s

    # ── Put Ratio Back Spread ─────────────────────────────────

    def deploy_put_ratio_spread(
        self,
        instrument: str,
        expiry: str = "",
        lots: int = 1,
        sell_delta: float = 0.40,
    ) -> Optional[Strategy]:
        """
        Put Ratio Spread: Sell 1 ATM/NTM put, Buy 2 OTM puts.
        Net debit or small credit. Profits sharply on big downmove.
        Ideal when: expecting volatile downmove + low cost entry.

        Note: Unlimited downside profits, limited upside loss.
        """
        inst = Config.instrument(instrument)
        stock_code = inst["breeze_code"]
        exchange = inst["exchange"]
        gap = inst["strike_gap"]
        qty = inst["lot_size"] * lots
        sl = Config.SL_PERCENTAGE

        if not expiry:
            expiry = breeze_expiry(next_weekly_expiry(instrument))

        spot = float(self.session.get_spot_price(stock_code))
        if spot <= 0:
            self.state.add_log("ERROR", "QStrat", "No spot for put ratio spread")
            return None

        sell_strike = self._find_put_strike_by_delta(stock_code, exchange, spot, expiry, sell_delta)
        if not sell_strike:
            return None

        buy_strike = sell_strike - gap * 2  # further OTM puts

        s = Strategy(
            strategy_type=StrategyType.MANUAL_SELL,
            stock_code=stock_code,
            target_delta=sell_delta,
            status=StrategyStatus.DEPLOYING,
        )

        short_pe = self._make_leg(s.strategy_id, stock_code, exchange, sell_strike,
                                  OptionRight.PUT, expiry, qty, OrderSide.SELL, sl)
        long_pe_1 = self._make_leg(s.strategy_id, stock_code, exchange, buy_strike,
                                   OptionRight.PUT, expiry, qty * 2, OrderSide.BUY, 0.0)
        s.legs = [short_pe, long_pe_1]

        self._save_and_subscribe(s)
        time.sleep(0.5)

        ok1 = self._sell_chase(short_pe)
        ok2 = self._buy_market(long_pe_1) if ok1 else False

        s.status = StrategyStatus.ACTIVE if ok1 else StrategyStatus.ERROR
        self.db.save_strategy(s)

        self.state.add_log("INFO", "QStrat", f"Put Ratio Spread: {ok1}/{ok2}")
        return s

    # ── Synthetic Short Futures ───────────────────────────────

    def deploy_synthetic_short(
        self,
        instrument: str,
        expiry: str = "",
        lots: int = 1,
    ) -> Optional[Strategy]:
        """
        Synthetic Short Futures = Buy ATM put + Sell ATM call (same strike).
        Delta ≈ -1, replicates short futures without futures account.
        Net premium is usually close to zero (put-call parity).
        Ideal when: strongly bearish, want futures-like exposure via options.
        """
        inst = Config.instrument(instrument)
        stock_code = inst["breeze_code"]
        exchange = inst["exchange"]
        gap = inst["strike_gap"]
        qty = inst["lot_size"] * lots
        sl = Config.SL_PERCENTAGE

        if not expiry:
            expiry = breeze_expiry(next_weekly_expiry(instrument))

        spot = float(self.session.get_spot_price(stock_code))
        if spot <= 0:
            self.state.add_log("ERROR", "QStrat", "No spot for synthetic short")
            return None

        strike = float(atm_strike(spot, gap))

        s = Strategy(
            strategy_type=StrategyType.MANUAL_SELL,
            stock_code=stock_code,
            target_delta=-1.0,
            status=StrategyStatus.DEPLOYING,
        )

        long_put = self._make_leg(s.strategy_id, stock_code, exchange, strike,
                                  OptionRight.PUT, expiry, qty, OrderSide.BUY, 0.0)
        short_call = self._make_leg(s.strategy_id, stock_code, exchange, strike,
                                    OptionRight.CALL, expiry, qty, OrderSide.SELL, sl)
        s.legs = [long_put, short_call]

        self._save_and_subscribe(s)
        time.sleep(0.5)

        ok1 = self._buy_market(long_put)
        ok2 = self._sell_chase(short_call) if ok1 else False

        s.status = StrategyStatus.ACTIVE if (ok1 or ok2) else StrategyStatus.ERROR
        self.db.save_strategy(s)

        self.state.add_log("INFO", "QStrat", f"Synthetic Short: {ok1}/{ok2}")
        return s

    # ── Strategy Descriptions (for UI) ───────────────────────

    @staticmethod
    def get_strategy_info() -> List[dict]:
        """Return metadata about available quick strategies for UI rendering."""
        return [
            {
                "id": "bull_put_spread",
                "name": "Bull Put Spread",
                "bias": "Bullish / Neutral",
                "risk": "Defined",
                "ideal": "High IV, expecting no downmove",
                "legs": "Short OTM PE + Long lower PE",
                "max_profit": "Net premium credit",
                "max_loss": "Wing width − credit",
                "emoji": "🟢",
            },
            {
                "id": "bear_call_spread",
                "name": "Bear Call Spread",
                "bias": "Bearish / Neutral",
                "risk": "Defined",
                "ideal": "High IV, expecting no upmove",
                "legs": "Short OTM CE + Long higher CE",
                "max_profit": "Net premium credit",
                "max_loss": "Wing width − credit",
                "emoji": "🔴",
            },
            {
                "id": "jade_lizard",
                "name": "Jade Lizard",
                "bias": "Slight Bullish",
                "risk": "No upside risk",
                "ideal": "High IV, slight bullish view",
                "legs": "Short PE + Short CE + Long higher CE",
                "max_profit": "Total net credit",
                "max_loss": "Below short PE (potentially large)",
                "emoji": "🦎",
            },
            {
                "id": "put_ratio_spread",
                "name": "Put Ratio Spread (1×2)",
                "bias": "Bearish / Volatile",
                "risk": "Defined upside, large downside profit",
                "ideal": "Expecting sharp down move",
                "legs": "Sell 1 ATM/NTM PE + Buy 2 OTM PE",
                "max_profit": "Unlimited on sharp down move",
                "max_loss": "Capped at buy strike level",
                "emoji": "📉",
            },
            {
                "id": "synthetic_short",
                "name": "Synthetic Short Futures",
                "bias": "Strongly Bearish",
                "risk": "Unlimited (same as short futures)",
                "ideal": "Strong directional bearish view",
                "legs": "Long ATM PE + Short ATM CE (same strike)",
                "max_profit": "Unlimited on downmove",
                "max_loss": "Unlimited on upmove",
                "emoji": "⬇️",
            },
        ]

    # ── Internal helpers ──────────────────────────────────────

    def _make_leg(self, strategy_id, stock_code, exchange, strike, right, expiry, qty, side, sl_pct) -> Leg:
        return Leg(
            strategy_id=strategy_id,
            stock_code=stock_code,
            exchange_code=exchange,
            strike_price=float(strike),
            right=right,
            expiry_date=expiry,
            side=side,
            quantity=int(qty),
            sl_percentage=float(sl_pct),
            status=LegStatus.ENTERING,
        )

    def _save_and_subscribe(self, s: Strategy) -> None:
        self.db.save_strategy(s)
        for leg in s.legs:
            self.db.save_leg(leg)
            try:
                self.session.subscribe_option(
                    stock_code=leg.stock_code,
                    strike=leg.strike_price,
                    right=leg.right.value,
                    expiry=leg.expiry_date,
                    exchange_code=leg.exchange_code,
                )
            except Exception as e:
                self.state.add_log("WARN", "QStrat", f"Subscribe failed: {leg.display_name}: {e}")

    def _sell_chase(self, leg: Leg) -> bool:
        if hasattr(self.omgr, "sell_chase"):
            return bool(self.omgr.sell_chase(leg))
        if hasattr(self.omgr, "execute_sell_with_chase"):
            return bool(self.omgr.execute_sell_with_chase(leg))
        return False

    def _buy_market(self, leg: Leg) -> bool:
        if hasattr(self.omgr, "buy_entry_market"):
            return bool(self.omgr.buy_entry_market(leg))
        if hasattr(self.omgr, "buy_market"):
            return bool(self.omgr.buy_market(leg))
        return False

    def _find_call_strike_by_delta(self, stock_code, exchange, spot, expiry, target_delta) -> Optional[float]:
        return self._find_strike_by_delta(stock_code, exchange, spot, expiry, target_delta, "call")

    def _find_put_strike_by_delta(self, stock_code, exchange, spot, expiry, target_delta) -> Optional[float]:
        return self._find_strike_by_delta(stock_code, exchange, spot, expiry, target_delta, "put")

    def _find_strike_by_delta(self, stock_code, exchange, spot, expiry, target_delta, side) -> Optional[float]:
        try:
            res = self.session.get_option_chain(stock_code, expiry, right="others", exchange_code=exchange)
            if not res or not res.get("Success"):
                return None

            chain = res["Success"]
            T = time_to_expiry(expiry)
            r = Config.RISK_FREE_RATE
            right_enum = OptionRight.CALL if side == "call" else OptionRight.PUT

            best_strike = None
            best_diff = 1e9

            for item in chain:
                k = safe_float(item.get("strike_price", 0))
                ltp = safe_float(item.get("ltp", 0))
                item_right = str(item.get("right", "")).lower()
                if k <= 0 or ltp <= 0:
                    continue
                if side == "call" and "call" not in item_right:
                    continue
                if side == "put" and "put" not in item_right:
                    continue
                if side == "call" and k <= spot:
                    continue
                if side == "put" and k >= spot:
                    continue

                iv = BlackScholes.implied_vol(ltp, spot, k, T, r, right_enum)
                d = abs(BlackScholes.delta(spot, k, T, r, iv, right_enum))
                diff = abs(d - target_delta)
                if diff < best_diff:
                    best_diff = diff
                    best_strike = k

            return best_strike
        except Exception as e:
            self.state.add_log("WARN", "QStrat", f"Strike scan error: {e}")
            return None
