"""
strategy_engine.py

Professional strategy deployment for:
  - short straddle
  - short strangle (target delta)
  - manual single-leg limit sell
  - iron condor (short strangle + protective wings)
  - iron butterfly (ATM short straddle + wings)

Notes:
  - Strike selection uses Breeze option chain + local BS Greeks.
  - Order placement uses OrderManager (preferred). If OrderManager lacks a method,
    we fall back to direct session.place_order for buy-to-open wings.
  - Every strategy is persisted to DB BEFORE execution for survivability.
"""

from __future__ import annotations

import time
from typing import List, Optional, Tuple

from app_config import Config
from connection_manager import SessionManager
from database import Database
from greeks_engine import BlackScholes, time_to_expiry
from models import (
    Strategy, StrategyType, StrategyStatus,
    Leg, LegStatus, OrderSide, OptionRight, Greeks
)
from order_manager import OrderManager
from shared_state import SharedState
from utils import LOG, breeze_expiry, next_weekly_expiry, atm_strike, safe_float


class StrategyEngine:
    def __init__(self, session: SessionManager, order_mgr: OrderManager, db: Database, state: SharedState):
        self.session = session
        self.omgr = order_mgr
        self.db = db
        self.state = state

    # ── Public deploy API ──────────────────────────────────────

    def deploy_short_straddle(
        self,
        instrument: str,
        expiry: str = "",
        lots: int = 1,
        sl_pct: float = 0.0,
    ) -> Optional[Strategy]:
        inst = Config.instrument(instrument)
        stock_code = inst["breeze_code"]
        exchange = inst["exchange"]
        gap = inst["strike_gap"]
        qty = inst["lot_size"] * int(lots)
        sl = float(sl_pct) if sl_pct > 0 else float(Config.SL_PERCENTAGE)

        if not expiry:
            expiry = breeze_expiry(next_weekly_expiry(instrument))

        spot = float(self.session.get_spot_price(stock_code))
        if spot <= 0:
            self.state.add_log("ERROR", "Strat", "No spot price; cannot deploy straddle")
            return None

        strike = atm_strike(spot, gap)

        s = Strategy(
            strategy_type=StrategyType.SHORT_STRADDLE,
            stock_code=stock_code,
            target_delta=0.5,
            status=StrategyStatus.DEPLOYING,
        )

        ce = self._make_sell_leg(s.strategy_id, stock_code, exchange, strike, OptionRight.CALL, expiry, qty, sl)
        pe = self._make_sell_leg(s.strategy_id, stock_code, exchange, strike, OptionRight.PUT, expiry, qty, sl)
        s.legs = [ce, pe]

        self._persist_strategy_and_legs(s)

        # subscribe feeds
        self._subscribe_legs(s.legs)
        time.sleep(0.5)

        ok_ce = self._sell_leg(ce)
        ok_pe = self._sell_leg(pe)

        s.status = StrategyStatus.ACTIVE if (ok_ce or ok_pe) else StrategyStatus.ERROR
        self.db.save_strategy(s)

        self.state.add_log("INFO", "Strat", f"Straddle deploy result: {s.status.value}")
        return s

    def deploy_short_strangle(
        self,
        instrument: str,
        target_delta: float = 0.15,
        expiry: str = "",
        lots: int = 1,
        sl_pct: float = 0.0,
    ) -> Optional[Strategy]:
        inst = Config.instrument(instrument)
        stock_code = inst["breeze_code"]
        exchange = inst["exchange"]
        qty = inst["lot_size"] * int(lots)
        sl = float(sl_pct) if sl_pct > 0 else float(Config.SL_PERCENTAGE)

        if not expiry:
            expiry = breeze_expiry(next_weekly_expiry(instrument))

        spot = float(self.session.get_spot_price(stock_code))
        if spot <= 0:
            self.state.add_log("ERROR", "Strat", "No spot price; cannot deploy strangle")
            return None

        ce_k, pe_k = self._find_delta_strikes(stock_code, exchange, spot, expiry, float(target_delta))
        if ce_k is None or pe_k is None:
            self.state.add_log("ERROR", "Strat", "Delta strike scan failed")
            return None

        s = Strategy(
            strategy_type=StrategyType.SHORT_STRANGLE,
            stock_code=stock_code,
            target_delta=float(target_delta),
            status=StrategyStatus.DEPLOYING,
        )
        ce = self._make_sell_leg(s.strategy_id, stock_code, exchange, ce_k, OptionRight.CALL, expiry, qty, sl)
        pe = self._make_sell_leg(s.strategy_id, stock_code, exchange, pe_k, OptionRight.PUT, expiry, qty, sl)
        s.legs = [ce, pe]

        self._persist_strategy_and_legs(s)
        self._subscribe_legs(s.legs)
        time.sleep(0.5)

        ok_ce = self._sell_leg(ce)
        ok_pe = self._sell_leg(pe)

        s.status = StrategyStatus.ACTIVE if (ok_ce or ok_pe) else StrategyStatus.ERROR
        self.db.save_strategy(s)
        self.state.add_log("INFO", "Strat", f"Strangle deploy result: {s.status.value}")
        return s

    def sell_single_leg_limit(
        self,
        instrument: str,
        strike: float,
        right: OptionRight,
        expiry: str,
        lots: int,
        limit_price: float,
        sl_pct: float = 0.0,
    ) -> Optional[Strategy]:
        inst = Config.instrument(instrument)
        stock_code = inst["breeze_code"]
        exchange = inst["exchange"]
        qty = inst["lot_size"] * int(lots)
        sl = float(sl_pct) if sl_pct > 0 else float(Config.SL_PERCENTAGE)

        s = Strategy(
            strategy_type=StrategyType.MANUAL_SELL,
            stock_code=stock_code,
            target_delta=0.0,
            status=StrategyStatus.DEPLOYING,
        )

        leg = self._make_sell_leg(s.strategy_id, stock_code, exchange, float(strike), right, expiry, qty, sl)
        s.legs = [leg]

        self._persist_strategy_and_legs(s)
        self._subscribe_legs([leg])
        time.sleep(0.3)

        ok = self._limit_sell_leg(leg, float(limit_price))
        s.status = StrategyStatus.ACTIVE if ok else StrategyStatus.ERROR
        self.db.save_strategy(s)
        return s

    def deploy_iron_condor(
        self,
        instrument: str,
        target_delta: float = 0.15,
        wing_width: int = 0,
        expiry: str = "",
        lots: int = 1,
        sl_pct: float = 0.0,
    ) -> Optional[Strategy]:
        """
        Iron Condor = short strangle + long wings further OTM.
        Execution order:
          - sell inner CE/PE (credit)
          - buy wings (protection)
        """
        inst = Config.instrument(instrument)
        stock_code = inst["breeze_code"]
        exchange = inst["exchange"]
        gap = inst["strike_gap"]
        qty = inst["lot_size"] * int(lots)
        sl = float(sl_pct) if sl_pct > 0 else float(Config.SL_PERCENTAGE)

        if not wing_width:
            wing_width = int(gap * 2)

        if not expiry:
            expiry = breeze_expiry(next_weekly_expiry(instrument))

        spot = float(self.session.get_spot_price(stock_code))
        if spot <= 0:
            self.state.add_log("ERROR", "Strat", "No spot price; cannot deploy iron condor")
            return None

        ce_inner, pe_inner = self._find_delta_strikes(stock_code, exchange, spot, expiry, float(target_delta))
        if ce_inner is None or pe_inner is None:
            self.state.add_log("ERROR", "Strat", "Delta scan failed")
            return None

        ce_outer = float(ce_inner + wing_width)
        pe_outer = float(pe_inner - wing_width)

        s = Strategy(
            strategy_type=getattr(StrategyType, "IRON_CONDOR", StrategyType.SHORT_STRANGLE),
            stock_code=stock_code,
            target_delta=float(target_delta),
            status=StrategyStatus.DEPLOYING,
        )

        sell_ce = self._make_sell_leg(s.strategy_id, stock_code, exchange, ce_inner, OptionRight.CALL, expiry, qty, sl)
        sell_pe = self._make_sell_leg(s.strategy_id, stock_code, exchange, pe_inner, OptionRight.PUT, expiry, qty, sl)

        buy_ce = self._make_buy_leg(s.strategy_id, stock_code, exchange, ce_outer, OptionRight.CALL, expiry, qty)
        buy_pe = self._make_buy_leg(s.strategy_id, stock_code, exchange, pe_outer, OptionRight.PUT, expiry, qty)

        s.legs = [sell_ce, sell_pe, buy_ce, buy_pe]
        self._persist_strategy_and_legs(s)

        self._subscribe_legs(s.legs)
        time.sleep(0.5)

        ok_s1 = self._sell_leg(sell_ce)
        ok_s2 = self._sell_leg(sell_pe)

        # Wings must be bought if any sell is live; otherwise abort.
        if ok_s1 or ok_s2:
            ok_b1 = self._buy_leg_market(buy_ce)
            ok_b2 = self._buy_leg_market(buy_pe)
            if not (ok_b1 and ok_b2):
                self.state.add_log("WARN", "Strat", "Wings incomplete; exposure may be naked")
            s.status = StrategyStatus.ACTIVE
        else:
            s.status = StrategyStatus.ERROR

        self.db.save_strategy(s)
        return s

    def deploy_iron_butterfly(
        self,
        instrument: str,
        wing_width: int = 0,
        expiry: str = "",
        lots: int = 1,
        sl_pct: float = 0.0,
    ) -> Optional[Strategy]:
        """
        Iron Butterfly = short ATM straddle + long wings.
        """
        inst = Config.instrument(instrument)
        stock_code = inst["breeze_code"]
        exchange = inst["exchange"]
        gap = inst["strike_gap"]
        qty = inst["lot_size"] * int(lots)
        sl = float(sl_pct) if sl_pct > 0 else float(Config.SL_PERCENTAGE)

        if not wing_width:
            wing_width = int(gap * 3)

        if not expiry:
            expiry = breeze_expiry(next_weekly_expiry(instrument))

        spot = float(self.session.get_spot_price(stock_code))
        if spot <= 0:
            self.state.add_log("ERROR", "Strat", "No spot price; cannot deploy iron butterfly")
            return None

        atm = float(atm_strike(spot, gap))

        s = Strategy(
            strategy_type=getattr(StrategyType, "IRON_BUTTERFLY", StrategyType.SHORT_STRADDLE),
            stock_code=stock_code,
            target_delta=0.5,
            status=StrategyStatus.DEPLOYING,
        )

        sell_ce = self._make_sell_leg(s.strategy_id, stock_code, exchange, atm, OptionRight.CALL, expiry, qty, sl)
        sell_pe = self._make_sell_leg(s.strategy_id, stock_code, exchange, atm, OptionRight.PUT, expiry, qty, sl)
        buy_ce = self._make_buy_leg(s.strategy_id, stock_code, exchange, atm + wing_width, OptionRight.CALL, expiry, qty)
        buy_pe = self._make_buy_leg(s.strategy_id, stock_code, exchange, atm - wing_width, OptionRight.PUT, expiry, qty)

        s.legs = [sell_ce, sell_pe, buy_ce, buy_pe]
        self._persist_strategy_and_legs(s)

        self._subscribe_legs(s.legs)
        time.sleep(0.5)

        ok_s1 = self._sell_leg(sell_ce)
        ok_s2 = self._sell_leg(sell_pe)

        if ok_s1 or ok_s2:
            self._buy_leg_market(buy_ce)
            self._buy_leg_market(buy_pe)
            s.status = StrategyStatus.ACTIVE
        else:
            s.status = StrategyStatus.ERROR

        self.db.save_strategy(s)
        return s

    # ── Option chain with greeks (UI) ─────────────────────────

    def get_chain_with_greeks(self, instrument: str, expiry: str = "") -> List[dict]:
        inst = Config.instrument(instrument)
        stock_code = inst["breeze_code"]
        exchange = inst["exchange"]

        if not expiry:
            expiry = breeze_expiry(next_weekly_expiry(instrument))

        # State cache to reduce API calls
        cache_key, cache_chain, cache_ts = self.state.get_chain_cache()
        want_key = f"{stock_code}|{expiry[:10]}"
        if cache_chain and cache_key == want_key and (time.time() - cache_ts) < 3.0:
            return cache_chain

        spot = float(self.session.get_spot_price(stock_code))
        res = self.session.get_option_chain(stock_code, expiry, right="others", exchange_code=exchange)
        if not res or not res.get("Success"):
            return cache_chain or []

        chain = res["Success"]
        T = time_to_expiry(expiry)
        r = Config.RISK_FREE_RATE

        enriched: List[dict] = []
        for item in chain:
            strike = safe_float(item.get("strike_price", 0))
            if strike <= 0:
                continue
            right_str = str(item.get("right", "")).lower()
            ltp = safe_float(item.get("ltp", 0))
            bid = safe_float(item.get("best_bid_price", 0))
            ask = safe_float(item.get("best_offer_price", 0))

            opt_right = OptionRight.CALL if "call" in right_str else OptionRight.PUT

            if spot > 0 and ltp > 0 and T > 0:
                iv = BlackScholes.implied_vol(ltp, spot, strike, T, r, opt_right)
                g = BlackScholes.greeks(spot, strike, T, r, iv, opt_right)
            else:
                iv = 0.0
                g = Greeks()

            enriched.append({
                "strike": int(strike),
                "right": "CALL" if opt_right == OptionRight.CALL else "PUT",
                "ltp": round(ltp, 2),
                "bid": round(bid, 2),
                "ask": round(ask, 2),
                "iv": round(iv * 100, 2),
                "delta": round(g.delta, 4),
                "gamma": round(g.gamma, 6),
                "theta": round(g.theta, 2),
                "vega": round(g.vega, 2),
                "oi": safe_float(item.get("open_interest", 0)),
                "volume": safe_float(item.get("volume", item.get("total_quantity_traded", 0))),
            })

        enriched.sort(key=lambda x: (x["strike"], x["right"]))
        self.state.set_chain_cache(want_key, enriched, time.time())
        return enriched

    # ── Internals ─────────────────────────────────────────────

    def _persist_strategy_and_legs(self, s: Strategy) -> None:
        self.db.save_strategy(s)
        for leg in s.legs:
            self.db.save_leg(leg)

    def _subscribe_legs(self, legs: List[Leg]) -> None:
        for leg in legs:
            try:
                self.session.subscribe_option(
                    stock_code=leg.stock_code,
                    strike=leg.strike_price,
                    right=leg.right.value,
                    expiry=leg.expiry_date,
                    exchange_code=leg.exchange_code,
                )
            except Exception as e:
                self.state.add_log("WARN", "WS", f"Subscribe failed: {leg.display_name}: {e}")

    def _make_sell_leg(
        self, strategy_id: str, stock_code: str, exchange: str,
        strike: float, right: OptionRight, expiry: str, qty: int, sl_pct: float
    ) -> Leg:
        return Leg(
            strategy_id=strategy_id,
            stock_code=stock_code,
            exchange_code=exchange,
            strike_price=float(strike),
            right=right,
            expiry_date=expiry,
            side=OrderSide.SELL,
            quantity=int(qty),
            sl_percentage=float(sl_pct),
            status=LegStatus.ENTERING,
        )

    def _make_buy_leg(
        self, strategy_id: str, stock_code: str, exchange: str,
        strike: float, right: OptionRight, expiry: str, qty: int
    ) -> Leg:
        return Leg(
            strategy_id=strategy_id,
            stock_code=stock_code,
            exchange_code=exchange,
            strike_price=float(strike),
            right=right,
            expiry_date=expiry,
            side=OrderSide.BUY,
            quantity=int(qty),
            sl_percentage=0.0,
            status=LegStatus.ENTERING,
        )

    def _sell_leg(self, leg: Leg) -> bool:
        # compatibility: different OrderManager versions
        if hasattr(self.omgr, "sell_chase"):
            return bool(self.omgr.sell_chase(leg))
        if hasattr(self.omgr, "execute_sell_with_chase"):
            return bool(self.omgr.execute_sell_with_chase(leg))
        raise RuntimeError("OrderManager missing sell chase method")

    def _limit_sell_leg(self, leg: Leg, limit_price: float) -> bool:
        if hasattr(self.omgr, "sell_limit"):
            return bool(self.omgr.sell_limit(leg, limit_price))
        if hasattr(self.omgr, "execute_limit_sell"):
            return bool(self.omgr.execute_limit_sell(leg, limit_price))
        raise RuntimeError("OrderManager missing limit sell method")

    def _buy_leg_market(self, leg: Leg) -> bool:
        """
        Buy-to-open wings. Prefer OrderManager.buy_entry_market if exists.
        Otherwise place a direct market BUY via SessionManager (and update leg state).
        """
        if hasattr(self.omgr, "buy_entry_market"):
            return bool(self.omgr.buy_entry_market(leg))

        # Fallback direct API (kept inside StrategyEngine to avoid editing other files)
        try:
            r = self.session.place_order(
                stock_code=leg.stock_code,
                exchange_code=leg.exchange_code,
                product="options",
                action="buy",
                order_type="market",
                stoploss="",
                quantity=str(leg.quantity),
                price="0",
                validity="day",
                validity_date="",
                disclosed_quantity="0",
                expiry_date=leg.expiry_date,
                right=leg.right.value,
                strike_price=str(int(leg.strike_price)),
            )
            if not r or r.get("Status") != 200:
                self.state.add_log("ERROR", "Order", f"Wing buy failed: {leg.display_name} {r}")
                return False

            order_id = r["Success"]["order_id"]
            # Try to read order detail for avg price
            avg_px = 0.0
            try:
                det = self.session.get_order_detail(order_id, exchange_code=leg.exchange_code)
                if det and det.get("Success"):
                    rec = det["Success"][0]
                    avg_px = safe_float(rec.get("average_price", 0)) or safe_float(rec.get("price", 0))
            except Exception:
                pass

            leg.entry_order_id = order_id
            leg.entry_price = float(avg_px) if avg_px > 0 else 0.0
            leg.current_price = leg.entry_price
            leg.status = LegStatus.ACTIVE
            self.db.save_leg(leg)
            return True

        except Exception as e:
            self.state.add_log("ERROR", "Order", f"Wing buy exception: {leg.display_name}: {e}")
            return False

    def _find_delta_strikes(
        self, stock_code: str, exchange_code: str, spot: float,
        expiry: str, target_delta: float
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Scan option chain. Compute IV and delta locally. Choose:
          - OTM CE strike with |Δ| closest to target_delta, strike > spot
          - OTM PE strike with |Δ| closest to target_delta, strike < spot
        """
        res = self.session.get_option_chain(stock_code, expiry, right="others", exchange_code=exchange_code)
        if not res or not res.get("Success"):
            return None, None

        chain = res["Success"]
        T = time_to_expiry(expiry)
        r = Config.RISK_FREE_RATE

        best_ce = None
        best_pe = None
        best_ce_diff = 1e9
        best_pe_diff = 1e9

        for item in chain:
            strike = safe_float(item.get("strike_price", 0))
            ltp = safe_float(item.get("ltp", 0))
            if strike <= 0 or ltp <= 0:
                continue

            right_str = str(item.get("right", "")).lower()
            if "call" in right_str:
                if strike <= spot:
                    continue
                iv = BlackScholes.implied_vol(ltp, spot, strike, T, r, OptionRight.CALL)
                d = abs(BlackScholes.delta(spot, strike, T, r, iv, OptionRight.CALL))
                diff = abs(d - target_delta)
                if diff < best_ce_diff:
                    best_ce_diff = diff
                    best_ce = strike
            elif "put" in right_str:
                if strike >= spot:
                    continue
                iv = BlackScholes.implied_vol(ltp, spot, strike, T, r, OptionRight.PUT)
                d = abs(BlackScholes.delta(spot, strike, T, r, iv, OptionRight.PUT))
                diff = abs(d - target_delta)
                if diff < best_pe_diff:
                    best_pe_diff = diff
                    best_pe = strike

        if best_ce and best_pe:
            self.state.add_log("INFO", "Strat", f"Delta strikes: CE={int(best_ce)} PE={int(best_pe)}")
        return best_ce, best_pe
