# ═══════════════════════════════════════════════════════════════
# FILE: strategy_engine.py
# ═══════════════════════════════════════════════════════════════
"""
Strategy deployer: straddle, strangle, manual sell.
Supports NFO (NIFTY, CNXBAN) and BFO (BSESEN).
"""

import time
from typing import Optional, Tuple, List
from models import (Strategy, Leg, StrategyType, StrategyStatus,
                    LegStatus, OrderSide, OptionRight, Greeks)
from greeks_engine import BlackScholes, time_to_expiry
from connection_manager import SessionManager
from order_manager import OrderManager
from database import Database
from shared_state import SharedState
from app_config import Config
from utils import LOG, atm_strike, breeze_expiry, next_weekly_expiry, safe_float


class StrategyEngine:

    def __init__(self, session: SessionManager, order_mgr: OrderManager,
                 db: Database, state: SharedState):
        self.session = session
        self.omgr = order_mgr
        self.db = db
        self.state = state

    def deploy_straddle(self, instrument: str, expiry="", lots=1, sl_pct=0.0):
        inst = Config.instrument(instrument)
        bc = inst["breeze_code"]
        exc = inst["exchange"]
        if not expiry:
            expiry = breeze_expiry(next_weekly_expiry(instrument))
        spot = self.session.get_spot_price(bc)
        if spot <= 0:
            self.state.add_log("ERROR", "Strat", "No spot"); return None
        gap = inst["strike_gap"]
        atm = atm_strike(spot, gap)
        qty = inst["lot_size"] * lots
        sl = sl_pct if sl_pct > 0 else Config.SL_PERCENTAGE
        self.state.add_log("INFO", "Strat",
                           f"STRADDLE: {bc} ATM={int(atm)} qty={qty}")
        s = Strategy(strategy_type=StrategyType.SHORT_STRADDLE,
                     stock_code=bc, target_delta=0.5,
                     status=StrategyStatus.DEPLOYING)
        ce = self._leg(s.strategy_id, bc, exc, atm, OptionRight.CALL, expiry, qty, sl)
        pe = self._leg(s.strategy_id, bc, exc, atm, OptionRight.PUT, expiry, qty, sl)
        s.legs = [ce, pe]
        self.db.save_strategy(s); self.db.save_leg(ce); self.db.save_leg(pe)
        self.session.subscribe_option(bc, atm, "call", expiry, exc)
        self.session.subscribe_option(bc, atm, "put", expiry, exc)
        time.sleep(0.5)
        ok_c = self.omgr.sell_chase(ce)
        ok_p = self.omgr.sell_chase(pe)
        s.status = StrategyStatus.ACTIVE if (ok_c or ok_p) else StrategyStatus.ERROR
        self.db.save_strategy(s)
        return s

    def deploy_strangle(self, instrument, delta=0.15, expiry="", lots=1, sl_pct=0.0):
        inst = Config.instrument(instrument)
        bc = inst["breeze_code"]
        exc = inst["exchange"]
        if not expiry:
            expiry = breeze_expiry(next_weekly_expiry(instrument))
        spot = self.session.get_spot_price(bc)
        if spot <= 0:
            self.state.add_log("ERROR", "Strat", "No spot"); return None
        self.state.add_log("INFO", "Strat", f"Scanning {delta:.2f}Δ strangle…")
        ce_s, pe_s = self._delta_strikes(bc, exc, spot, expiry, delta)
        if not ce_s or not pe_s:
            self.state.add_log("ERROR", "Strat", "Delta scan failed"); return None
        qty = inst["lot_size"] * lots
        sl = sl_pct if sl_pct > 0 else Config.SL_PERCENTAGE
        s = Strategy(strategy_type=StrategyType.SHORT_STRANGLE,
                     stock_code=bc, target_delta=delta,
                     status=StrategyStatus.DEPLOYING)
        ce = self._leg(s.strategy_id, bc, exc, ce_s, OptionRight.CALL, expiry, qty, sl)
        pe = self._leg(s.strategy_id, bc, exc, pe_s, OptionRight.PUT, expiry, qty, sl)
        s.legs = [ce, pe]
        self.db.save_strategy(s); self.db.save_leg(ce); self.db.save_leg(pe)
        self.session.subscribe_option(bc, ce_s, "call", expiry, exc)
        self.session.subscribe_option(bc, pe_s, "put", expiry, exc)
        time.sleep(0.5)
        ok_c = self.omgr.sell_chase(ce)
        ok_p = self.omgr.sell_chase(pe)
        s.status = StrategyStatus.ACTIVE if (ok_c or ok_p) else StrategyStatus.ERROR
        self.db.save_strategy(s)
        return s

    def sell_single(self, instrument, strike, right, expiry, lots, price, sl_pct=0.0):
        inst = Config.instrument(instrument)
        bc = inst["breeze_code"]
        exc = inst["exchange"]
        qty = inst["lot_size"] * lots
        sl = sl_pct if sl_pct > 0 else Config.SL_PERCENTAGE
        s = Strategy(strategy_type=StrategyType.MANUAL_SELL,
                     stock_code=bc, status=StrategyStatus.DEPLOYING)
        leg = self._leg(s.strategy_id, bc, exc, strike, right, expiry, qty, sl)
        s.legs = [leg]
        self.db.save_strategy(s); self.db.save_leg(leg)
        self.session.subscribe_option(bc, strike, right.value, expiry, exc)
        time.sleep(0.3)
        ok = self.omgr.sell_limit(leg, price)
        s.status = StrategyStatus.ACTIVE if ok else StrategyStatus.ERROR
        self.db.save_strategy(s)
        return s

    # ── Chain with Greeks ────────────────────────────────────

    def chain_with_greeks(self, instrument, expiry=""):
        inst = Config.instrument(instrument)
        bc = inst["breeze_code"]
        exc = inst["exchange"]
        if not expiry:
            expiry = breeze_expiry(next_weekly_expiry(instrument))
        cached, ct = self.state.get_chain()
        if cached and (time.time() - ct) < 3.0:
            return cached
        spot = self.session.get_spot_price(bc)
        r_api = self.session.get_option_chain(bc, expiry, "others", exc)
        if not r_api or not r_api.get("Success"):
            return cached or []
        T = time_to_expiry(expiry)
        r_rate = Config.RISK_FREE_RATE
        out = []
        for item in r_api["Success"]:
            strike = safe_float(item.get("strike_price", 0))
            rs = str(item.get("right", "")).strip().lower()
            ltp = safe_float(item.get("ltp", 0))
            if strike <= 0: continue
            right = OptionRight.CALL if "call" in rs else OptionRight.PUT
            if ltp > 0 and T > 0 and spot > 0:
                iv = BlackScholes.implied_vol(ltp, spot, strike, T, r_rate, right)
                g = BlackScholes.greeks(spot, strike, T, r_rate, iv, right)
            else:
                iv = 0; g = Greeks()
            out.append({
                "strike": int(strike),
                "right": "CALL" if right == OptionRight.CALL else "PUT",
                "ltp": round(ltp, 2),
                "bid": safe_float(item.get("best_bid_price", 0)),
                "ask": safe_float(item.get("best_offer_price", 0)),
                "iv": round(iv * 100, 2), "delta": round(g.delta, 4),
                "gamma": round(g.gamma, 6), "theta": round(g.theta, 2),
                "vega": round(g.vega, 2),
                "oi": safe_float(item.get("open_interest", 0)),
                "volume": safe_float(item.get("volume",
                                    item.get("total_quantity_traded", 0))),
            })
        out.sort(key=lambda x: (x["strike"], x["right"]))
        self.state.set_chain(out, time.time())
        return out

    # ── Helpers ──────────────────────────────────────────────

    def _leg(self, sid, bc, exc, strike, right, expiry, qty, sl):
        return Leg(strategy_id=sid, stock_code=bc, exchange_code=exc,
                   strike_price=strike, right=right, expiry_date=expiry,
                   side=OrderSide.SELL, quantity=qty, sl_percentage=sl,
                   status=LegStatus.ENTERING)

    def _delta_strikes(self, bc, exc, spot, expiry, target):
        r = self.session.get_option_chain(bc, expiry, "others", exc)
        if not r or not r.get("Success"):
            return None, None
        T = time_to_expiry(expiry)
        rr = Config.RISK_FREE_RATE
        best_ce, best_cd = None, 99.0
        best_pe, best_pd = None, 99.0
        for item in r["Success"]:
            strike = safe_float(item.get("strike_price", 0))
            rs = str(item.get("right", "")).lower()
            ltp = safe_float(item.get("ltp", 0))
            if strike <= 0 or ltp <= 0: continue
            if "call" in rs and strike > spot:
                iv = BlackScholes.implied_vol(ltp, spot, strike, T, rr, OptionRight.CALL)
                d = abs(abs(BlackScholes.delta(spot, strike, T, rr, iv, OptionRight.CALL)) - target)
                if d < best_cd: best_cd, best_ce = d, strike
            elif "put" in rs and strike < spot:
                iv = BlackScholes.implied_vol(ltp, spot, strike, T, rr, OptionRight.PUT)
                d = abs(abs(BlackScholes.delta(spot, strike, T, rr, iv, OptionRight.PUT)) - target)
                if d < best_pd: best_pd, best_pe = d, strike
        return best_ce, best_pe
