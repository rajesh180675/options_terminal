"""
main.py (Streamlit UI)

Tabs:
  1) Chain
  2) Limit Sell
  3) Positions (with manual exit)
  4) Broker (Funds / Orders / Trades / Positions / Recon)
  5) Analytics (Max Pain / PCR / Expected Move / Skew / Payoff / What-if)
  6) Journal (if journal.py exists)
  7) Backtest (if backtester.py exists)
  8) Logs

This UI is designed to be "flicker-free" using Streamlit containers and caching.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta

import streamlit as st
import pandas as pd

from app_config import Config, INSTRUMENTS
from models import StrategyStatus, LegStatus, OptionRight
from utils import breeze_expiry, next_weekly_expiry, is_expiry_day, atm_strike

st.set_page_config(
    page_title="Options Terminal",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Auto-refresh
try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=1500, limit=None, key="refresh")
except Exception:
    pass


@st.cache_resource
def boot_engine():
    from trading_engine import TradingEngine
    eng = TradingEngine()
    ok = eng.start()
    return eng, ok


engine, engine_ok = boot_engine()
state = engine.state


# ── CSS ──────────────────────────────────────────────────────
st.markdown("""
<style>
.pnl-pos { background: linear-gradient(135deg, #0d4d2b, #1a6b3f);
    color: #00ff88; padding: 12px 18px; border-radius: 10px;
    font-size: 26px; font-weight: 800; text-align: center; }
.pnl-neg { background: linear-gradient(135deg, #4d0d0d, #6b1a1a);
    color: #ff4444; padding: 12px 18px; border-radius: 10px;
    font-size: 26px; font-weight: 800; text-align: center; }
.badge { padding: 4px 12px; border-radius: 12px;
    font-size: 12px; font-weight: 700; display: inline-block; }
.bg { background: #0d4d2b; color: #00ff88; }
.br { background: #4d0d0d; color: #ff4444; }
.by { background: #4d4d0d; color: #ffff44; }
.bb { background: #0d2d4d; color: #44aaff; }
.logbox { font-family: 'Courier New', monospace; font-size: 11px;
    background: #0d1117; padding: 10px; border-radius: 8px;
    max-height: 420px; overflow-y: auto; border: 1px solid #21262d; }
</style>
""", unsafe_allow_html=True)


# ── HEADER ───────────────────────────────────────────────────
mtm = state.get_total_mtm()
pnl_cls = "pnl-pos" if mtm >= 0 else "pnl-neg"
pnl_sign = "+" if mtm >= 0 else ""

hdr = st.columns([3, 1, 1, 1, 1, 1])

hdr[0].markdown(f'<div class="{pnl_cls}">Total MTM: ₹{pnl_sign}{mtm:,.2f}</div>', unsafe_allow_html=True)

hdr[1].markdown(
    f'<span class="badge {"bg" if state.connected else "br"}">API {"✓" if state.connected else "✗"}</span>',
    unsafe_allow_html=True
)
hdr[2].markdown(
    f'<span class="badge {"bg" if state.ws_connected else "br"}">WS {"✓" if state.ws_connected else "✗"}</span>',
    unsafe_allow_html=True
)

dc = Config.breeze_code(Config.DEFAULT_STOCK)
spot = state.get_spot(dc)
hdr[3].markdown(f'<span class="badge by">{Config.DEFAULT_STOCK}: ₹{spot:,.1f}</span>', unsafe_allow_html=True)

# Margin badge
funds = engine.broker.get_funds() if getattr(engine, "broker", None) else None
margin_txt = f"₹{funds.free_margin:,.0f}" if funds else "N/A"
hdr[4].markdown(f'<span class="badge bb">Margin: {margin_txt}</span>', unsafe_allow_html=True)

mode_cls = "br" if Config.is_live() else "by"
hdr[5].markdown(f'<span class="badge {mode_cls}">{Config.TRADING_MODE.upper()}</span>', unsafe_allow_html=True)

st.divider()


# ── SIDEBAR ──────────────────────────────────────────────────
with st.sidebar:
    st.header("Deploy")

    inst_name = st.selectbox("Instrument", list(INSTRUMENTS.keys()), index=0)
    inst = Config.instrument(inst_name)
    breeze_code = inst["breeze_code"]
    exchange = inst["exchange"]

    exp_dt = next_weekly_expiry(inst_name)
    exp_str = breeze_expiry(exp_dt)

    wd = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    st.caption(f"Expiry: {exp_dt.strftime('%d-%b-%Y')} ({wd[inst['expiry_weekday']]})")
    st.caption(f"Exchange/Code: {exchange}/{breeze_code}")
    st.caption(f"Lot: {inst['lot_size']}  |  Strike gap: {inst['strike_gap']}")

    if is_expiry_day(inst_name):
        st.warning("Today is expiry day")

    lots = st.number_input("Lots", min_value=1, max_value=50, value=1)
    sl_pct = st.slider("SL %", min_value=10, max_value=200, value=int(Config.SL_PERCENTAGE))

    st.subheader("Short Straddle")
    if st.button("Deploy Straddle", use_container_width=True, type="primary"):
        with st.spinner("Deploying..."):
            r = engine.deploy_straddle(inst_name, exp_str, int(lots), float(sl_pct))
        if r:
            st.success(f"Deployed: {r.strategy_id}")
        else:
            st.error("Deployment failed")

    st.subheader("Short Strangle")
    tdelta = st.slider("Target Delta", 0.05, 0.40, 0.15, 0.01)
    if st.button("Deploy Strangle", use_container_width=True, type="primary"):
        with st.spinner("Deploying..."):
            r = engine.deploy_strangle(inst_name, float(tdelta), exp_str, int(lots), float(sl_pct))
        if r:
            st.success(f"Deployed: {r.strategy_id}")
        else:
            st.error("Deployment failed")

    st.subheader("Iron Condor")
    ic_delta = st.slider("IC Delta", 0.05, 0.30, 0.15, 0.01)
    ic_wing = st.number_input("Wing width (points)", value=int(inst["strike_gap"] * 2), step=int(inst["strike_gap"]))
    if st.button("Deploy Iron Condor", use_container_width=True):
        with st.spinner("Deploying..."):
            r = engine.deploy_iron_condor(inst_name, float(ic_delta), int(ic_wing), exp_str, int(lots), float(sl_pct))
        if r:
            st.success(f"Deployed: {r.strategy_id}")
        else:
            st.error("Deployment failed")

    st.subheader("Iron Butterfly")
    ib_wing = st.number_input("IB wing width (points)", value=int(inst["strike_gap"] * 3), step=int(inst["strike_gap"]))
    if st.button("Deploy Iron Butterfly", use_container_width=True):
        with st.spinner("Deploying..."):
            r = engine.deploy_iron_butterfly(inst_name, int(ib_wing), exp_str, int(lots), float(sl_pct))
        if r:
            st.success(f"Deployed: {r.strategy_id}")
        else:
            st.error("Deployment failed")

    st.divider()
    st.subheader("Kill Switch")
    confirm = st.checkbox("Confirm close ALL positions")
    if st.button("PANIC EXIT", use_container_width=True, type="primary", disabled=not confirm):
        engine.trigger_panic()
        st.error("Panic exit triggered")

    with st.expander("Config (non-sensitive)"):
        st.json(Config.dump())


# ── Tabs ─────────────────────────────────────────────────────
tabs = st.tabs(["Chain", "Limit Sell", "Positions", "Broker", "Analytics", "Journal", "Backtest", "Logs",
                "Scanner", "Dashboard", "Risk", "Strategies", "Calculator",
                "MTM Live", "IV Rank", "Theta", "Delta Hedge", "GEX", "Calendar",
                "OI Flow", "Trade Journal", "Monitor", "Payoff", "Margin", "Regime"])
tab_chain, tab_limit, tab_pos, tab_broker, tab_analytics, tab_journal, tab_backtest, tab_logs, \
    tab_scanner, tab_dashboard, tab_risk, tab_strategies, tab_calc, \
    tab_mtm, tab_iv_rank, tab_theta, tab_delta, tab_gex, tab_calendar, \
    tab_oi_flow, tab_trade_journal, tab_monitor, tab_payoff, tab_margin, tab_regime = tabs

# ── Engine extensions (new modules – non-breaking) ────────────
try:
    from engine_extensions import extend_engine
    if not hasattr(engine, "quick"):
        extend_engine(engine)
except Exception as _ext_err:
    pass


# ── Tab: Chain ───────────────────────────────────────────────
with tab_chain:
    st.subheader(f"Option Chain — {inst_name}")

    try:
        chain = engine.get_chain(inst_name, exp_str)
    except Exception as e:
        chain = []
        st.warning(f"Chain error: {e}")

    if not chain:
        st.info("Loading chain...")
    else:
        df = pd.DataFrame(chain)
        ce = df[df["right"] == "CALL"].rename(columns={
            "ltp": "CE LTP", "bid": "CE Bid", "ask": "CE Ask",
            "iv": "CE IV%", "delta": "CE Δ", "theta": "CE Θ",
            "vega": "CE V", "oi": "CE OI",
        })[["strike", "CE OI", "CE IV%", "CE Δ", "CE Θ", "CE V", "CE Bid", "CE LTP", "CE Ask"]]

        pe = df[df["right"] == "PUT"].rename(columns={
            "ltp": "PE LTP", "bid": "PE Bid", "ask": "PE Ask",
            "iv": "PE IV%", "delta": "PE Δ", "theta": "PE Θ",
            "vega": "PE V", "oi": "PE OI",
        })[["strike", "PE Bid", "PE LTP", "PE Ask", "PE Δ", "PE Θ", "PE V", "PE IV%", "PE OI"]]

        merged = pd.merge(ce, pe, on="strike", how="outer").sort_values("strike").fillna(0)

        spot_here = state.get_spot(breeze_code)
        if spot_here > 0:
            atm = atm_strike(spot_here, inst["strike_gap"])

            def hl(row):
                return ["background-color:#1a3a5c"] * len(row) if row["strike"] == atm else [""] * len(row)

            st.dataframe(merged.style.apply(hl, axis=1), use_container_width=True, height=520, hide_index=True)
        else:
            st.dataframe(merged, use_container_width=True, height=520, hide_index=True)


# ── Tab: Limit Sell ──────────────────────────────────────────
with tab_limit:
    st.subheader("Manual Limit Sell")

    # build strike choices from chain
    if "chain" in locals() and chain:
        strikes = sorted(set(c["strike"] for c in chain))
    else:
        center = int(spot) if spot > 0 else 24000
        gap = int(inst["strike_gap"])
        strikes = [center + i * gap for i in range(-10, 11)]

    c1, c2, c3 = st.columns(3)
    with c1:
        strike = st.selectbox("Strike", strikes, index=len(strikes)//2 if strikes else 0)
    with c2:
        right_str = st.radio("Type", ["CALL", "PUT"], horizontal=True)
        right = OptionRight.CALL if right_str == "CALL" else OptionRight.PUT
    with c3:
        l_lots = st.number_input("Lots", min_value=1, max_value=50, value=1, key="limit_lots")
        l_sl = st.slider("SL %", 10, 200, int(Config.SL_PERCENTAGE), key="limit_sl")

    # Default price from chain bid
    default_px = 100.0
    if chain:
        m = [c for c in chain if c["strike"] == strike and c["right"] == right_str]
        if m:
            default_px = float(m[0]["bid"] if m[0]["bid"] > 0 else m[0]["ltp"])
            mcols = st.columns(5)
            mcols[0].metric("Bid", f"₹{m[0]['bid']:.2f}")
            mcols[1].metric("LTP", f"₹{m[0]['ltp']:.2f}")
            mcols[2].metric("Ask", f"₹{m[0]['ask']:.2f}")
            mcols[3].metric("IV%", f"{m[0]['iv']:.1f}")
            mcols[4].metric("Δ", f"{m[0]['delta']:.4f}")

    limit_px = st.number_input("Limit price", min_value=0.05, value=float(round(default_px, 2)), step=0.05, format="%.2f")

    qty = inst["lot_size"] * int(l_lots)
    sl_val = round(limit_px * (1 + float(l_sl) / 100.0), 2)
    st.info(f"SELL {qty} × {breeze_code} {int(strike)} {right_str} @ ₹{limit_px:.2f} | SL ₹{sl_val:.2f}")

    # Margin check (if broker sync available)
    can_place = True
    if getattr(engine, "broker", None):
        try:
            ok_m, req_m, msg_m = engine.broker.check_margin(
                breeze_code, exchange, float(strike), right.value, exp_str, qty, float(limit_px), action="sell"
            )
            can_place = bool(ok_m)
            st.success(f"Margin: {msg_m}") if ok_m else st.error(f"Margin: {msg_m}")
        except Exception as e:
            st.warning(f"Margin check unavailable: {e}")

    if st.button("Place Limit Sell", use_container_width=True, type="primary", disabled=not can_place):
        with st.spinner("Placing order..."):
            r = engine.deploy_limit_sell(inst_name, float(strike), right, exp_str, int(l_lots), float(limit_px), float(l_sl))
        if r:
            st.success(f"Order placed: {r.strategy_id}")
        else:
            st.error("Order failed — check logs")


# ── Tab: Positions ───────────────────────────────────────────
with tab_pos:
    st.subheader("Active Strategies")

    strategies = state.get_strategies()
    if not strategies:
        st.info("No active strategies.")
    else:
        for s in strategies:
            pnl = s.compute_total_pnl()
            emoji = {
                StrategyStatus.ACTIVE: "🟢",
                StrategyStatus.DEPLOYING: "🟡",
                StrategyStatus.PARTIAL_EXIT: "🟠",
                StrategyStatus.CLOSED: "⚫",
                StrategyStatus.ERROR: "🔴",
            }.get(s.status, "⚪")

            with st.expander(
                f"{emoji} {s.strategy_type.value.replace('_',' ').upper()} [{s.strategy_id}]  "
                f"P&L: {'₹%+.2f' % pnl}",
                expanded=(s.status == StrategyStatus.ACTIVE),
            ):
                for leg in s.legs:
                    cols = st.columns([1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.5])
                    cols[0].write(f"**{leg.right.value[0].upper()}E**")
                    cols[1].write(f"K={int(leg.strike_price)}")
                    cols[2].write(f"Entry={leg.entry_price:.2f}")
                    cols[3].write(f"LTP={leg.current_price:.2f}")
                    cols[4].write(f"SL={leg.sl_price:.2f}{' (trail)' if leg.trailing_active else ''}")
                    cols[5].write(f"P&L={leg.pnl:+.0f}")
                    cols[6].write(f"Δ={leg.greeks.delta:.3f}")
                    cols[7].write(f"{leg.status.value}")

                    if leg.status == LegStatus.ACTIVE:
                        if st.button("Exit Leg (Market)", key=f"exit_{leg.leg_id}", type="primary"):
                            ok = engine.exit_leg(leg.leg_id)
                            st.success("Exited") if ok else st.error("Exit failed")
                            st.rerun()

                ng = s.net_greeks
                gcols = st.columns(4)
                gcols[0].metric("Net Δ", f"{ng.delta:+.1f}")
                gcols[1].metric("Net Γ", f"{ng.gamma:+.4f}")
                gcols[2].metric("Net Θ", f"₹{ng.theta:+.1f}")
                gcols[3].metric("Net V", f"{ng.vega:+.1f}")

    # Reconciliation issues if broker sync exists
    if getattr(engine, "broker", None):
        issues = engine.broker.get_recon_issues()
        if issues:
            st.divider()
            st.subheader("Reconciliation Issues")
            for x in issues:
                st.warning(x)


# ── Tab: Broker ──────────────────────────────────────────────
with tab_broker:
    st.subheader("Broker")

    if not getattr(engine, "broker", None):
        st.info("broker_sync.py not available in this build.")
    else:
        t_funds, t_orders, t_trades, t_positions, t_recon = st.tabs(["Funds", "Orders", "Trades", "Positions", "Recon"])

        with t_funds:
            st.markdown("Account funds")
            funds = engine.broker.get_funds()
            if funds:
                c = st.columns(4)
                c[0].metric("Allocated", f"₹{funds.allocated:,.0f}")
                c[1].metric("Utilized", f"₹{funds.utilized:,.0f}")
                c[2].metric("Free", f"₹{funds.free_margin:,.0f}")
                c[3].metric("Blocked", f"₹{funds.blocked:,.0f}")
            if st.button("Refresh funds"):
                engine.broker.fetch_funds()
                st.rerun()

            cust = engine.broker.get_customer_details()
            if cust:
                with st.expander("Customer details"):
                    st.json(cust)

        with t_orders:
            if st.button("Refresh orders"):
                engine.broker.fetch_orders("NFO")
                engine.broker.fetch_orders("BFO")
                st.rerun()

            orders = engine.broker.get_orders()
            if orders:
                odf = pd.DataFrame([{
                    "order_id": o.order_id,
                    "exc": o.exchange_code,
                    "stock": o.stock_code,
                    "strike": int(o.strike_price) if o.strike_price else "",
                    "type": (o.right[:1].upper() if o.right else ""),
                    "action": o.action,
                    "qty": o.quantity,
                    "price": o.price,
                    "avg": o.avg_price,
                    "status": o.status,
                    "time": o.order_time,
                } for o in orders])
                st.dataframe(odf, use_container_width=True, height=380, hide_index=True)

                pending = [o for o in orders if str(o.status).lower() in ("pending", "open")]
                if pending:
                    st.markdown("Pending order actions")
                    for o in pending[:10]:
                        c = st.columns([3, 1, 1, 1.5])
                        c[0].write(f"{o.order_id[-8:]} {o.exchange_code} {o.stock_code} {int(o.strike_price)} {o.action} @ {o.price}")
                        if c[1].button("Cancel", key=f"cancel_{o.order_id}"):
                            engine.broker.cancel_order_from_ui(o.order_id, o.exchange_code)
                            st.rerun()
                        new_px = c[2].number_input("New", value=float(o.price), step=0.05, key=f"new_{o.order_id}", label_visibility="collapsed")
                        if c[3].button("Modify", key=f"modify_{o.order_id}"):
                            engine.broker.modify_order_from_ui(o.order_id, float(new_px), o.exchange_code)
                            st.rerun()
            else:
                st.info("No orders available.")

        with t_trades:
            if st.button("Refresh trades"):
                engine.broker.fetch_trades("NFO")
                engine.broker.fetch_trades("BFO")
                st.rerun()

            trades = engine.broker.get_trades()
            if trades:
                tdf = pd.DataFrame([{
                    "trade_id": t.trade_id,
                    "order_id": t.order_id,
                    "exc": t.exchange_code,
                    "stock": t.stock_code,
                    "strike": int(t.strike_price) if t.strike_price else "",
                    "type": (t.right[:1].upper() if t.right else ""),
                    "action": t.action,
                    "qty": t.quantity,
                    "price": t.trade_price,
                    "time": t.trade_time,
                } for t in trades])
                st.dataframe(tdf, use_container_width=True, height=380, hide_index=True)
            else:
                st.info("No trades available.")

        with t_positions:
            if st.button("Refresh positions"):
                engine.broker.fetch_positions()
                st.rerun()

            pos = engine.broker.get_positions()
            if pos:
                pdf = pd.DataFrame([{
                    "stock": p.stock_code,
                    "exc": p.exchange_code,
                    "strike": int(p.strike_price),
                    "type": (p.right[:1].upper() if p.right else ""),
                    "qty": p.quantity,
                    "avg": p.avg_price,
                    "ltp": p.ltp,
                    "pnl": p.pnl,
                } for p in pos])
                st.dataframe(pdf, use_container_width=True, hide_index=True)
            else:
                st.info("No broker positions.")

        with t_recon:
            if st.button("Run reconciliation"):
                engine.broker.reconcile()
                st.rerun()
            issues = engine.broker.get_recon_issues()
            if issues:
                for x in issues:
                    st.warning(x)
            else:
                st.success("No reconciliation issues detected.")


# ── Tab: Analytics ───────────────────────────────────────────
with tab_analytics:
    st.subheader("Analytics")

    if not chain:
        st.info("Chain required for analytics.")
    else:
        # optional analytics module
        try:
            from analytics import (
                calculate_max_pain, calculate_pcr, calculate_expected_move,
                calculate_skew, generate_payoff, what_if_analysis, calculate_pop,
                greeks_pnl_attribution,
            )
            analytics_ok = True
        except Exception as e:
            analytics_ok = False
            st.warning(f"analytics.py not available: {e}")

        if analytics_ok:
            spot_here = state.get_spot(breeze_code)
            mp, curve = calculate_max_pain(chain)
            pcr = calculate_pcr(chain)

            c = st.columns(3)
            c[0].metric("Max Pain", f"{int(mp):,}")
            c[1].metric("PCR (OI)", f"{pcr['pcr_oi']:.2f}")
            c[2].metric("PCR (Vol)", f"{pcr['pcr_vol']:.2f}")

            if curve:
                pdf = pd.DataFrame(curve).set_index("strike")
                st.bar_chart(pdf["total"], height=180)

            # Expected move from ATM straddle
            if spot_here > 0:
                atm = atm_strike(spot_here, inst["strike_gap"])
                atm_ce = [x for x in chain if x["strike"] == atm and x["right"] == "CALL"]
                atm_pe = [x for x in chain if x["strike"] == atm and x["right"] == "PUT"]
                if atm_ce and atm_pe:
                    from greeks_engine import time_to_expiry
                    dte = time_to_expiry(exp_str) * 365.0
                    em = calculate_expected_move(spot_here, float(atm_ce[0]["ltp"]), float(atm_pe[0]["ltp"]), dte)
                    st.metric("Expected Move", f"±{em['expected_move']:,.0f} ({em['em_percent']:.1f}%)")
                    rcols = st.columns(2)
                    rcols[0].metric("Upper", f"{em['upper_range']:,.0f}")
                    rcols[1].metric("Lower", f"{em['lower_range']:,.0f}")

            st.divider()
            st.markdown("Volatility Skew")
            skew = calculate_skew(chain, spot_here if spot_here > 0 else 1.0)
            if skew:
                sdf = pd.DataFrame(skew)
                sdf_ce = sdf[sdf["right"] == "CALL"][["strike", "iv"]].rename(columns={"iv": "CE IV"})
                sdf_pe = sdf[sdf["right"] == "PUT"][["strike", "iv"]].rename(columns={"iv": "PE IV"})
                sm = pd.merge(sdf_ce, sdf_pe, on="strike", how="outer").set_index("strike")
                st.line_chart(sm, height=220)

            st.divider()
            st.markdown("Strategy Payoff / What-if")

            strategies = [s for s in state.get_strategies() if s.status == StrategyStatus.ACTIVE]
            if strategies:
                labels = [f"{s.strategy_id} | {s.strategy_type.value}" for s in strategies]
                choice = st.selectbox("Select strategy", labels)
                s = strategies[labels.index(choice)]

                pop = calculate_pop(s, spot_here)
                st.metric("Probability of Profit (approx)", f"{pop:.1f}%")

                pcols = st.columns(2)
                with pcols[0]:
                    st.markdown("Payoff at expiry")
                    payoff = generate_payoff(s, spot_here)
                    if payoff:
                        pf = pd.DataFrame(payoff).set_index("spot")
                        st.line_chart(pf["pnl"], height=220)
                with pcols[1]:
                    st.markdown("What-if (mark-to-model)")
                    wi = what_if_analysis(s, spot_here)
                    if wi:
                        wdf = pd.DataFrame(wi)
                        st.dataframe(wdf, use_container_width=True, hide_index=True, height=260)

                st.markdown("Greeks P&L attribution (small move)")
                attr = greeks_pnl_attribution(s, spot_change=max(spot_here * 0.005, 1.0))
                ac = st.columns(5)
                ac[0].metric("Δ", f"{attr['delta_pnl']:+.0f}")
                ac[1].metric("Γ", f"{attr['gamma_pnl']:+.0f}")
                ac[2].metric("Θ", f"{attr['theta_pnl']:+.0f}")
                ac[3].metric("V", f"{attr['vega_pnl']:+.0f}")
                ac[4].metric("Total", f"{attr['total_attributed']:+.0f}")
            else:
                st.info("No active strategies for payoff analysis.")

    # Adjustments (if adjustment engine active)
    adj = state.get_adjustments()
    if adj:
        st.divider()
        st.subheader("Adjustment Suggestions")
        for a in adj:
            sev = getattr(a, "severity", "INFO")
            msg = getattr(a, "message", "")
            det = getattr(a, "details", "")
            sug = getattr(a, "suggested_action", "")
            with st.expander(f"[{sev}] {msg}"):
                if det:
                    st.write(det)
                if sug:
                    st.info(sug)


# ── Tab: Journal ─────────────────────────────────────────────
with tab_journal:
    st.subheader("Journal")

    if not getattr(engine, "journal", None):
        st.info("journal.py not available in this build.")
    else:
        stats = engine.journal.get_stats()
        if stats.total_trades <= 0:
            st.info("No journal entries yet.")
        else:
            c = st.columns(4)
            c[0].metric("Trades", stats.total_trades)
            c[1].metric("Win rate", f"{stats.win_rate:.1f}%")
            c[2].metric("Profit factor", f"{stats.profit_factor:.2f}")
            c[3].metric("Total P&L", f"₹{stats.total_pnl:+,.0f}")

            c2 = st.columns(4)
            c2[0].metric("Max DD", f"₹{stats.max_drawdown:,.0f}")
            c2[1].metric("Avg win", f"₹{stats.avg_win:,.0f}")
            c2[2].metric("Avg loss", f"₹{-stats.avg_loss:,.0f}")
            c2[3].metric("Expectancy", f"₹{stats.expectancy:+,.0f}")

        st.divider()
        st.markdown("Recent entries")
        entries = engine.journal.get_entries(50)
        if entries:
            edf = pd.DataFrame([{
                "exit_date": e.exit_date[:19],
                "type": e.strategy_type,
                "stock": e.stock_code,
                "legs": e.legs_count,
                "pnl": e.pnl,
                "reason": e.exit_reason,
                "dur_min": e.duration_minutes,
            } for e in entries])
            st.dataframe(edf, use_container_width=True, hide_index=True, height=360)

        st.divider()
        st.markdown("Daily P&L")
        daily = engine.journal.get_daily_pnl(30)
        if daily:
            ddf = pd.DataFrame(daily)
            st.line_chart(ddf.set_index("date")["total"], height=200)
            st.dataframe(ddf, use_container_width=True, hide_index=True)


# ── Tab: Backtest ────────────────────────────────────────────
with tab_backtest:
    st.subheader("Backtest")

    if not getattr(engine, "backtester", None):
        st.info("backtester.py not available in this build.")
    else:
        bc1, bc2 = st.columns(2)
        with bc1:
            bt_inst = st.selectbox("Instrument", list(INSTRUMENTS.keys()), key="bt_inst")
            bt_strategy = st.selectbox("Strategy", ["strangle", "straddle"], key="bt_strategy")
            bt_delta = st.slider("Delta (strangle)", 0.05, 0.40, 0.15, 0.01, key="bt_delta")
            bt_sl = st.slider("SL %", 10, 200, 50, key="bt_sl")
        with bc2:
            bt_dte = st.slider("Entry DTE", 1, 15, 5, key="bt_dte")
            bt_iv = st.slider("IV assumption %", 5, 50, 15, key="bt_iv")
            bt_lots = st.number_input("Lots", 1, 10, 1, key="bt_lots")
            bt_days = st.slider("Lookback days", 30, 730, 365, key="bt_days")

        if st.button("Run backtest", type="primary", use_container_width=True):
            with st.spinner("Running..."):
                start = (datetime.now() - timedelta(days=int(bt_days))).strftime("%Y-%m-%d")
                res = engine.run_backtest(
                    instrument=bt_inst,
                    strategy=bt_strategy,
                    start_date=start,
                    end_date=datetime.now().strftime("%Y-%m-%d"),
                    entry_dte=int(bt_dte),
                    target_delta=float(bt_delta),
                    sl_pct=float(bt_sl),
                    iv_assumption=float(bt_iv) / 100.0,
                    lots=int(bt_lots),
                )

            # Render results (res is a dataclass in backtester.py)
            if getattr(res, "total_trades", 0) <= 0:
                st.warning("No trades generated.")
            else:
                c = st.columns(4)
                c[0].metric("Trades", res.total_trades)
                c[1].metric("Total P&L", f"₹{res.total_pnl:+,.0f}")
                c[2].metric("Win rate", f"{res.win_rate:.1f}%")
                c[3].metric("Max DD", f"₹{res.max_drawdown:,.0f}")

                st.markdown("Equity curve")
                st.line_chart(res.equity_curve, height=250)

                if res.trades:
                    tdf = pd.DataFrame([{
                        "entry": t.entry_date,
                        "exit": t.exit_date,
                        "entry_spot": t.entry_spot,
                        "exit_spot": t.exit_spot,
                        "ce_k": int(t.ce_strike),
                        "pe_k": int(t.pe_strike),
                        "pnl": t.pnl,
                        "reason": t.exit_reason,
                    } for t in res.trades])
                    st.dataframe(tdf, use_container_width=True, hide_index=True, height=380)


# ── Tab: Logs ────────────────────────────────────────────────
with tab_logs:
    st.subheader("Logs")

    c1, c2 = st.columns([3, 2])
    with c1:
        logs = state.get_logs(200)
        if logs:
            lines = []
            for e in reversed(logs):
                color = {"INFO": "#88ccff", "WARN": "#ffcc44", "ERROR": "#ff4444", "CRIT": "#ff0000"}.get(e.level, "#cccccc")
                lines.append(
                    f'<span style="color:{color}">[{e.timestamp}] {e.level:5s}</span>'
                    f' | <span style="color:#888">{e.source:10s}</span>'
                    f' | {e.message}'
                )
            st.markdown('<div class="logbox">' + "<br>".join(lines) + "</div>", unsafe_allow_html=True)
        else:
            st.info("No logs yet.")

    with c2:
        st.markdown("Order Log (DB)")
        try:
            ol = engine.db.get_recent_logs(50) if hasattr(engine.db, "get_recent_logs") else engine.db.get_recent_order_logs(50)
            if ol:
                odf = pd.DataFrame(ol)
                cols = [c for c in ["timestamp", "action", "status", "price", "quantity", "order_id", "message"] if c in odf.columns]
                st.dataframe(odf[cols], use_container_width=True, hide_index=True, height=420)
            else:
                st.info("No DB order logs.")
        except Exception as e:
            st.warning(f"Order log error: {e}")


# ════════════════════════════════════════════════════════════════
# ══  NEW TABS (non-breaking additions below existing code)  ════
# ════════════════════════════════════════════════════════════════

# ── Tab: Scanner ─────────────────────────────────────────────
with tab_scanner:
    st.subheader("Market Scanner")

    if not chain:
        st.info("Option chain required for scanning. Load the Chain tab first.")
    else:
        try:
            from scanner import MarketScanner
            scanner_engine = MarketScanner()

            scan_cols = st.columns([2, 1, 1])
            with scan_cols[0]:
                scan_types = st.multiselect(
                    "Scan Types",
                    ["High IV", "Unusual OI", "Gamma Concentration", "Theta Rich", "Skew Outlier", "Max Pain Deviation", "Illiquidity"],
                    default=["High IV", "Unusual OI", "Gamma Concentration"],
                )
            with scan_cols[1]:
                auto_scan = st.checkbox("Auto-refresh with chain", value=True)
            with scan_cols[2]:
                if st.button("Run Scan Now", type="primary", use_container_width=True):
                    st.session_state["last_scan"] = time.time()

            results = scanner_engine.run_all(chain, spot if spot > 0 else state.get_spot(breeze_code),
                                              instrument=inst_name, expiry=exp_str)

            # Signal Engine overlay
            try:
                from signal_engine import SignalEngine
                sig_engine = SignalEngine()
                signals = sig_engine.run_all_signals(chain, state.get_spot(breeze_code),
                                                     inst_name, exp_str)
            except Exception:
                signals = []

            if signals:
                st.markdown("#### 📡 Market Signals")
                sig_cols = st.columns(min(len(signals), 4))
                for i, sig in enumerate(signals[:4]):
                    color = {"OPPORTUNITY": "#00c853", "WARN": "#ff6d00", "INFO": "#1976d2"}.get(sig.severity, "#90a4ae")
                    sig_cols[i % 4].markdown(
                        f"""<div style="background:{color}20; border-left:4px solid {color};
                            padding:10px; border-radius:8px; margin:4px;">
                            <b>{sig.signal_type.replace('_',' ')}</b><br>
                            <span style="font-size:18px; font-weight:800;">{sig.label}</span>
                            <span style="font-size:12px; color:{color};"> {sig.value}</span><br>
                            <span style="font-size:11px; color:#aaa;">{sig.description}</span><br>
                            <span style="font-size:11px; color:#ccc; font-style:italic;">{sig.action}</span>
                        </div>""",
                        unsafe_allow_html=True,
                    )

            st.divider()
            st.markdown("#### 🔍 Scan Results")

            if results:
                sev_colors = {"ALERT": "#d50000", "WARN": "#ff6d00", "INFO": "#1976d2"}
                for r in results:
                    rd = r.to_dict()
                    col = sev_colors.get(rd["severity"], "#90a4ae")
                    with st.expander(f"[{rd['severity']}] {rd['scan_type']} | {rd['description'][:60]}...",
                                     expanded=(rd["severity"] == "ALERT")):
                        dc = st.columns(4)
                        dc[0].metric("Type", rd["scan_type"])
                        dc[1].metric("Strike", rd["strike"] or "—")
                        dc[2].metric("Right", rd["right"] or "—")
                        dc[3].metric("Value", f"{rd['value']:.2f}")
                        st.markdown(
                            f'<div style="border-left:4px solid {col}; padding:8px; background:{col}15; border-radius:4px;">'
                            f'{rd["description"]}</div>',
                            unsafe_allow_html=True
                        )
            else:
                st.success("No scan alerts at the moment.")

            # OI Change tracker (stored in session state)
            st.divider()
            st.markdown("#### 📊 OI Change Tracker")
            if "oi_prev_chain" not in st.session_state:
                st.session_state["oi_prev_chain"] = chain
                st.info("Baseline OI snapshot taken. Refresh to see changes.")
            else:
                oi_changes = scanner_engine.oi_change_analysis(chain, st.session_state["oi_prev_chain"], inst_name)
                if oi_changes:
                    oi_df = pd.DataFrame([c.to_dict() for c in oi_changes])
                    st.dataframe(oi_df, use_container_width=True, hide_index=True)
                else:
                    st.info("No significant OI changes detected since last snapshot.")
                if st.button("Reset OI baseline"):
                    st.session_state["oi_prev_chain"] = chain
                    st.success("OI baseline reset.")

        except Exception as e:
            st.warning(f"Scanner error: {e}")


# ── Tab: Dashboard ───────────────────────────────────────────
with tab_dashboard:
    st.subheader("Portfolio Dashboard")

    try:
        from portfolio_dashboard import PortfolioDashboard
        if not hasattr(engine, "dashboard") or engine.dashboard is None:
            engine.dashboard = PortfolioDashboard()

        strategies_all = state.get_strategies()
        spot_map_ui = state.get_all_spots()
        spot_ui = spot if spot > 0 else state.get_spot(breeze_code)

        # Portfolio summary
        summary = engine.dashboard.build_portfolio_summary(strategies_all, spot_map_ui)

        # ── KPI strip
        kpi_cols = st.columns(5)
        kpi_cols[0].metric("Active Strategies", summary.active_strategies)
        kpi_cols[1].metric("Active Legs", summary.active_legs)
        kpi_cols[2].metric("Total Premium", f"₹{summary.total_premium:,.0f}")
        kpi_cols[3].metric("Daily Theta", f"₹{summary.total_theta:,.1f}")
        kpi_cols[4].metric("Portfolio Health", f"{summary.portfolio_health:.0f}/100")

        # MTM live chart
        mtm_hist = state.get_mtm_history()
        if mtm_hist:
            st.markdown("#### 📈 Intraday MTM")
            spark_data = engine.dashboard.get_mtm_sparkline_data(mtm_hist)
            if spark_data:
                spark_df = pd.DataFrame(spark_data).set_index("time")
                st.area_chart(spark_df["MTM"], height=180, use_container_width=True)

        # Greeks gauges
        st.markdown("#### ⚡ Portfolio Greeks")
        gauges = engine.dashboard.get_greeks_gauges(summary)
        g_cols = st.columns(4)
        for i, g in enumerate(gauges):
            val = g["value"]
            desc = g["description"]
            g_cols[i].metric(g["name"], f"{val:+.2f} {g['unit']}", help=desc)

        # Win/Loss ratio
        st.markdown("#### 🎯 Strategy Scorecard")
        sc_cols = st.columns(4)
        sc_cols[0].metric("In Profit", summary.strategies_in_profit, delta=None)
        sc_cols[1].metric("In Loss", summary.strategies_in_loss, delta=None)
        sc_cols[2].metric("Biggest Winner", summary.biggest_winner[:6] if summary.biggest_winner else "—")
        sc_cols[3].metric("Biggest Loser", summary.biggest_loser[:6] if summary.biggest_loser else "—")

        # Strategy health cards
        st.markdown("#### 🃏 Strategy Health Cards")
        active_strats = [s for s in strategies_all if s.status.value in ("active", "partial_exit")]
        if active_strats:
            for s in active_strats:
                card = engine.dashboard.build_strategy_card(s, spot_ui)
                if not card:
                    continue
                emoji = {"Excellent": "🟢", "Good": "🟡", "Caution": "🟠", "Warning": "🔴", "Critical": "💥"}.get(card.health_label, "⚪")
                with st.expander(
                    f"{emoji} {card.strategy_type.replace('_',' ').upper()} [{card.strategy_id}]  "
                    f"P&L: ₹{card.pnl:+,.0f}  Health: {card.health_label} ({card.health_score:.0f})",
                    expanded=True
                ):
                    cc = st.columns(5)
                    cc[0].metric("Premium", f"₹{card.total_premium:,.0f}")
                    cc[1].metric("P&L %", f"{card.pnl_pct:+.1f}%")
                    cc[2].metric("DTE", f"{card.dte:.1f}d")
                    cc[3].metric("BE Upper", f"{card.breakeven_upper:,.0f}")
                    cc[4].metric("BE Lower", f"{card.breakeven_lower:,.0f}")

                    gc2 = st.columns(4)
                    gc2[0].metric("Net Δ", f"{card.net_delta:+.2f}")
                    gc2[1].metric("Net Θ/day", f"₹{card.net_theta:+.2f}")
                    gc2[2].metric("Margin (est)", f"₹{card.margin_utilised:,.0f}")
                    gc2[3].metric("Θ/Margin%", f"{card.theta_margin_ratio:.3f}%")

                    if card.notes:
                        for note in card.notes:
                            st.caption(note)
        else:
            st.info("No active strategies.")

        # Margin utilisation
        if summary.total_margin > 0:
            st.divider()
            st.markdown("#### 💰 Margin Utilisation")
            funds_obj = engine.broker.get_funds() if getattr(engine, "broker", None) else None
            total_avail = float(funds_obj.allocated) if funds_obj else 0
            if total_avail > 0:
                util_pct = summary.total_margin / total_avail * 100
                st.progress(min(int(util_pct), 100), text=f"Estimated margin used: {util_pct:.0f}% (₹{summary.total_margin:,.0f} / ₹{total_avail:,.0f})")
            else:
                st.caption(f"Estimated margin in use: ₹{summary.total_margin:,.0f}")

    except Exception as e:
        st.warning(f"Dashboard error: {e}")
        import traceback
        st.code(traceback.format_exc())


# ── Tab: Risk ────────────────────────────────────────────────
with tab_risk:
    st.subheader("Risk Matrix")

    try:
        from risk_matrix import RiskMatrix
        if not hasattr(engine, "risk_matrix_engine") or engine.risk_matrix_engine is None:
            engine.risk_matrix_engine = RiskMatrix()

        strategies_all = state.get_strategies()
        spot_map_ui = state.get_all_spots()

        active_strats = [s for s in strategies_all if s.status.value in ("active", "partial_exit")]
        if not active_strats:
            st.info("No active strategies for risk analysis.")
        else:
            rm_cols = st.columns(2)
            with rm_cols[0]:
                days_fwd = st.slider("Days forward for P&L repricing", 0, 10, 1, key="rm_days")
            with rm_cols[1]:
                if st.button("Compute Risk Matrix", type="primary", use_container_width=True):
                    with st.spinner("Computing scenarios..."):
                        rm_result = engine.risk_matrix_engine.build_risk_matrix(
                            active_strats, spot_map_ui, float(days_fwd)
                        )
                        st.session_state["rm_result"] = rm_result

            if "rm_result" in st.session_state:
                rm = st.session_state["rm_result"]

                # Summary metrics
                mc = st.columns(4)
                mc[0].metric("Worst Case", f"₹{rm.worst_case:,.0f}")
                mc[1].metric("Best Case", f"₹{rm.best_case:,.0f}")
                mc[2].metric("1-Day VaR (95%)", f"₹{rm.var_95:,.0f}")
                mc[3].metric("Worst/Margin", f"{rm.max_margin_ratio:.1%}")

                st.caption(engine.risk_matrix_engine.get_worst_scenario_description(rm))
                st.caption(engine.risk_matrix_engine.get_best_scenario_description(rm))

                # Matrix table
                st.markdown("#### P&L Matrix (rows = IV Δ pp | cols = Spot %)")
                matrix_rows = engine.risk_matrix_engine.get_matrix_as_dataframe_dicts(rm)
                if matrix_rows:
                    mdf = pd.DataFrame(matrix_rows)
                    # Color-code: green = positive, red = negative
                    def color_pnl(val):
                        if isinstance(val, (int, float)):
                            if val > 0:
                                intensity = min(int(val / 5000 * 80), 80)
                                return f"background-color: rgba(0,200,100,{intensity/100}); color: white"
                            elif val < 0:
                                intensity = min(int(abs(val) / 5000 * 80), 80)
                                return f"background-color: rgba(255,50,50,{intensity/100}); color: white"
                        return ""
                    styled = mdf.set_index("IV Δ (pp)").style.map(color_pnl)
                    st.dataframe(styled, use_container_width=True, hide_index=False)

                # Stress test
                st.divider()
                st.markdown("#### ⚠️ Stress Test Scenarios")
                with st.spinner("Running stress tests..."):
                    stress = engine.risk_matrix_engine.stress_test_portfolio(active_strats, spot_map_ui)

                s_cols = st.columns(4)
                s_cols[0].metric("Spot +5%", f"₹{stress.spot_up_5pct:+,.0f}")
                s_cols[1].metric("Spot -5%", f"₹{stress.spot_down_5pct:+,.0f}")
                s_cols[2].metric("Spot +10%", f"₹{stress.spot_up_10pct:+,.0f}")
                s_cols[3].metric("Spot -10%", f"₹{stress.spot_down_10pct:+,.0f}")

                s_cols2 = st.columns(4)
                s_cols2[0].metric("IV +5pp", f"₹{stress.iv_up_5pts:+,.0f}")
                s_cols2[1].metric("IV -5pp", f"₹{stress.iv_down_5pts:+,.0f}")
                s_cols2[2].metric("1-Day Decay", f"₹{stress.one_day_decay:+,.0f}")
                s_cols2[3].metric("Expiry Now", f"₹{stress.expiry_day:+,.0f}")

            # Portfolio Payoff
            st.divider()
            st.markdown("#### 📉 Combined Portfolio Payoff at Expiry")
            with st.spinner("Computing payoff..."):
                payoff_data = engine.risk_matrix_engine.compute_portfolio_payoff(active_strats, spot_map_ui)
            if payoff_data:
                pdf = pd.DataFrame(payoff_data).set_index("spot")
                st.line_chart(pdf, height=280, use_container_width=True)
            else:
                st.info("No payoff data available.")

    except Exception as e:
        st.warning(f"Risk matrix error: {e}")
        import traceback
        st.code(traceback.format_exc())


# ── Tab: Strategies ──────────────────────────────────────────
with tab_strategies:
    st.subheader("Quick Strategy Templates")

    if not getattr(engine, "quick", None):
        try:
            from engine_extensions import extend_engine
            extend_engine(engine)
        except Exception:
            pass

    st.info("ℹ️ These are additional strategy templates layered on top of the original engine. "
            "All existing strategies (straddle, strangle, condor, butterfly) remain unchanged in the sidebar.")

    try:
        from quick_strategies import QuickStrategyEngine

        # Strategy info cards
        if getattr(engine, "quick", None):
            infos = QuickStrategyEngine.get_strategy_info()
        else:
            infos = []

        # Display strategy tiles
        st.markdown("#### Available Templates")
        if infos:
            tile_cols = st.columns(3)
            for i, info in enumerate(infos):
                with tile_cols[i % 3]:
                    st.markdown(
                        f"""<div style="border:1px solid #333; padding:12px; border-radius:10px; margin:4px; background:#161b22;">
                            <h4>{info['emoji']} {info['name']}</h4>
                            <b>Bias:</b> {info['bias']}<br>
                            <b>Risk:</b> {info['risk']}<br>
                            <b>Legs:</b> <code>{info['legs']}</code><br>
                            <b>Best for:</b> {info['ideal']}<br>
                            <hr style="border-color:#333;">
                            <b>Max Profit:</b> {info['max_profit']}<br>
                            <b>Max Loss:</b> {info['max_loss']}
                        </div>""",
                        unsafe_allow_html=True,
                    )

        st.divider()

        # Deploy panel
        st.markdown("#### Deploy Quick Strategy")
        qs_cols = st.columns([2, 1, 1, 1])
        with qs_cols[0]:
            qs_type = st.selectbox("Strategy", [
                "Bull Put Spread", "Bear Call Spread",
                "Jade Lizard", "Put Ratio Spread (1×2)", "Synthetic Short Futures",
            ])
        with qs_cols[1]:
            qs_inst = st.selectbox("Instrument", list(INSTRUMENTS.keys()), key="qs_inst")
        with qs_cols[2]:
            qs_lots = st.number_input("Lots", 1, 20, 1, key="qs_lots")
        with qs_cols[3]:
            qs_sl = st.slider("SL %", 10, 200, 50, key="qs_sl")

        # Strategy-specific params
        qs_inst_data = Config.instrument(qs_inst)
        qs_gap = qs_inst_data["strike_gap"]

        if qs_type in ("Bull Put Spread", "Bear Call Spread"):
            sp_cols = st.columns(2)
            with sp_cols[0]:
                qs_delta = st.slider("Short leg delta", 0.10, 0.45, 0.30, 0.05, key="qs_delta")
            with sp_cols[1]:
                qs_wing = st.number_input("Wing width (pts)", value=int(qs_gap * 2), step=int(qs_gap), key="qs_wing")

        elif qs_type == "Jade Lizard":
            jl_cols = st.columns(2)
            with jl_cols[0]:
                qs_put_delta = st.slider("Put delta", 0.15, 0.40, 0.30, 0.05, key="qs_put_delta")
            with jl_cols[1]:
                qs_call_delta = st.slider("Call spread delta", 0.10, 0.35, 0.20, 0.05, key="qs_call_delta")

        elif qs_type == "Put Ratio Spread (1×2)":
            qs_sell_delta = st.slider("Sell delta", 0.25, 0.50, 0.40, 0.05, key="qs_ratio_delta")

        if st.button(f"Deploy {qs_type}", type="primary", use_container_width=True):
            if not getattr(engine, "quick", None):
                st.error("Quick strategy engine not initialised.")
            else:
                with st.spinner("Deploying..."):
                    result = None
                    qs_expiry = exp_str  # use currently selected expiry

                    if qs_type == "Bull Put Spread":
                        result = engine.quick.deploy_bull_put_spread(
                            qs_inst, qs_expiry, qs_lots, qs_sl, qs_delta, qs_wing
                        )
                    elif qs_type == "Bear Call Spread":
                        result = engine.quick.deploy_bear_call_spread(
                            qs_inst, qs_expiry, qs_lots, qs_sl, qs_delta, qs_wing
                        )
                    elif qs_type == "Jade Lizard":
                        result = engine.quick.deploy_jade_lizard(
                            qs_inst, qs_expiry, qs_lots, qs_sl, qs_put_delta, qs_call_delta
                        )
                    elif qs_type == "Put Ratio Spread (1×2)":
                        result = engine.quick.deploy_put_ratio_spread(
                            qs_inst, qs_expiry, qs_lots, qs_sell_delta
                        )
                    elif qs_type == "Synthetic Short Futures":
                        result = engine.quick.deploy_synthetic_short(qs_inst, qs_expiry, qs_lots)

                if result:
                    st.success(f"✅ Deployed: {result.strategy_id} ({result.status.value})")
                else:
                    st.error("Deployment failed — check Logs tab.")

        # Pre-trade checklist
        st.divider()
        st.markdown("#### ✅ Pre-Trade Checklist")
        if chain and state.get_spot(breeze_code) > 0:
            try:
                from signal_engine import SignalEngine
                sig_eng = SignalEngine()
                funds_obj = engine.broker.get_funds() if getattr(engine, "broker", None) else None
                avail_margin = float(funds_obj.free_margin) if funds_obj else 0.0

                checklist = sig_eng.pre_trade_checklist(
                    chain,
                    state.get_spot(breeze_code),
                    inst_name,
                    qs_type,
                    exp_str,
                    lots=int(qs_lots),
                    available_margin=avail_margin,
                )
                verdict_colors = {"GO": "#00c853", "CAUTION": "#ff6d00", "NO-GO": "#d50000"}
                vc = verdict_colors.get(checklist.go_nogo, "#90a4ae")
                st.markdown(
                    f'<div style="border:2px solid {vc}; border-radius:10px; padding:14px; background:{vc}15;">'
                    f'<h3 style="color:{vc};">{checklist.go_nogo}</h3>'
                    f'<p>{checklist.summary}</p></div>',
                    unsafe_allow_html=True,
                )
                st.write("")
                for chk in checklist.checks:
                    st.markdown(f"{chk['status']} **{chk['check']}**: {chk['detail']} *(score: {chk['score']:.0f})*")
            except Exception as e:
                st.warning(f"Checklist error: {e}")
        else:
            st.info("Load the option chain first for pre-trade checklist.")

    except Exception as e:
        st.warning(f"Quick strategies error: {e}")
        import traceback
        st.code(traceback.format_exc())


# ── Tab: Calculator ──────────────────────────────────────────
with tab_calc:
    st.subheader("Options Calculator")

    try:
        from options_calculator import OptionsCalculator, CustomLeg
        calc = OptionsCalculator()

        calc_mode = st.radio(
            "Mode",
            ["Option Pricer", "Greeks Ladder", "Decay Table", "Spread Builder", "Roll Analyser"],
            horizontal=True,
        )

        current_spot = state.get_spot(breeze_code)
        default_spot = float(current_spot) if current_spot > 0 else 24000.0

        # ── Option Pricer ──────────────────────────────────
        if calc_mode == "Option Pricer":
            pc = st.columns([1, 1, 1, 1, 1])
            with pc[0]:
                calc_spot = st.number_input("Spot", value=default_spot, step=50.0, key="c_spot")
            with pc[1]:
                calc_strike = st.number_input("Strike", value=float(int(default_spot / 50) * 50), step=50.0, key="c_strike")
            with pc[2]:
                calc_right = st.radio("Type", ["CALL", "PUT"], key="c_right")
            with pc[3]:
                calc_iv = st.number_input("IV %", value=15.0, min_value=1.0, max_value=200.0, step=0.5, key="c_iv") / 100
            with pc[4]:
                calc_dte = st.number_input("DTE (days)", value=7.0, min_value=0.1, max_value=365.0, step=1.0, key="c_dte")

            if st.button("Calculate", type="primary"):
                res = calc.price_option(calc_spot, calc_strike, calc_iv, calc_dte, calc_right)
                rdict = res.to_dict()
                rc = st.columns(4)
                rc[0].metric("Price", f"₹{res.price}")
                rc[1].metric("Intrinsic", f"₹{res.intrinsic}")
                rc[2].metric("Time Value", f"₹{res.time_value}")
                rc[3].metric("Breakeven", f"{res.breakeven_at_expiry:,.1f}")
                rc2 = st.columns(4)
                rc2[0].metric("Delta", f"{res.delta}")
                rc2[1].metric("Gamma", f"{res.gamma}")
                rc2[2].metric("Theta/day", f"₹{res.theta}")
                rc2[3].metric("Vega (per 1% IV)", f"₹{res.vega}")

        # ── Greeks Ladder ──────────────────────────────────
        elif calc_mode == "Greeks Ladder":
            lc = st.columns(4)
            l_spot = lc[0].number_input("Center Spot", value=default_spot, step=50.0, key="l_spot")
            l_strike = lc[1].number_input("Strike", value=float(int(default_spot / 50) * 50), step=50.0, key="l_strike")
            l_iv = lc[2].number_input("IV %", value=15.0, step=0.5, key="l_iv") / 100
            l_dte = lc[3].number_input("DTE", value=7.0, step=1.0, key="l_dte")
            l_right = st.radio("Type", ["CALL", "PUT"], horizontal=True, key="l_right")

            if st.button("Build Ladder", type="primary"):
                ladder = calc.greeks_ladder(l_spot, l_strike, l_iv, l_dte, l_right)
                ldf = pd.DataFrame(OptionsCalculator.greeks_ladder_to_dicts(ladder))
                # Highlight current spot row
                def hl_spot(row):
                    if abs(row["Spot"] - l_spot) < 50:
                        return ["background-color:#1a3a5c"] * len(row)
                    return [""] * len(row)
                st.dataframe(ldf.style.apply(hl_spot, axis=1), use_container_width=True, hide_index=True, height=420)

        # ── Decay Table ────────────────────────────────────
        elif calc_mode == "Decay Table":
            dc = st.columns(4)
            d_spot = dc[0].number_input("Spot", value=default_spot, step=50.0, key="d_spot")
            d_strike = dc[1].number_input("Strike", value=float(int(default_spot / 50) * 50), step=50.0, key="d_strike")
            d_iv = dc[2].number_input("IV %", value=15.0, step=0.5, key="d_iv") / 100
            d_max_dte = dc[3].number_input("Max DTE", value=30.0, step=5.0, key="d_maxdte")
            d_right = st.radio("Type", ["CALL", "PUT"], horizontal=True, key="d_right")

            if st.button("Build Decay Table", type="primary"):
                decay = calc.decay_table(d_spot, d_strike, d_iv, d_max_dte, d_right)
                ddf = pd.DataFrame(OptionsCalculator.decay_table_to_dicts(decay))
                st.dataframe(ddf, use_container_width=True, hide_index=True)
                # Decay curve
                st.line_chart(ddf.set_index("DTE")["Price"].sort_index(), height=220)

        # ── Spread Builder ─────────────────────────────────
        elif calc_mode == "Spread Builder":
            st.markdown("Build a custom spread (up to 4 legs).")
            sb_cols = st.columns(4)
            sb_spot = sb_cols[0].number_input("Spot", value=default_spot, step=50.0, key="sb_spot")
            sb_iv = sb_cols[1].number_input("IV % (for pricing)", value=15.0, step=0.5, key="sb_iv") / 100
            sb_dte = sb_cols[2].number_input("DTE", value=7.0, step=1.0, key="sb_dte")
            sb_legs_n = sb_cols[3].number_input("Number of legs", 1, 4, 2, key="sb_n")

            legs_input = []
            for i in range(int(sb_legs_n)):
                lc = st.columns(5)
                k = lc[0].number_input(f"Leg {i+1} Strike", value=float(int(default_spot / 50) * 50 + (i - int(sb_legs_n) // 2) * 100), step=50.0, key=f"sb_k{i}")
                r = lc[1].radio(f"Leg {i+1} Type", ["CALL", "PUT"], horizontal=True, key=f"sb_r{i}")
                s = lc[2].radio(f"Leg {i+1} Side", ["SELL", "BUY"], horizontal=True, key=f"sb_s{i}")
                q = lc[3].number_input(f"Leg {i+1} Qty", 1, 200, 75, key=f"sb_q{i}")
                ep = lc[4].number_input(f"Leg {i+1} Entry px (0=model)", value=0.0, step=0.5, key=f"sb_ep{i}")
                legs_input.append(CustomLeg(strike=k, right=r, side=s, quantity=q, entry_price=ep, lots=1))

            if st.button("Analyze Spread", type="primary"):
                analysis = calc.analyze_spread(legs_input, sb_spot, sb_dte, sb_iv)
                ac = st.columns(4)
                credit_label = "Credit" if analysis.net_premium > 0 else "Debit"
                ac[0].metric(f"Net {credit_label}", f"₹{abs(analysis.net_premium):,.0f}")
                ac[1].metric("Max Profit", f"₹{analysis.max_profit:,.0f}")
                ac[2].metric("Max Loss", f"₹{analysis.max_loss:,.0f}")
                ac[3].metric("POP (approx)", f"{analysis.pop_estimate:.0f}%")

                bec = st.columns(2)
                bec[0].metric("BE Lower", f"{analysis.breakeven_lower:,.1f}")
                bec[1].metric("BE Upper", f"{analysis.breakeven_upper:,.1f}")

                if analysis.payoff_curve:
                    pdf = pd.DataFrame(analysis.payoff_curve).set_index("spot")
                    st.line_chart(pdf["pnl"], height=240, use_container_width=True)

        # ── Roll Analyser ──────────────────────────────────
        elif calc_mode == "Roll Analyser":
            st.markdown("Analyse cost/benefit of rolling a position to a different strike or expiry.")
            rc1, rc2 = st.columns(2)
            with rc1:
                st.markdown("**Current Position**")
                curr_px = st.number_input("Current market price", value=100.0, step=1.0, key="roll_curr_px")
                curr_dte = st.number_input("Current DTE", value=3.0, step=1.0, key="roll_curr_dte")
                curr_strike = st.number_input("Current strike", value=float(int(default_spot / 50) * 50), step=50.0, key="roll_curr_k")
            with rc2:
                st.markdown("**Target Position**")
                tgt_dte = st.number_input("Target DTE", value=10.0, step=1.0, key="roll_tgt_dte")
                tgt_strike = st.number_input("Target strike", value=float(int(default_spot / 50) * 50 + 100), step=50.0, key="roll_tgt_k")
                roll_iv = st.number_input("IV % assumption", value=15.0, step=0.5, key="roll_iv") / 100
            roll_right = st.radio("Option type", ["CALL", "PUT"], horizontal=True, key="roll_type")
            roll_spot = st.number_input("Current spot", value=default_spot, step=50.0, key="roll_spot")

            if st.button("Analyse Roll", type="primary"):
                roll_result = calc.roll_analysis(
                    curr_px, 0.0, curr_dte, tgt_dte, roll_spot,
                    curr_strike, tgt_strike, roll_right, roll_iv
                )
                rr_cols = st.columns(4)
                rr_cols[0].metric("Roll Type", roll_result["roll_type"])
                rr_cols[1].metric("Roll Cost", f"₹{roll_result['roll_cost']:.2f}")
                rr_cols[2].metric("Target Price", f"₹{roll_result['target_price']:.2f}")
                rr_cols[3].metric("New Breakeven", f"{roll_result['new_breakeven']:,.1f}")
                if roll_result["roll_worthwhile"]:
                    st.success("Roll creates a credit — generally worthwhile.")
                else:
                    st.warning("Roll creates a debit — evaluate if new position is worth the cost.")

    except Exception as e:
        st.warning(f"Calculator error: {e}")
        import traceback
        st.code(traceback.format_exc())


# ════════════════════════════════════════════════════════════════
# ══  LAYER 2 NEW TABS  ══════════════════════════════════════════
# ════════════════════════════════════════════════════════════════

# ── Tab: MTM Live ─────────────────────────────────────────────
with tab_mtm:
    st.subheader("MTM Live Analytics")

    try:
        from mtm_live import MTMLiveEngine
        if not hasattr(engine, "mtm_live") or engine.mtm_live is None:
            engine.mtm_live = MTMLiveEngine()

        mtm_history = state.get_mtm_history()

        if not mtm_history or len(mtm_history) < 2:
            st.info("Accumulating MTM data... check back in a few seconds.")
        else:
            # Count active lots
            active_strategies = [s for s in state.get_strategies() if s.status.value in ("active", "partial_exit")]
            active_lots = sum(
                leg.quantity // (inst.get("lot_size", 50) if inst else 50)
                for s in active_strategies
                for leg in s.legs
                if leg.status.value == "active" and leg.side.value == "sell"
            )

            summary = engine.mtm_live.analyze(
                mtm_history,
                active_lots=max(active_lots, 1),
                daily_target=float(getattr(Config, "DAILY_TARGET_PNL", 0)),
            )

            if summary:
                # KPI strip
                kpi = st.columns(5)
                sign = "+" if summary.current_pnl >= 0 else ""
                kpi[0].metric("Current MTM", f"₹{sign}{summary.current_pnl:,.0f}")
                kpi[1].metric("Peak Today", f"₹{summary.peak_pnl:,.0f}",
                              delta=f"DD: ₹{summary.drawdown.current_drawdown:,.0f}")
                kpi[2].metric("Intraday DD", f"₹{summary.drawdown.max_drawdown_today:,.0f}")
                kpi[3].metric("P&L / Lot", f"₹{summary.pnl_per_lot:,.0f}")
                kpi[4].metric("Intraday Sharpe", f"{summary.rolling_sharpe:+.2f}")

                # Velocity
                vel = summary.velocity
                vel_colors = {
                    "ACCELERATING_UP": "#00c853",
                    "RECOVERING": "#64dd17",
                    "DECELERATING": "#ff9800",
                    "ACCELERATING_DOWN": "#d50000",
                    "FLAT": "#90a4ae"
                }
                vc = vel_colors.get(vel.direction, "#90a4ae")
                st.markdown(
                    f'<div style="border-left:4px solid {vc}; padding:8px; background:{vc}15; border-radius:4px; margin:8px 0;">'
                    f'<b>P&L Velocity:</b> {vel.direction.replace("_", " ")} | '
                    f'₹{vel.velocity_per_min:+,.1f}/min | '
                    f'Last 5 ticks: ₹{vel.last_5min_pnl:+,.0f}</div>',
                    unsafe_allow_html=True
                )

                # Equity curve with peak/drawdown
                st.markdown("#### 📈 Equity Curve + Peak/Drawdown")
                curve, annotations = engine.mtm_live.get_equity_curve_with_annotations(
                    mtm_history, active_strategies
                )
                if curve:
                    import pandas as pd
                    curve_df = pd.DataFrame(curve).set_index("time")
                    st.area_chart(curve_df[["MTM", "Peak"]], height=260, use_container_width=True)

                # Drawdown panel
                dd = summary.drawdown
                dd_cols = st.columns(4)
                dd_cols[0].metric("Peak P&L", f"₹{dd.peak_pnl:,.0f}", help=f"Reached at {dd.peak_time}")
                dd_cols[1].metric("Current DD", f"₹{dd.current_drawdown:,.0f}")
                dd_cols[2].metric("Max DD Today", f"₹{dd.max_drawdown_today:,.0f}")
                dd_cols[3].metric("DD%", f"{dd.drawdown_pct:+.1f}%")

                # Hourly breakdown
                if summary.hourly:
                    st.divider()
                    st.markdown("#### 🕐 Hourly P&L Breakdown")
                    hourly_data = [{
                        "Hour": h.hour + ":00",
                        "Open": h.open_pnl,
                        "Close": h.close_pnl,
                        "High": h.high_pnl,
                        "Low": h.low_pnl,
                        "Net P&L": h.net_pnl,
                        "Ticks": h.num_ticks,
                    } for h in summary.hourly]
                    hdf = pd.DataFrame(hourly_data)
                    def color_net(val):
                        if isinstance(val, (int, float)):
                            return "color: #00c853" if val >= 0 else "color: #ff4444"
                        return ""
                    st.dataframe(hdf.style.map(color_net, subset=["Net P&L"]),
                                 use_container_width=True, hide_index=True)

                # Daily target progress
                if summary.target_pnl > 0:
                    st.divider()
                    st.markdown("#### 🎯 Daily Target Progress")
                    prog = min(max(summary.target_pct / 100, 0), 1)
                    st.progress(prog, text=f"{summary.target_pct:.0f}% of ₹{summary.target_pnl:,.0f} target")

    except Exception as e:
        st.warning(f"MTM Live error: {e}")
        import traceback; st.code(traceback.format_exc())


# ── Tab: IV Rank ─────────────────────────────────────────────
with tab_iv_rank:
    st.subheader("IV Rank & Percentile Tracker")

    if not chain:
        st.info("Option chain required. Load the Chain tab first.")
    else:
        try:
            from iv_rank import IVRankEngine, IVSnapshot
            if not hasattr(engine, "iv_rank") or engine.iv_rank is None:
                engine.iv_rank = IVRankEngine()

            spot_here = state.get_spot(breeze_code)

            # Initialize IV history in session state
            if "iv_snapshot_history" not in st.session_state:
                st.session_state["iv_snapshot_history"] = []

            iv_history = st.session_state["iv_snapshot_history"]

            # Auto-take snapshot every render
            new_snap = engine.iv_rank.take_snapshot(chain, spot_here, inst_name, iv_history)
            if not iv_history or iv_history[-1].timestamp != new_snap.timestamp:
                iv_history.append(new_snap)
                if len(iv_history) > 480:  # keep last ~2 hours at 15-sec intervals
                    iv_history = iv_history[-480:]
                st.session_state["iv_snapshot_history"] = iv_history

            # Build summary
            summary = engine.iv_rank.build_summary(chain, spot_here, inst_name, iv_history)

            if summary:
                # Regime badge
                regime_colors = {
                    "EXTREME": "#d50000", "HIGH": "#ff6d00",
                    "NORMAL": "#1976d2", "LOW": "#90a4ae"
                }
                rc = regime_colors.get(summary.iv_regime, "#90a4ae")

                # Alert banners
                if summary.spike_alert:
                    st.error(f"🚨 IV SPIKE DETECTED: ATM IV surged +3pp recently → Possible news/event!")
                if summary.crush_alert:
                    st.success(f"💰 IV CRUSH IN PROGRESS: ATM IV dropped >3pp → Premium sellers benefiting!")

                st.markdown(
                    f'<div style="border:2px solid {rc}; border-radius:10px; padding:14px; '
                    f'background:{rc}15; margin:8px 0;">'
                    f'<h3 style="color:{rc}; margin:0;">{summary.iv_regime} IV ENVIRONMENT</h3>'
                    f'<p style="margin:4px 0;">ATM IV = {summary.atm_iv:.1f}% | '
                    f'IV Rank = {summary.iv_rank:.0f}/100 | '
                    f'IV Percentile = {summary.iv_percentile:.0f}th</p></div>',
                    unsafe_allow_html=True
                )

                # Main metrics
                mc = st.columns(5)
                mc[0].metric("ATM IV", f"{summary.atm_iv:.2f}%")
                mc[1].metric("CE IV", f"{summary.atm_iv_ce:.2f}%")
                mc[2].metric("PE IV", f"{summary.atm_iv_pe:.2f}%")
                mc[3].metric("IV Skew (CE-PE)", f"{summary.iv_skew:+.2f}pp")
                mc[4].metric("IV Momentum", summary.iv_momentum)

                mc2 = st.columns(4)
                mc2[0].metric("IV Rank (Session)", f"{summary.iv_rank:.0f}/100")
                mc2[1].metric("IV Change (Session)", f"{summary.iv_change_session:+.2f}pp")
                mc2[2].metric("Session High IV", f"{summary.session_high_iv:.1f}%")
                mc2[3].metric("Session Low IV", f"{summary.session_low_iv:.1f}%")

                # IV History chart
                if iv_history:
                    st.markdown("#### 📊 IV History (Session)")
                    hist_data = engine.iv_rank.iv_history_to_chart_data(iv_history)
                    if hist_data:
                        hdf = pd.DataFrame(hist_data).set_index("time")
                        st.line_chart(hdf[["ATM IV", "CE IV", "PE IV"]], height=200, use_container_width=True)

                # Volatility Cone
                st.divider()
                st.markdown("#### 🔔 Volatility Cone (IV vs DTE)")
                iv_std = max(summary.atm_iv * 0.15, 2.0)  # estimate std as 15% of ATM IV
                cone = engine.iv_rank.build_volatility_cone(summary.atm_iv, iv_std)
                cone_df = pd.DataFrame({
                    "DTE": cone.dte_points,
                    "Current IV": cone.current_iv,
                    "+1 Std": cone.upper_1std,
                    "-1 Std": cone.lower_1std,
                    "+2 Std": cone.upper_2std,
                    "-2 Std": cone.lower_2std,
                }).set_index("DTE")
                st.line_chart(cone_df, height=220, use_container_width=True)

                # Strike IV table
                if summary.strike_ranks:
                    st.divider()
                    st.markdown("#### 🏷️ Per-Strike IV vs ATM")
                    srank_data = [{
                        "Strike": int(sr.strike),
                        "Type": sr.right,
                        "IV%": sr.iv,
                        "IV vs ATM": f"{sr.iv_vs_atm:+.1f}pp",
                        "Moneyness": f"{sr.moneyness:.4f}",
                    } for sr in summary.strike_ranks if sr.iv > 0]
                    if srank_data:
                        srdf = pd.DataFrame(srank_data)
                        st.dataframe(srdf, use_container_width=True, hide_index=True, height=320)

                if st.button("Reset IV History"):
                    st.session_state["iv_snapshot_history"] = []
                    st.success("IV history reset.")
                    st.rerun()

        except Exception as e:
            st.warning(f"IV Rank error: {e}")
            import traceback; st.code(traceback.format_exc())


# ── Tab: Theta ───────────────────────────────────────────────
with tab_theta:
    st.subheader("Theta Decay Projector")

    try:
        from theta_projector import ThetaProjector
        if not hasattr(engine, "theta_proj") or engine.theta_proj is None:
            engine.theta_proj = ThetaProjector()

        strategies_all = state.get_strategies()
        spot_map_here = state.get_all_spots()
        active_strats = [s for s in strategies_all if s.status.value in ("active", "partial_exit")]

        if not active_strats:
            st.info("No active strategies to project theta for.")
        else:
            # Portfolio-level projection
            portfolio_proj = engine.theta_proj.project_portfolio(active_strats, spot_map_here)

            # Portfolio KPIs
            pk = st.columns(4)
            pk[0].metric("Daily Theta (₹)", f"₹{portfolio_proj.total_daily_theta:,.0f}")
            pk[1].metric("Total Premium", f"₹{portfolio_proj.total_premium:,.0f}")
            pk[2].metric("Theta Earned", f"₹{portfolio_proj.total_theta_earned:,.0f}")
            pk[3].metric("Capture Rate", f"{portfolio_proj.portfolio_capture_rate:.1f}%")

            # Aggregate theta curve
            if portfolio_proj.aggregate_curve:
                st.markdown("#### 📉 Portfolio Theta Projection (Combined)")
                agg_df = pd.DataFrame(portfolio_proj.aggregate_curve).set_index("DTE")
                st.area_chart(agg_df["Portfolio Theta P&L"], height=200, use_container_width=True)

            # Per-strategy projections
            st.markdown("#### 🃏 Strategy Theta Cards")
            for proj in portfolio_proj.strategy_projections:
                with st.expander(
                    f"{proj.strategy_type.replace('_', ' ').upper()} [{proj.strategy_id}] "
                    f"| Daily Θ: ₹{proj.daily_theta:,.0f} | Capture: {proj.theta_capture_rate:.0f}%",
                    expanded=False,
                ):
                    pc = st.columns(4)
                    pc[0].metric("DTE", f"{proj.current_dte:.1f}d")
                    pc[1].metric("Daily Θ", f"₹{proj.daily_theta:,.0f}")
                    pc[2].metric("Theta Earned", f"₹{proj.theta_earned_so_far:,.0f}")
                    pc[3].metric("Optimal Exit DTE", f"{proj.optimal_exit_dte:.1f}d")

                    pc2 = st.columns(3)
                    pc2[0].metric("Days to 50% profit", f"{proj.days_to_target_50pct:.1f}d")
                    pc2[1].metric("Days to 75% profit", f"{proj.days_to_target_75pct:.1f}d")
                    pc2[2].metric("Capture Rate", f"{proj.theta_capture_rate:.0f}%")

                    for note in proj.notes:
                        st.caption(note)

                    # Projection curve
                    if proj.projection_curve:
                        curve_dicts = engine.theta_proj.curve_to_dicts(proj.projection_curve)
                        cdf = pd.DataFrame(curve_dicts).set_index("DTE")
                        st.line_chart(cdf[["Projected P&L", "Daily Theta"]], height=180, use_container_width=True)

    except Exception as e:
        st.warning(f"Theta projector error: {e}")
        import traceback; st.code(traceback.format_exc())


# ── Tab: Delta Hedge ─────────────────────────────────────────
with tab_delta:
    st.subheader("Delta Hedging Dashboard")

    try:
        from delta_hedger import DeltaHedger
        if not hasattr(engine, "delta_hedger") or engine.delta_hedger is None:
            engine.delta_hedger = DeltaHedger()

        strategies_all = state.get_strategies()
        spot_map_here = state.get_all_spots()

        dh_cols = st.columns(3)
        with dh_cols[0]:
            dh_band_lower = st.number_input("Delta Band Lower", value=-30.0, step=5.0, key="dh_lower")
        with dh_cols[1]:
            dh_band_upper = st.number_input("Delta Band Upper", value=30.0, step=5.0, key="dh_upper")
        with dh_cols[2]:
            st.write("")  # spacer

        dash = engine.delta_hedger.build_dashboard(
            strategies_all,
            spot_map_here,
            instrument=inst_name,
            chain=chain if chain else None,
            delta_band=(float(dh_band_lower), float(dh_band_upper)),
        )

        # Status
        if dash.hedge_needed:
            st.error(
                f"⚠️ HEDGE NEEDED: Net Delta = {dash.portfolio_delta:+.1f} "
                f"(outside band [{dh_band_lower:+.0f}, {dh_band_upper:+.0f}])"
            )
        else:
            st.success(f"✅ Delta within band. Net Delta = {dash.portfolio_delta:+.1f}")

        # Main metrics
        dm = st.columns(4)
        dm[0].metric("Portfolio Net Δ", f"{dash.portfolio_delta:+.2f}")
        dm[1].metric("Δ in ₹ (per 1% move)", f"₹{dash.portfolio_delta_inr:+,.0f}")
        dm[2].metric("Band Breach", f"{dash.band_breach_pct:.1f}%" if dash.hedge_needed else "None")
        dm[3].metric("Spot", f"₹{dash.spot:,.1f}")

        # Hedge suggestions
        if dash.hedge_needed:
            st.divider()
            st.markdown("#### 🛡️ Hedge Suggestions")

            if dash.hedge_suggestion:
                hs = dash.hedge_suggestion
                st.markdown(
                    f'<div style="border:2px solid #ff6d00; border-radius:8px; padding:12px; background:#ff6d0015;">'
                    f'<b>FULL HEDGE:</b> {hs.description}<br>'
                    f'<b>Quantity:</b> {hs.quantity} ({hs.lots:.1f} lots) | '
                    f'<b>Estimated Cost:</b> ₹{hs.estimated_cost:,.0f} | '
                    f'<b>Delta Reduction:</b> {hs.delta_reduction:+.1f}</div>',
                    unsafe_allow_html=True
                )

            if dash.mini_hedge:
                mh = dash.mini_hedge
                st.markdown(
                    f'<div style="border:1px solid #ff9800; border-radius:8px; padding:12px; background:#ff980015; margin-top:8px;">'
                    f'<b>MINI HEDGE (50%):</b> {mh.description}<br>'
                    f'<b>Cost:</b> ₹{mh.estimated_cost:,.0f}</div>',
                    unsafe_allow_html=True
                )

        # Tail hedge suggestion (always show)
        st.divider()
        st.markdown("#### 🔐 Tail Risk Protection")
        if dash.tail_hedge:
            th = dash.tail_hedge
            st.info(f"💡 {th.description} | Est. Cost: ₹{th.estimated_cost:,.0f}")

        # Per-strategy breakdown
        if dash.delta_by_strategy:
            st.divider()
            st.markdown("#### 📊 Delta by Strategy")
            strat_data = [{
                "Strategy": d.strategy_id,
                "Type": d.strategy_type.replace("_", " ").upper(),
                "Net Δ": d.net_delta,
                "Δ in ₹ (1% move)": d.delta_in_rupees,
                "Biggest Contributor": d.biggest_contributor,
            } for d in dash.delta_by_strategy]
            sdf = pd.DataFrame(strat_data)
            st.dataframe(sdf, use_container_width=True, hide_index=True)

        # By instrument
        if dash.delta_by_instrument:
            st.divider()
            st.markdown("#### 🎯 Delta by Instrument")
            instr_cols = st.columns(len(dash.delta_by_instrument))
            for i, (code, delta) in enumerate(dash.delta_by_instrument.items()):
                instr_cols[i].metric(code, f"{delta:+.2f}")

    except Exception as e:
        st.warning(f"Delta hedge error: {e}")
        import traceback; st.code(traceback.format_exc())


# ── Tab: GEX ────────────────────────────────────────────────
with tab_gex:
    st.subheader("Gamma Exposure (GEX) Analyzer")

    if not chain:
        st.info("Option chain required. Load the Chain tab first.")
    else:
        try:
            from gex import GEXEngine
            if not hasattr(engine, "gex") or engine.gex is None:
                engine.gex = GEXEngine()

            spot_here = state.get_spot(breeze_code)

            with st.spinner("Computing GEX profile..."):
                gex_profile = engine.gex.compute_gex_profile(
                    chain, spot_here, inst_name, exp_str,
                    lot_size=inst.get("lot_size", 50)
                )

            if not gex_profile:
                st.warning("Could not compute GEX — ensure chain has OI and IV data.")
            else:
                # Regime banner
                regime_color = "#00c853" if gex_profile.regime == "POSITIVE" else "#d50000"
                st.markdown(
                    f'<div style="border:2px solid {regime_color}; border-radius:10px; padding:14px; '
                    f'background:{regime_color}15; margin-bottom:12px;">'
                    f'<h3 style="color:{regime_color}; margin:0;">{gex_profile.regime} GEX REGIME</h3>'
                    f'<p style="margin:4px 0;">{gex_profile.regime_description}</p></div>',
                    unsafe_allow_html=True
                )

                # Key metrics
                gkm = st.columns(4)
                gkm[0].metric("Net GEX", f"₹{gex_profile.net_total_gex:+,.1f}M")
                gkm[1].metric("GEX Flip Strike", f"{int(gex_profile.gex_flip_strike):,}")
                gkm[2].metric("Call Wall (Resistance)", f"{int(gex_profile.largest_call_wall):,}")
                gkm[3].metric("Put Wall (Support)", f"{int(gex_profile.largest_put_wall):,}")

                grng = st.columns(2)
                grng[0].metric("GEX Upper Range", f"{gex_profile.gex_range_upper:,.0f}")
                grng[1].metric("GEX Lower Range", f"{gex_profile.gex_range_lower:,.0f}")

                # Key levels
                st.divider()
                st.markdown("#### 🔑 Key GEX Levels")
                key_levels = engine.gex.get_key_levels(gex_profile)
                for lvl in key_levels:
                    lc = lvl["color"]
                    st.markdown(
                        f'<div style="border-left:4px solid {lc}; padding:8px; background:{lc}15; '
                        f'border-radius:4px; margin:4px 0;">'
                        f'<b>{lvl["level"]}: {lvl["strike"]:,}</b> — {lvl["description"]}</div>',
                        unsafe_allow_html=True
                    )

                # GEX chart
                st.divider()
                st.markdown("#### 📊 GEX by Strike")
                gex_data = gex_profile.to_chart_data()
                if gex_data:
                    gdf = pd.DataFrame(gex_data).set_index("Strike")
                    st.bar_chart(gdf["Net GEX (₹M)"], height=300, use_container_width=True)

                # Full table
                with st.expander("Full GEX Table"):
                    full_data = [{
                        "Strike": int(gs.strike),
                        "Call OI": int(gs.call_oi),
                        "Put OI": int(gs.put_oi),
                        "Call GEX (₹M)": round(gs.call_gex / 1e6, 2),
                        "Put GEX (₹M)": round(gs.put_gex / 1e6, 2),
                        "Net GEX (₹M)": round(gs.net_gex / 1e6, 2),
                        "Flip Zone": "⚡" if gs.is_flip_zone else "",
                        "Pin Zone": "📌" if gs.is_pin_zone else "",
                    } for gs in gex_profile.strikes]
                    gfull_df = pd.DataFrame(full_data)
                    st.dataframe(gfull_df, use_container_width=True, hide_index=True, height=420)

        except Exception as e:
            st.warning(f"GEX error: {e}")
            import traceback; st.code(traceback.format_exc())


# ── Tab: Calendar ────────────────────────────────────────────
with tab_calendar:
    st.subheader("Calendar Spread Strategies")

    try:
        from calendar_strategies import CalendarStrategyEngine

        # Ensure engine.calendar is initialised
        if not hasattr(engine, "calendar") or engine.calendar is None:
            try:
                engine.calendar = CalendarStrategyEngine(
                    session=engine.session,
                    order_mgr=engine.omgr,
                    db=engine.db,
                    state=engine.state,
                )
            except Exception as _ce:
                engine.calendar = None

        st.info("📅 Calendar spreads profit from time decay differential between near and far expiries. "
                "Positive vega — benefit from IV expansion on the long leg.")

        # Strategy info cards
        cal_infos = CalendarStrategyEngine.get_strategy_info()
        tile_c = st.columns(3)
        for i, info in enumerate(cal_infos):
            with tile_c[i % 3]:
                st.markdown(
                    f"""<div style="border:1px solid #333; padding:12px; border-radius:10px; margin:4px; background:#161b22;">
                        <h4>{info['emoji']} {info['name']}</h4>
                        <b>Bias:</b> {info['bias']}<br>
                        <b>Risk:</b> {info['risk']}<br>
                        <b>Legs:</b> <code>{info['legs']}</code><br>
                        <b>Best for:</b> {info['ideal']}<br>
                        <hr style="border-color:#333;">
                        <b>Max Profit:</b> {info['max_profit']}<br>
                        <b>Max Loss:</b> {info['max_loss']}
                    </div>""",
                    unsafe_allow_html=True,
                )

        st.divider()
        st.markdown("#### 🚀 Deploy Calendar Strategy")

        cal_cols = st.columns([2, 1, 1, 1])
        with cal_cols[0]:
            cal_type = st.selectbox("Strategy Type", [
                "Calendar Spread", "Diagonal Spread", "Double Calendar"
            ], key="cal_type")
        with cal_cols[1]:
            cal_inst = st.selectbox("Instrument", list(INSTRUMENTS.keys()), key="cal_inst")
        with cal_cols[2]:
            cal_lots = st.number_input("Lots", 1, 10, 1, key="cal_lots")
        with cal_cols[3]:
            cal_right = st.radio("Type", ["CALL", "PUT"], horizontal=True, key="cal_right")

        # Expiry selection
        exp_cols = st.columns(2)
        with exp_cols[0]:
            st.caption("Near Expiry (SELL — faster decay)")
            cal_near_exp = exp_str  # current selected expiry from sidebar
            st.info(f"Near: {cal_near_exp}")
        with exp_cols[1]:
            st.caption("Far Expiry (BUY — vega buffer)")
            # Compute next-next expiry (2 weeks out)
            try:
                from utils import next_weekly_expiry, breeze_expiry
                import datetime as _dt
                cal_inst_data = Config.instrument(cal_inst)
                far_dt = next_weekly_expiry(cal_inst) + _dt.timedelta(weeks=2)
                cal_far_exp = breeze_expiry(far_dt)
            except Exception:
                cal_far_exp = exp_str
            st.info(f"Far: {cal_far_exp}")

        # Strategy-specific params
        if cal_type == "Calendar Spread":
            cs_spot_here = state.get_spot(breeze_code) or 24000
            cs_inst_data = Config.instrument(cal_inst)
            cs_gap = cs_inst_data["strike_gap"]
            from utils import atm_strike as _atm
            default_cal_strike = float(_atm(cs_spot_here, cs_gap))
            cal_strike = st.number_input("Strike (0 = ATM)", value=default_cal_strike, step=float(cs_gap), key="cal_strike")

        elif cal_type == "Diagonal Spread":
            cal_near_delta = st.slider("Near-expiry short delta", 0.15, 0.45, 0.30, 0.05, key="cal_nd")
            cal_far_atm = st.checkbox("Far leg at ATM (else same strike as near)", value=True, key="cal_fatm")

        elif cal_type == "Double Calendar":
            dbl_c = st.columns(2)
            with dbl_c[0]:
                dbl_call_delta = st.slider("Call side delta", 0.15, 0.40, 0.20, 0.05, key="dbl_cd")
            with dbl_c[1]:
                dbl_put_delta = st.slider("Put side delta", 0.15, 0.40, 0.20, 0.05, key="dbl_pd")

        if st.button(f"Deploy {cal_type}", type="primary", use_container_width=True):
            if not getattr(engine, "calendar", None):
                st.error("Calendar engine not initialised. Check logs.")
            else:
                with st.spinner(f"Deploying {cal_type}..."):
                    cal_result = None
                    if cal_type == "Calendar Spread":
                        cal_result = engine.calendar.deploy_calendar_spread(
                            cal_inst, cal_near_exp, cal_far_exp,
                            int(cal_lots), float(cal_strike), cal_right
                        )
                    elif cal_type == "Diagonal Spread":
                        cal_result = engine.calendar.deploy_diagonal_spread(
                            cal_inst, cal_near_exp, cal_far_exp,
                            int(cal_lots), float(cal_near_delta), bool(cal_far_atm), cal_right
                        )
                    elif cal_type == "Double Calendar":
                        cal_result = engine.calendar.deploy_double_calendar(
                            cal_inst, cal_near_exp, cal_far_exp,
                            int(cal_lots), float(dbl_call_delta), float(dbl_put_delta)
                        )

                if cal_result:
                    st.success(f"✅ Deployed: {cal_result.strategy_id} ({cal_result.status.value})")
                else:
                    st.error("Deployment failed — check Logs tab.")

        # Pre-entry analysis
        if chain and cal_type == "Calendar Spread":
            st.divider()
            st.markdown("#### 📐 Calendar Entry Analysis")
            try:
                cs_spot_now = state.get_spot(breeze_code)
                analysis = engine.calendar.analyze_calendar_entry(
                    chain, chain,  # same chain for near/far (simplified)
                    cs_spot_now,
                    float(cal_strike) if cal_strike > 0 else float(_atm(cs_spot_now, cs_gap)),
                    cal_right,
                )
                if "error" not in analysis:
                    ac = st.columns(4)
                    ac[0].metric("Near IV", f"{analysis['near_iv']}%")
                    ac[1].metric("Far IV", f"{analysis['far_iv']}%")
                    ac[2].metric("IV Diff", f"{analysis['iv_differential']:+.1f}pp")
                    ac[3].metric("Net Debit", f"₹{analysis['net_debit']:.2f}")
                    iv_edge_color = "#00c853" if analysis["iv_edge"] == "SELL NEAR" else "#ff9800"
                    st.markdown(
                        f'<div style="border-left:4px solid {iv_edge_color}; padding:8px; '
                        f'background:{iv_edge_color}15; border-radius:4px;">'
                        f'<b>IV Edge:</b> {analysis["iv_edge"]}</div>',
                        unsafe_allow_html=True
                    )
                    for note in analysis.get("notes", []):
                        st.caption(f"• {note}")
            except Exception as _ae:
                st.caption(f"Analysis unavailable: {_ae}")

    except Exception as e:
        st.warning(f"Calendar strategies error: {e}")
        import traceback; st.code(traceback.format_exc())


# ══════════════════════════════════════════════════════════════════
# LAYER 3 TABS (6 new: OI Flow, Trade Journal, Monitor, Payoff, Margin, Regime)
# ══════════════════════════════════════════════════════════════════

# ── Tab: OI Flow (Open Interest Analytics) ────────────────────────
with tab_oi_flow:
    try:
        st.subheader("📊 Open Interest Analytics")
        _instrument_oi = st.selectbox("Instrument", list(INSTRUMENTS.keys()), key="oi_inst")
        _chain_oi = engine.strategy_engine.get_chain_with_greeks(_instrument_oi)
        _spot_oi = engine.state.get_spot(INSTRUMENTS[_instrument_oi]["breeze_code"])

        if _chain_oi and _spot_oi > 0:
            _oi_eng = getattr(engine, "oi_analytics", None)
            if _oi_eng is None:
                from oi_analytics import OIAnalyticsEngine
                _oi_eng = OIAnalyticsEngine()

            # Get or init prev chain from session
            _prev_oi_chain = st.session_state.get("prev_oi_chain", None)
            _oi_summary = _oi_eng.analyze(_chain_oi, _spot_oi, prev_chain=_prev_oi_chain)
            st.session_state["prev_oi_chain"] = _chain_oi

            # PCR + Max Pain banner
            _pcr_color = "#00c853" if _oi_summary.pcr_interpretation in ("BULLISH", "EXTREME_BULLISH") else (
                "#f44336" if "BEARISH" in _oi_summary.pcr_interpretation else "#ff9800"
            )
            st.markdown(
                f'<div style="border-left:5px solid {_pcr_color}; padding:12px; '
                f'background:{_pcr_color}18; border-radius:6px; margin-bottom:12px;">'
                f'<b>PCR (OI): {_oi_summary.pcr_oi:.3f}</b> — {_oi_summary.pcr_interpretation} &nbsp;|&nbsp; '
                f'{_oi_summary.pcr_signal}</div>',
                unsafe_allow_html=True
            )

            # KPI row
            _c1, _c2, _c3, _c4, _c5 = st.columns(5)
            _c1.metric("PCR (OI)", f"{_oi_summary.pcr_oi:.3f}")
            _c2.metric("PCR (Vol)", f"{_oi_summary.pcr_vol:.3f}")
            _c3.metric("Max Pain", f"{int(_oi_summary.max_pain.max_pain_strike):,}")
            _mp_dist = _oi_summary.max_pain.distance_pct
            _c4.metric("Spot vs Max Pain", f"{_mp_dist:+.2f}%", delta_color="normal" if abs(_mp_dist) < 1 else "inverse")
            _c5.metric("Unusual Signals", str(len(_oi_summary.unusual_signals)))

            st.markdown("---")
            _ocol1, _ocol2 = st.columns(2)
            with _ocol1:
                st.markdown("**🏰 Call Wall (Resistance)**")
                _cwall_data = [d for d in _oi_summary.strike_data if d.ce_oi > 0]
                _cwall_sorted = sorted(_cwall_data, key=lambda d: d.ce_oi, reverse=True)[:5]
                for _cw in _cwall_sorted:
                    _cw_pct = (_cw.strike - _spot_oi) / _spot_oi * 100
                    st.write(f"**{int(_cw.strike):,}** ({_cw_pct:+.1f}%) — OI: {_cw.ce_oi:,.0f}")

            with _ocol2:
                st.markdown("**🛡️ Put Wall (Support)**")
                _pwall_data = [d for d in _oi_summary.strike_data if d.pe_oi > 0]
                _pwall_sorted = sorted(_pwall_data, key=lambda d: d.pe_oi, reverse=True)[:5]
                for _pw in _pwall_sorted:
                    _pw_pct = (_pw.strike - _spot_oi) / _spot_oi * 100
                    st.write(f"**{int(_pw.strike):,}** ({_pw_pct:+.1f}%) — OI: {_pw.pe_oi:,.0f}")

            # OI chart
            st.markdown("---")
            st.markdown("**📉 OI Distribution by Strike**")
            import pandas as pd, plotly.graph_objects as go
            _oi_df = pd.DataFrame([{
                "strike": int(d.strike), "CE OI": d.ce_oi, "PE OI": d.pe_oi,
                "CE Chg": d.ce_oi_chg, "PE Chg": d.pe_oi_chg,
            } for d in _oi_summary.strike_data])
            if not _oi_df.empty:
                _oi_near = _oi_df[(_oi_df["strike"] >= _spot_oi * 0.93) & (_oi_df["strike"] <= _spot_oi * 1.07)]
                if not _oi_near.empty:
                    _fig_oi = go.Figure()
                    _fig_oi.add_bar(x=_oi_near["strike"], y=_oi_near["CE OI"], name="CE OI", marker_color="#2196f3")
                    _fig_oi.add_bar(x=_oi_near["strike"], y=_oi_near["PE OI"], name="PE OI", marker_color="#f44336")
                    _fig_oi.add_vline(x=_spot_oi, line_dash="dash", line_color="yellow",
                                       annotation_text=f"Spot {int(_spot_oi):,}")
                    _fig_oi.add_vline(x=_oi_summary.max_pain.max_pain_strike,
                                       line_dash="dot", line_color="orange",
                                       annotation_text=f"MaxPain {int(_oi_summary.max_pain.max_pain_strike):,}")
                    _fig_oi.update_layout(barmode="group", height=300, margin=dict(l=0, r=0, t=20, b=0),
                                           plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                                           font_color="white", xaxis_title="Strike", yaxis_title="OI")
                    st.plotly_chart(_fig_oi, use_container_width=True)

            # Unusual signals
            if _oi_summary.unusual_signals:
                st.markdown("**⚡ Unusual OI Activity**")
                for _sig in _oi_summary.unusual_signals[:8]:
                    _sig_color = "#f44336" if _sig.severity == "CRITICAL" else "#ff9800"
                    st.markdown(
                        f'<div style="border-left:3px solid {_sig_color}; padding:6px; '
                        f'background:{_sig_color}15; border-radius:4px; margin:4px 0; font-size:13px;">'
                        f'{_sig.right} {int(_sig.strike):,}: {_sig.signal} | OI Chg: {_sig.oi_chg:+,.0f} '
                        f'({_sig.oi_chg_pct:+.1f}%)</div>',
                        unsafe_allow_html=True
                    )

            # PCR History
            if _oi_summary.pcr_history:
                st.markdown("**📈 PCR Trend (Session)**")
                _pcr_df = pd.DataFrame(_oi_summary.pcr_history)
                if "time" in _pcr_df.columns:
                    import plotly.express as px
                    _pfig = px.line(_pcr_df, x="time", y="pcr_oi", title="PCR (OI) Intraday")
                    _pfig.add_hline(y=1.0, line_dash="dash", line_color="white", opacity=0.4)
                    _pfig.add_hline(y=1.2, line_dash="dot", line_color="#00c853", opacity=0.5, annotation_text="Bullish")
                    _pfig.add_hline(y=0.8, line_dash="dot", line_color="#f44336", opacity=0.5, annotation_text="Bearish")
                    _pfig.update_layout(height=200, margin=dict(l=0, r=0, t=30, b=0),
                                         plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font_color="white")
                    st.plotly_chart(_pfig, use_container_width=True)
        else:
            st.info("Waiting for option chain data...")
    except Exception as e:
        st.warning(f"OI Analytics error: {e}")
        import traceback; st.code(traceback.format_exc())


# ── Tab: Trade Journal ────────────────────────────────────────────
with tab_trade_journal:
    try:
        st.subheader("📓 Trade Journal & P&L Attribution")
        _tj = getattr(engine, "trade_journal", None)
        if _tj is None:
            from trade_journal import TradeJournal
            _tj = TradeJournal()
        _all_strats = engine.state.get_strategies()
        _tj_summary = _tj.analyze(_all_strats)

        if _tj_summary.total_trades == 0:
            st.info("No closed trades yet. Trade history will appear here after positions are exited.")
        else:
            # Top metrics
            _tm1, _tm2, _tm3, _tm4, _tm5, _tm6 = st.columns(6)
            _tm1.metric("Total Trades", _tj_summary.total_trades)
            _tm2.metric("Win Rate", f"{_tj_summary.win_rate:.1f}%")
            _tm3.metric("Total P&L", f"₹{_tj_summary.total_pnl:,.0f}",
                        delta=f"₹{_tj_summary.today_pnl:,.0f} today")
            _tm4.metric("Profit Factor", f"{_tj_summary.profit_factor:.2f}")
            _tm5.metric("Avg P&L", f"₹{_tj_summary.avg_pnl:,.0f}")
            _tm6.metric("Avg Hold", f"{_tj_summary.avg_hold_hours:.1f}h")

            # MTD / WTD
            _pt1, _pt2 = st.columns(2)
            _pt1.metric("MTD P&L", f"₹{_tj_summary.mtd_pnl:,.0f}")
            _pt2.metric("WTD P&L", f"₹{_tj_summary.wtd_pnl:,.0f}")

            st.markdown("---")
            # Streak info
            _sk = _tj_summary.streak
            _sk_label = f"{'🟢' if _sk.current_streak > 0 else '🔴'} {'Win' if _sk.current_streak > 0 else 'Loss'} streak: {abs(_sk.current_streak)}"
            _l5 = " ".join(["✅" if r == "W" else "❌" for r in _sk.last_5_results])
            st.info(f"**Streak:** {_sk_label}  |  Last 5: {_l5}  |  Best win run: {_sk.max_win_streak}  |  Worst loss run: {_sk.max_loss_streak}")

            # Daily P&L chart
            if _tj_summary.daily_pnl:
                st.markdown("**📅 Daily P&L**")
                import pandas as pd, plotly.graph_objects as go
                _dpnl_df = pd.DataFrame([{"date": d.date, "pnl": d.pnl, "trades": d.trades} for d in _tj_summary.daily_pnl])
                _dpnl_colors = ["#00c853" if v >= 0 else "#f44336" for v in _dpnl_df["pnl"]]
                _dpnl_fig = go.Figure([go.Bar(x=_dpnl_df["date"], y=_dpnl_df["pnl"],
                                               marker_color=_dpnl_colors, name="Daily P&L")])
                _dpnl_fig.add_hline(y=0, line_color="white", opacity=0.4)
                _dpnl_fig.update_layout(height=220, margin=dict(l=0, r=0, t=10, b=0),
                                         plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font_color="white")
                st.plotly_chart(_dpnl_fig, use_container_width=True)

            # Per-strategy type breakdown
            if _tj_summary.strategy_stats:
                st.markdown("**📊 Strategy Type Performance**")
                import pandas as pd
                _ss_data = [{
                    "Strategy": s.strategy_type.replace("_", " ").title(),
                    "Trades": s.total_trades,
                    "Win%": f"{s.win_rate:.0f}%",
                    "P&L": f"₹{s.total_pnl:,.0f}",
                    "Avg Win": f"₹{s.avg_win:,.0f}",
                    "Avg Loss": f"₹{s.avg_loss:,.0f}",
                    "PF": f"{s.profit_factor:.2f}",
                    "Avg Hold": f"{s.avg_hold_hours:.1f}h",
                } for s in _tj_summary.strategy_stats]
                st.dataframe(pd.DataFrame(_ss_data), use_container_width=True, hide_index=True)

            # Recent trades table
            st.markdown("**🕐 Recent Trades**")
            if _tj_summary.recent_trades:
                import pandas as pd
                _rt_data = [{
                    "Date": r.exit_date, "Type": r.strategy_type.replace("_", " ").title(),
                    "Inst": r.instrument, "Lots": r.lots,
                    "Premium": f"₹{r.premium_collected:,.0f}",
                    "P&L": f"₹{r.pnl:,.0f}",
                    "P&L%": f"{r.pnl_pct:.1f}%",
                    "Hold": f"{r.hold_hours:.1f}h",
                    "Exit": r.exit_reason,
                } for r in _tj_summary.recent_trades[:15]]
                _rt_df = pd.DataFrame(_rt_data)
                st.dataframe(_rt_df, use_container_width=True, hide_index=True)
    except Exception as e:
        st.warning(f"Trade Journal error: {e}")
        import traceback; st.code(traceback.format_exc())


# ── Tab: Monitor (Strategy Health) ───────────────────────────────
with tab_monitor:
    try:
        st.subheader("🩺 Live Strategy Health Monitor")
        _mon = getattr(engine, "strategy_monitor", None)
        if _mon is None:
            from strategy_monitor import StrategyMonitor
            _mon = StrategyMonitor()

        _strats_active = engine.state.get_strategies()
        _spots_mon = engine.state.get_all_spots()
        _mon_summary = _mon.evaluate(_strats_active, _spots_mon)

        if not _mon_summary.strategies:
            st.info("No active strategies to monitor. Deploy strategies to see health scores here.")
        else:
            # Portfolio header
            _ph_color = "#00c853" if _mon_summary.portfolio_health >= 65 else (
                "#ff9800" if _mon_summary.portfolio_health >= 45 else "#f44336"
            )
            st.markdown(
                f'<div style="background:{_ph_color}22; border:1px solid {_ph_color}55; '
                f'border-radius:8px; padding:12px; margin-bottom:12px;">'
                f'<b style="font-size:18px;">Portfolio Health Score: '
                f'<span style="color:{_ph_color}">{_mon_summary.portfolio_health:.0f}/100</span></b>'
                f' &nbsp;|&nbsp; Total P&L: ₹{_mon_summary.total_pnl:,.0f}'
                f' &nbsp;|&nbsp; 🔴 {_mon_summary.critical_count} Critical'
                f' &nbsp;|&nbsp; 🟡 {_mon_summary.warn_count} Warn'
                f' &nbsp;|&nbsp; 🟢 {_mon_summary.ok_count} OK</div>',
                unsafe_allow_html=True
            )

            # Per-strategy cards
            for _h in _mon_summary.strategies:
                _hc = {"EXCELLENT": "#00c853", "GOOD": "#69f0ae", "CAUTION": "#ffd740",
                        "DANGER": "#ff9800", "CRITICAL": "#f44336"}.get(_h.health_label, "#ffffff")
                _action_c = {"LOW": "#69f0ae", "MEDIUM": "#ffd740", "HIGH": "#ff9800",
                              "CRITICAL": "#f44336"}.get(_h.action_urgency, "#ffffff")
                with st.expander(
                    f"{'🟢' if _h.health_label in ('EXCELLENT','GOOD') else '🟡' if _h.health_label=='CAUTION' else '🔴'} "
                    f"{_h.strategy_type.replace('_',' ').title()} | {_h.instrument} | "
                    f"P&L: ₹{_h.pnl:,.0f} ({_h.pnl_pct:+.1f}%) | Score: {_h.health_score:.0f}/100",
                    expanded=_h.health_label in ("CRITICAL", "DANGER")
                ):
                    _hc1, _hc2, _hc3, _hc4, _hc5 = st.columns(5)
                    _hc1.metric("Health", f"{_h.health_score:.0f}/100")
                    _hc2.metric("SL Buffer", f"{_h.sl_proximity_pct:.0f}%")
                    _hc3.metric("Delta Drift", f"{_h.delta_drift:.1f}")
                    _hc4.metric("DTE", f"{_h.dte:.1f}d")
                    _hc5.metric("In Trade", f"{_h.time_in_trade_hrs:.0f}h")

                    st.markdown(
                        f'<div style="border-left:4px solid {_action_c}; padding:8px; '
                        f'background:{_action_c}18; border-radius:4px;">'
                        f'<b>Recommended Action:</b> {_h.action} &nbsp;—&nbsp; {_h.action_reason}</div>',
                        unsafe_allow_html=True
                    )

                    # Component breakdown
                    if _h.components:
                        _comp_cols = st.columns(len(_h.components))
                        for _ci, _comp in enumerate(_h.components):
                            _cc = "#00c853" if _comp.status == "OK" else "#ff9800" if _comp.status == "WARN" else "#f44336"
                            _comp_cols[_ci].markdown(
                                f'<div style="text-align:center; font-size:12px;">'
                                f'<div style="color:{_cc}; font-weight:bold;">{_comp.score:.0f}</div>'
                                f'<div>{_comp.name}</div>'
                                f'<div style="color:{_cc}; font-size:10px;">{_comp.status}</div></div>',
                                unsafe_allow_html=True
                            )

                    # Leg details
                    if _h.legs_summary:
                        import pandas as pd
                        st.dataframe(pd.DataFrame(_h.legs_summary), use_container_width=True, hide_index=True)
    except Exception as e:
        st.warning(f"Strategy Monitor error: {e}")
        import traceback; st.code(traceback.format_exc())


# ── Tab: Payoff Diagram Builder ───────────────────────────────────
with tab_payoff:
    try:
        st.subheader("📈 Payoff Diagram Builder")
        _pb = getattr(engine, "payoff", None)
        if _pb is None:
            from payoff_builder import PayoffBuilder
            _pb = PayoffBuilder()

        _active_strats_pb = [s for s in engine.state.get_strategies()
                              if hasattr(s, "status") and s.status.value in ("active", "partial_exit")]

        if not _active_strats_pb:
            st.info("No active strategies. Deploy strategies to visualize their payoff diagrams.")
        else:
            _pb_strat_opts = {f"{s.strategy_type.value.replace('_', ' ').title()} | {s.stock_code} | {s.strategy_id[:8]}": s
                              for s in _active_strats_pb}
            _pb_select = st.selectbox("Select Strategy", list(_pb_strat_opts.keys()), key="pb_sel")
            _pb_strat = _pb_strat_opts[_pb_select]
            _pb_spot = engine.state.get_spot(_pb_strat.stock_code)

            _pb_range = st.slider("Spot Range ±%", 5, 25, 12, key="pb_range")

            if _pb_spot > 0:
                _pb_summary = _pb.build(_pb_strat, _pb_spot, spot_range_pct=float(_pb_range))
                _pb_curve = _pb_summary.curve

                # KPIs
                _pk1, _pk2, _pk3, _pk4, _pk5 = st.columns(5)
                _pk1.metric("Max Profit", f"₹{_pb_summary.max_profit:,.0f}")
                _pk2.metric("Max Loss", f"₹{_pb_summary.max_loss:,.0f}")
                _pk3.metric("Risk:Reward", f"1:{_pb_summary.risk_reward:.2f}")
                _pk4.metric("PoP", f"{_pb_summary.probability_of_profit_pct:.1f}%")
                _pk5.metric("Expected Move", f"±₹{_pb_summary.expected_move_1sd:,.0f}")

                if _pb_curve.breakeven_points:
                    _be_str = " | ".join([f"₹{b:,.0f}" for b in _pb_curve.breakeven_points])
                    st.info(f"**Break-even points:** {_be_str}")

                # Payoff chart
                import plotly.graph_objects as go, pandas as pd
                _pb_fig = go.Figure()
                if _pb_curve.spot_range and _pb_curve.pnl_expiry:
                    _pb_fig.add_trace(go.Scatter(
                        x=_pb_curve.spot_range, y=_pb_curve.pnl_expiry,
                        name="P&L at Expiry", line=dict(color="#2196f3", width=2)
                    ))
                if _pb_curve.pnl_now:
                    _pb_fig.add_trace(go.Scatter(
                        x=_pb_curve.spot_range, y=_pb_curve.pnl_now,
                        name="P&L Now (BS)", line=dict(color="#00c853", width=2, dash="dot")
                    ))
                if _pb_curve.pnl_iv_up:
                    _pb_fig.add_trace(go.Scatter(
                        x=_pb_curve.spot_range, y=_pb_curve.pnl_iv_up,
                        name="IV +5%", line=dict(color="#ff9800", width=1, dash="dash")
                    ))
                if _pb_curve.pnl_iv_down:
                    _pb_fig.add_trace(go.Scatter(
                        x=_pb_curve.spot_range, y=_pb_curve.pnl_iv_down,
                        name="IV -5%", line=dict(color="#e91e63", width=1, dash="dash")
                    ))
                _pb_fig.add_hline(y=0, line_color="white", opacity=0.5)
                _pb_fig.add_vline(x=_pb_spot, line_dash="dash", line_color="yellow",
                                   annotation_text=f"Spot {int(_pb_spot):,}")
                for _be in _pb_curve.breakeven_points:
                    _pb_fig.add_vline(x=_be, line_dash="dot", line_color="#ff9800", opacity=0.7)
                _pb_fig.update_layout(
                    height=400, margin=dict(l=0, r=0, t=20, b=0),
                    plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                    font_color="white", xaxis_title="Spot Price", yaxis_title="P&L (₹)",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(_pb_fig, use_container_width=True)

                # Scenario table
                st.markdown("**📊 Scenario Analysis**")
                _sc_data = [{
                    "Scenario": s.label,
                    "Spot": f"₹{s.spot:,.0f}",
                    "P&L at Expiry": f"₹{s.pnl_expiry:,.0f}",
                    "P&L Now": f"₹{s.pnl_now:,.0f}",
                    "Change from Current": f"₹{s.pnl_change_from_current:+,.0f}",
                } for s in _pb_summary.scenarios]
                st.dataframe(pd.DataFrame(_sc_data), use_container_width=True, hide_index=True)

            # Combined portfolio payoff
            st.markdown("---")
            st.markdown("**🔗 Combined Portfolio Payoff**")
            if st.button("Generate Combined Payoff", key="pb_combined"):
                _pb_combined = _pb.build_combined_curve(_active_strats_pb, _pb_spot)
                if _pb_combined:
                    _comb_fig = go.Figure()
                    _comb_fig.add_trace(go.Scatter(
                        x=_pb_combined.spot_range, y=_pb_combined.pnl_expiry,
                        name="Combined Expiry", line=dict(color="#2196f3", width=2.5)
                    ))
                    _comb_fig.add_trace(go.Scatter(
                        x=_pb_combined.spot_range, y=_pb_combined.pnl_now,
                        name="Combined Now", line=dict(color="#00c853", width=2, dash="dot")
                    ))
                    _comb_fig.add_hline(y=0, line_color="white", opacity=0.5)
                    _comb_fig.add_vline(x=_pb_spot, line_dash="dash", line_color="yellow",
                                         annotation_text=f"Spot {int(_pb_spot):,}")
                    _comb_fig.update_layout(height=350, margin=dict(l=0, r=0, t=20, b=0),
                                             plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                                             font_color="white", title="Portfolio Combined Payoff")
                    st.plotly_chart(_comb_fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Payoff Builder error: {e}")
        import traceback; st.code(traceback.format_exc())


# ── Tab: Margin Ledger ────────────────────────────────────────────
with tab_margin:
    try:
        st.subheader("💰 Margin & Capital Efficiency")
        _ml = getattr(engine, "margin_ledger", None)
        if _ml is None:
            from margin_ledger import MarginLedger
            _ml = MarginLedger()

        _cap_input = st.number_input("Total Capital (₹)", value=500_000, step=100_000, format="%d", key="ml_cap")
        _strats_ml = engine.state.get_strategies()
        _spots_ml = engine.state.get_all_spots()
        _ml_summary = _ml.compute(_strats_ml, _spots_ml, available_capital=float(_cap_input))

        # Utilization gauge
        _util = _ml_summary.margin_utilization_pct
        _util_color = "#00c853" if _util < 50 else "#ff9800" if _util < 75 else "#f44336"
        st.markdown(
            f'<div style="background:#1e1e2e; border-radius:8px; padding:14px; margin-bottom:12px;">'
            f'<b>Margin Utilization: <span style="color:{_util_color}; font-size:20px;">{_util:.1f}%</span></b> &nbsp; '
            f'Free: ₹{_ml_summary.free_margin:,.0f} ({_ml_summary.free_margin_pct:.1f}%)</div>',
            unsafe_allow_html=True
        )

        # Progress bar
        st.progress(min(1.0, _util / 100))

        # Alerts
        for _al in _ml_summary.alerts:
            _al_fn = st.error if _al["level"] == "CRITICAL" else st.warning if _al["level"] == "WARN" else st.info
            _al_fn(_al["message"])

        # KPIs
        _mc1, _mc2, _mc3, _mc4, _mc5 = st.columns(5)
        _mc1.metric("SPAN Margin", f"₹{_ml_summary.total_span_margin:,.0f}")
        _mc2.metric("Exposure Margin", f"₹{_ml_summary.total_exposure_margin:,.0f}")
        _mc3.metric("Total Blocked", f"₹{_ml_summary.total_margin_blocked:,.0f}")
        _mc4.metric("ROM (Trade)", f"{_ml_summary.portfolio_rom_pct:.2f}%")
        _mc5.metric("ROM (Annual)", f"{_ml_summary.portfolio_rom_annual_pct:.1f}%")

        _mc6, _mc7 = st.columns(2)
        _mc6.metric("Margin at Risk", f"₹{_ml_summary.margin_at_risk:,.0f}")
        _mc7.metric("Capital Efficiency", f"{_ml_summary.portfolio_capital_efficiency:.3f}")

        # Per-strategy table
        if _ml_summary.strategy_margins:
            st.markdown("---")
            st.markdown("**📋 Strategy Margin Breakdown**")
            import pandas as pd
            _sm_data = [{
                "Strategy": m.strategy_type.replace("_", " ").title(),
                "Instrument": m.instrument,
                "Lots": m.lots,
                "Notional": f"₹{m.notional:,.0f}",
                "SPAN": f"₹{m.span_margin:,.0f}",
                "Exposure": f"₹{m.exposure_margin:,.0f}",
                "Total Margin": f"₹{m.total_margin:,.0f}",
                "Premium": f"₹{m.premium_collected:,.0f}",
                "P&L": f"₹{m.current_pnl:+,.0f}",
                "ROM%": f"{m.rom_trade_pct:.2f}%",
                "ROM Ann%": f"{m.rom_annual_pct:.1f}%",
                "Efficiency": f"{m.capital_efficiency:.3f}",
                "Days": f"{m.days_held:.1f}",
            } for m in _ml_summary.strategy_margins]
            st.dataframe(pd.DataFrame(_sm_data), use_container_width=True, hide_index=True)

            # Margin pie chart
            import plotly.express as px
            _pie_data = pd.DataFrame([{
                "Label": f"{m.strategy_type[:12]} {m.instrument}",
                "Margin": m.total_margin,
            } for m in _ml_summary.strategy_margins])
            if not _pie_data.empty and _pie_data["Margin"].sum() > 0:
                _pie_fig = px.pie(_pie_data, values="Margin", names="Label",
                                   title="Margin Allocation",
                                   color_discrete_sequence=px.colors.sequential.Plasma)
                _pie_fig.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0),
                                        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font_color="white")
                st.plotly_chart(_pie_fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Margin Ledger error: {e}")
        import traceback; st.code(traceback.format_exc())


# ── Tab: Regime Detector ──────────────────────────────────────────
with tab_regime:
    try:
        st.subheader("🌡️ Market Regime & Rollover Intelligence")
        _regime_inst = st.selectbox("Instrument", list(INSTRUMENTS.keys()), key="reg_inst")

        _tabs_reg = st.tabs(["Regime", "Rollover", "Position Sizer", "IV Surface"])

        with _tabs_reg[0]:
            # Regime Detector
            _reg_engine = getattr(engine, "regime", None)
            if _reg_engine is None:
                from regime_detector import RegimeDetector
                _reg_engine = RegimeDetector()

            _chain_reg = engine.strategy_engine.get_chain_with_greeks(_regime_inst)
            _spot_reg = engine.state.get_spot(INSTRUMENTS[_regime_inst]["breeze_code"])
            _atm_iv_reg = 0.0
            if _chain_reg and _spot_reg > 0:
                _atm_rows = sorted(_chain_reg, key=lambda r: abs(r.get("strike", 0) - _spot_reg))
                _atm_iv_reg = _atm_rows[0].get("iv", 15.0) / 100 if _atm_rows else 0.18

            # Build IV history from session state
            _iv_hist_key = f"iv_history_{_regime_inst}"
            _iv_history = st.session_state.get(_iv_hist_key, [])
            if _atm_iv_reg > 0:
                _iv_history.append(_atm_iv_reg * 100)
                st.session_state[_iv_hist_key] = _iv_history[-252:]  # keep 1 year max

            _reg_result = _reg_engine.analyze(
                current_iv=_atm_iv_reg * 100 if _atm_iv_reg > 0 else 18.0,
                iv_history=_iv_history,
                spot_history=st.session_state.get(f"spot_hist_{_regime_inst}", [_spot_reg] * 10),
            )

            # Update spot history
            _sh_key = f"spot_hist_{_regime_inst}"
            _sh = st.session_state.get(_sh_key, [])
            if _spot_reg > 0:
                _sh.append(_spot_reg)
                st.session_state[_sh_key] = _sh[-252:]

            # Regime banner
            _vol_colors = {"low": "#2196f3", "normal": "#00c853", "high": "#ff9800", "extreme": "#f44336"}
            _vol_c = _vol_colors.get(_reg_result.vol_regime.value, "#ffffff")
            st.markdown(
                f'<div style="background:{_vol_c}22; border:1px solid {_vol_c}55; '
                f'border-radius:8px; padding:12px; margin-bottom:12px;">'
                f'<b>Vol Regime: <span style="color:{_vol_c}">{_reg_result.vol_regime.value.upper()}</span></b>'
                f' &nbsp;|&nbsp; IV: {_reg_result.current_iv:.1f}%'
                f' &nbsp;|&nbsp; IVP: {_reg_result.iv_percentile:.0f}th pct'
                f' &nbsp;|&nbsp; IV/HV: {_reg_result.iv_rv_ratio:.2f}x'
                f' &nbsp;|&nbsp; {_reg_result.iv_state.value.replace("_", " ").title()}</div>',
                unsafe_allow_html=True
            )

            _rc1, _rc2, _rc3, _rc4 = st.columns(4)
            _rc1.metric("Current IV", f"{_reg_result.current_iv:.1f}%")
            _rc2.metric("IV Rank", f"{_reg_result.iv_rank:.1f}")
            _rc3.metric("Trend", _reg_result.trend_regime.value.replace("_", " ").title())
            _rc4.metric("Term Structure", _reg_result.ts_shape.value.title())

            # Strategy recommendations
            if _reg_result.preferred_strategies:
                st.success("**✅ Preferred Strategies:** " + ", ".join([s.replace("_", " ").title() for s in _reg_result.preferred_strategies]))
            if _reg_result.avoid_strategies:
                st.error("**🚫 Avoid:** " + ", ".join([s.replace("_", " ").title() for s in _reg_result.avoid_strategies]))

            st.markdown("**📋 Regime Notes:**")
            for _note in _reg_result.regime_notes:
                st.caption(f"• {_note}")

        with _tabs_reg[1]:
            # Rollover Engine
            _roll_eng = getattr(engine, "rollover", None)
            if _roll_eng is None:
                from rollover import RolloverEngine
                _roll_eng = RolloverEngine()

            st.markdown("**🔄 Rollover Analysis**")
            _roll_strats = engine.state.get_strategies()
            _roll_spots = engine.state.get_all_spots()

            if not _roll_strats:
                st.info("No active strategies to analyze for rollover.")
            else:
                _roll_analyses = _roll_eng.analyze_portfolio(_roll_strats, _roll_spots)
                if not _roll_analyses:
                    st.success("✅ No roll actions needed. All strategies have sufficient DTE.")
                else:
                    for _ra in _roll_analyses:
                        _urg_color = {"LOW": "#00c853", "MEDIUM": "#ffd740", "HIGH": "#ff9800", "CRITICAL": "#f44336"}.get(_ra.urgency, "#ffffff")
                        st.markdown(
                            f'<div style="border-left:4px solid {_urg_color}; padding:10px; '
                            f'background:{_urg_color}15; border-radius:4px; margin:8px 0;">'
                            f'<b>{_ra.instrument}</b> — DTE: {_ra.dte_current:.1f}d | '
                            f'Urgency: <span style="color:{_urg_color}">{_ra.urgency}</span><br>'
                            f'Roll P&L: ₹{_ra.roll_pnl:+,.0f} | '
                            f'Theta Curr: ₹{_ra.current_theta:.0f}/d → Next: ₹{_ra.next_theta:.0f}/d<br>'
                            f'{"<br>".join("• " + r for r in _ra.reasons)}</div>',
                            unsafe_allow_html=True
                        )

        with _tabs_reg[2]:
            # Position Sizer
            _sizer = getattr(engine, "sizer", None)
            if _sizer is None:
                from position_sizer import PositionSizer
                _sizer = PositionSizer()

            st.markdown("**📐 Position Sizing Calculator**")
            _ps_col1, _ps_col2 = st.columns(2)
            with _ps_col1:
                _ps_capital = st.number_input("Available Capital (₹)", 100_000, 10_000_000, 500_000, 100_000, key="ps_cap")
                _ps_used = st.number_input("Used Capital (₹)", 0, 10_000_000, 0, 50_000, key="ps_used")
                _ps_win_rate = st.slider("Expected Win Rate %", 30, 90, 65, key="ps_wr") / 100
                _ps_strat = st.selectbox("Strategy Type", ["straddle", "strangle", "iron_condor", "iron_butterfly"], key="ps_strat")

            with _ps_col2:
                _ps_spot = engine.state.get_spot(INSTRUMENTS[_regime_inst]["breeze_code"])
                _ps_atm_iv = _atm_iv_reg if _atm_iv_reg > 0 else 0.18
                _ps_expiry = st.text_input("Expiry (YYYY-MM-DDT06:00:00.000Z)", key="ps_exp",
                                            value=f"{(datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d')}T06:00:00.000Z")
                _ps_lot = INSTRUMENTS[_regime_inst]["lot_size"]
                st.metric("ATM IV", f"{_ps_atm_iv*100:.1f}%")
                st.metric("Lot Size", _ps_lot)

            if st.button("Calculate Optimal Lots", key="ps_calc"):
                try:
                    from datetime import timedelta
                    _ps_result = _sizer.size_for_premium(
                        instrument=_regime_inst,
                        spot=float(_ps_spot) if _ps_spot > 0 else 24000,
                        atm_iv=float(_ps_atm_iv),
                        expiry=_ps_expiry,
                        strategy_type=_ps_strat,
                        available_capital=float(_ps_capital),
                        used_capital=float(_ps_used),
                        win_rate=float(_ps_win_rate),
                    )
                    _ps_cols = st.columns(4)
                    _ps_cols[0].metric("✅ Recommended Lots", _ps_result.recommended_lots)
                    _ps_cols[1].metric("Max (Margin)", _ps_result.max_lots_margin)
                    _ps_cols[2].metric("Max (Kelly)", _ps_result.max_lots_kelly)
                    _ps_cols[3].metric("Max (Risk)", _ps_result.max_lots_risk)
                    st.info(f"**Margin Required:** ₹{_ps_result.margin_required:,.0f} | "
                            f"**Premium:** ₹{_ps_result.premium_collected:,.0f} | "
                            f"**Max Loss:** ₹{_ps_result.max_loss_estimate:,.0f} | "
                            f"**R:R:** {_ps_result.risk_reward:.2f}")
                    for _note in _ps_result.notes:
                        st.caption(f"• {_note}")
                except Exception as _pse:
                    st.error(f"Sizing error: {_pse}")

        with _tabs_reg[3]:
            # IV Surface
            _ivs = getattr(engine, "iv_surface", None)
            if _ivs is None:
                from iv_surface import IVSurfaceEngine
                _ivs = IVSurfaceEngine()

            _chain_ivs = engine.strategy_engine.get_chain_with_greeks(_regime_inst)
            _spot_ivs = engine.state.get_spot(INSTRUMENTS[_regime_inst]["breeze_code"])

            st.markdown("**🌐 IV Surface (Moneyness × DTE)**")
            if _chain_ivs and _spot_ivs > 0:
                try:
                    _surface = _ivs.build(_chain_ivs, _spot_ivs)
                    if _surface:
                        import plotly.graph_objects as go, pandas as pd
                        _surf_df = pd.DataFrame([{
                            "Strike": int(p.strike),
                            "DTE": round(p.dte_days, 1),
                            "IV%": round(p.iv * 100, 2),
                            "Moneyness": round(p.moneyness, 3),
                            "Right": p.right,
                        } for p in _surface])
                        st.dataframe(_surf_df.sort_values("Strike"), use_container_width=True, hide_index=True)

                    # Term structure
                    _ts = _ivs.term_structure(_chain_ivs, _spot_ivs)
                    if _ts:
                        _ts_fig = go.Figure(go.Scatter(
                            x=[t["dte"] for t in _ts], y=[t["atm_iv"] for t in _ts],
                            mode="lines+markers", line=dict(color="#2196f3", width=2),
                            name="ATM IV by DTE"
                        ))
                        _ts_fig.update_layout(
                            title="Term Structure (ATM IV vs DTE)",
                            height=250, margin=dict(l=0, r=0, t=30, b=0),
                            plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                            font_color="white", xaxis_title="DTE (days)", yaxis_title="ATM IV (%)"
                        )
                        st.plotly_chart(_ts_fig, use_container_width=True)

                    # Skew
                    _skew = _ivs.skew_metrics(_chain_ivs, _spot_ivs, expiry=None)
                    if _skew:
                        st.markdown("**Volatility Skew**")
                        import pandas as pd
                        _sk_df = pd.DataFrame(_skew).rename(columns={"moneyness": "Moneyness", "iv": "IV%", "strike": "Strike", "right": "Right"})
                        st.dataframe(_sk_df, use_container_width=True, hide_index=True)
                except Exception as _ivse:
                    st.caption(f"IV Surface: {_ivse}")
            else:
                st.info("Waiting for chain data...")
    except Exception as e:
        st.warning(f"Regime/Rollover error: {e}")
        import traceback; st.code(traceback.format_exc())
