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
tabs = st.tabs(["Chain", "Limit Sell", "Positions", "Broker", "Analytics", "Journal", "Backtest", "Logs"])
tab_chain, tab_limit, tab_pos, tab_broker, tab_analytics, tab_journal, tab_backtest, tab_logs = tabs


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
