"""
pages/02_Scanner.py  — NEW PAGE

Options Market Scanner
  • Auto-run scans on live chain data
  • High IV / Unusual OI / Gamma concentration
  • Skew outliers / Max pain deviation
  • Liquidity / spread checks
  • OI change tracker (compare snapshots)
"""

import time
import streamlit as st
import pandas as pd
from datetime import datetime

st.set_page_config(
    page_title="Scanner | Options Terminal",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 Options Market Scanner")

try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=30_000, limit=None, key="scanner_refresh")
except Exception:
    pass


@st.cache_resource
def boot_engine():
    from trading_engine import TradingEngine
    eng = TradingEngine()
    eng.start()
    return eng

engine = boot_engine()
state = engine.state

try:
    from scanner import MarketScanner, ScanResult
    from app_config import Config, INSTRUMENTS
    from utils import breeze_expiry, next_weekly_expiry
    scanner_ok = True
except Exception as e:
    st.error(f"Scanner module error: {e}")
    scanner_ok = False

if not scanner_ok:
    st.stop()

market_scanner = MarketScanner()

# ── Sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    st.header("Scanner Config")
    inst_name = st.selectbox("Instrument", list(INSTRUMENTS.keys()))
    inst = Config.instrument(inst_name)
    breeze_code = inst["breeze_code"]

    exp_dt = next_weekly_expiry(inst_name)
    exp_str = breeze_expiry(exp_dt)
    st.caption(f"Expiry: {exp_dt.strftime('%d-%b-%Y')}")

    st.divider()
    st.subheader("Scan Settings")
    run_high_iv = st.checkbox("High IV", value=True)
    run_unusual_oi = st.checkbox("Unusual OI", value=True)
    run_gamma = st.checkbox("Gamma Concentration", value=True)
    run_theta = st.checkbox("Theta Rich", value=True)
    run_skew = st.checkbox("Skew Outliers", value=True)
    run_max_pain = st.checkbox("Max Pain Deviation", value=True)
    run_liquidity = st.checkbox("Illiquid Options", value=True)

    st.divider()
    auto_scan = st.toggle("Auto-scan every 30s", value=True)
    if st.button("🔄 Scan Now", use_container_width=True, type="primary"):
        st.rerun()

# ── Load chain ───────────────────────────────────────────────────
spot = state.get_spot(breeze_code)

try:
    chain = engine.get_chain(inst_name, exp_str)
except Exception as e:
    chain = []
    st.warning(f"Chain error: {e}")

# ── Header bar ───────────────────────────────────────────────────
h1, h2, h3, h4 = st.columns(4)
h1.metric("Spot", f"₹{spot:,.0f}" if spot > 0 else "N/A")
h2.metric("Chain Rows", len(chain))
h3.metric("Last Scan", datetime.now().strftime("%H:%M:%S"))
h4.metric("Instrument", inst_name)

st.divider()

if not chain or spot <= 0:
    st.info("Waiting for chain data...")
    st.stop()

# ── Run selected scans ───────────────────────────────────────────
results = []

if run_high_iv:
    results.extend(market_scanner.scan_high_iv(chain, spot, inst_name))
if run_unusual_oi:
    results.extend(market_scanner.scan_unusual_oi(chain, spot, inst_name))
if run_gamma:
    results.extend(market_scanner.scan_gamma_concentration(chain, spot, inst_name))
if run_theta:
    results.extend(market_scanner.scan_theta_rich(chain, spot, inst_name, exp_str))
if run_skew:
    results.extend(market_scanner.scan_put_call_skew(chain, spot, inst_name))
if run_max_pain:
    results.extend(market_scanner.scan_max_pain_deviation(chain, spot, inst_name))
if run_liquidity:
    results.extend(market_scanner.scan_liquidity(chain, inst_name))

# Sort by severity
_order = {"ALERT": 0, "WARN": 1, "INFO": 2}
results.sort(key=lambda r: _order.get(r.severity, 2))

# ── Results display ──────────────────────────────────────────────
alerts = [r for r in results if r.severity == "ALERT"]
warns = [r for r in results if r.severity == "WARN"]
infos = [r for r in results if r.severity == "INFO"]

# Summary
c1, c2, c3 = st.columns(3)
c1.metric("🚨 Alerts", len(alerts))
c2.metric("⚠️ Warnings", len(warns))
c3.metric("ℹ️ Info", len(infos))

st.divider()

if not results:
    st.success("✅ No significant signals detected. Market looks normal.")
else:
    # Alerts
    if alerts:
        st.subheader("🚨 Alerts")
        for r in alerts:
            with st.container():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.error(f"**[{r.scan_type}]** {r.description}")
                with col2:
                    st.caption(f"Strike: {int(r.strike) if r.strike else 'N/A'}")
                    st.caption(f"Value: {r.value:.2f}")

    # Warnings
    if warns:
        st.subheader("⚠️ Warnings")
        for r in warns:
            with st.container():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.warning(f"**[{r.scan_type}]** {r.description}")
                with col2:
                    st.caption(f"Strike: {int(r.strike) if r.strike else 'N/A'}")

    # Info
    if infos:
        with st.expander(f"ℹ️ Info ({len(infos)} signals)", expanded=False):
            for r in infos:
                st.info(f"**[{r.scan_type}]** {r.description}")

    st.divider()

    # Raw table
    st.subheader("All Scan Results")
    results_df = pd.DataFrame([r.to_dict() for r in results])
    if not results_df.empty:
        def color_severity(row):
            if row["severity"] == "ALERT":
                return ["background-color: #4d0d0d; color: #ff6644"] * len(row)
            if row["severity"] == "WARN":
                return ["background-color: #4d4d0d; color: #ffcc44"] * len(row)
            return [""] * len(row)

        styled_df = results_df.style.apply(color_severity, axis=1)
        st.dataframe(styled_df, use_container_width=True, hide_index=True, height=380)

# ── OI Snapshot Manager ──────────────────────────────────────────
st.divider()
st.subheader("📊 OI Change Tracker")
st.caption("Snapshot OI data to compare changes over time")

if "oi_snapshots" not in st.session_state:
    st.session_state.oi_snapshots = []

col1, col2 = st.columns([1, 3])

with col1:
    if st.button("📸 Take Snapshot", use_container_width=True):
        snapshot = {
            "time": datetime.now().strftime("%H:%M:%S"),
            "chain": chain.copy(),
        }
        st.session_state.oi_snapshots.append(snapshot)
        if len(st.session_state.oi_snapshots) > 5:  # keep last 5
            st.session_state.oi_snapshots.pop(0)
        st.success(f"Snapshot #{len(st.session_state.oi_snapshots)} taken")

    if st.button("🗑️ Clear Snapshots"):
        st.session_state.oi_snapshots = []
        st.rerun()

with col2:
    if len(st.session_state.oi_snapshots) >= 2:
        snap_labels = [f"#{i+1} @ {s['time']}" for i, s in enumerate(st.session_state.oi_snapshots)]
        base_choice = st.selectbox("Compare FROM", snap_labels[:-1], key="base_snap")
        comp_choice = st.selectbox("Compare TO", snap_labels[1:], index=len(snap_labels)-2, key="comp_snap")

        base_idx = snap_labels.index(base_choice)
        comp_idx = snap_labels.index(comp_choice)

        oi_changes = market_scanner.oi_change_analysis(
            st.session_state.oi_snapshots[comp_idx]["chain"],
            st.session_state.oi_snapshots[base_idx]["chain"],
            inst_name,
        )

        if oi_changes:
            st.markdown("**OI Changes:**")
            for r in oi_changes:
                if r.severity == "ALERT":
                    st.error(f"**[{r.scan_type}]** {r.description}")
                else:
                    st.warning(f"**[{r.scan_type}]** {r.description}")
        else:
            st.success("No significant OI changes between snapshots")
    else:
        st.info(f"Take at least 2 snapshots to compare. ({len(st.session_state.oi_snapshots)} taken)")

# ── Chain viewer with scan overlay ──────────────────────────────
with st.expander("📋 Full Chain + Scan Overlay"):
    if chain:
        # Merge scan flags into chain
        flagged_strikes = {}
        for r in results:
            if r.strike > 0:
                key = (int(r.strike), r.right)
                existing = flagged_strikes.get(key, [])
                existing.append(f"{r.scan_type}({r.severity})")
                flagged_strikes[key] = existing

        chain_df = pd.DataFrame(chain)
        chain_df["scan_flags"] = chain_df.apply(
            lambda row: ", ".join(flagged_strikes.get((int(row["strike"]), row["right"]), [])),
            axis=1
        )

        # Color flagged rows
        def highlight_flagged(row):
            flags = row.get("scan_flags", "")
            if "ALERT" in flags:
                return ["background-color: #4d0d0d"] * len(row)
            if "WARN" in flags:
                return ["background-color: #4d4d0d"] * len(row)
            return [""] * len(row)

        styled_chain = chain_df.style.apply(highlight_flagged, axis=1)
        st.dataframe(styled_chain, use_container_width=True, hide_index=True, height=480)
