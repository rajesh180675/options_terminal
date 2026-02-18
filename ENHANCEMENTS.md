# Options Terminal — Layer 3 Enhancements

## Architecture Overview
Layer 1 (6 modules) + Layer 2 (6 modules) + Layer 3 (12 modules) = 24 modules total
25 tabs in main.py | 50 Python files | 18,737 lines of code

---

## Layer 3 Modules (NEW — Zero breaking changes)

### 6 Brand-New Modules

| Module | Classes | Purpose |
|--------|---------|---------|
| oi_analytics.py | OIAnalyticsEngine, OIAnalyticsSummary, MaxPainResult | PCR tracking, OI heatmap, max pain, unusual OI signals |
| trade_journal.py | TradeJournal, JournalSummary, StrategyStats, StreakInfo | P&L attribution, win rate, streaks, daily calendar |
| strategy_monitor.py | StrategyMonitor, StrategyHealth, MonitorSummary | Health score 0-100, SL proximity, delta drift, action recs |
| payoff_builder.py | PayoffBuilder, PayoffCurve, PayoffSummary | Payoff diagram (expiry + BS now), scenarios, IV shock |
| margin_ledger.py | MarginLedger, MarginLedgerSummary, StrategyMargin | SPAN margin tracking, ROM, capital efficiency, alerts |
| order_flow_tracker.py | OrderFlowTracker, OrderFlowSummary, FlowSignal | Volume spike detection, big print alerts, smart money |

### 6 Previously Orphaned Modules — Now Wired

| Module | Wired As | New UI |
|--------|----------|--------|
| rollover.py | engine.rollover | Rollover tab (inside Regime) |
| regime_detector.py | engine.regime | Regime tab |
| position_sizer.py | engine.sizer | Position Sizer subtab |
| heatmap_engine.py | engine.heatmap | Available for Payoff/Risk tabs |
| iv_surface.py | engine.iv_surface | IV Surface subtab |
| analytics.py | engine.analytics_mod | Available system-wide |

### Alert System — Now Wired
- notifier.py hooked into RMS (engine.rms.alerts = engine.notifier)
- Telegram/webhook alerts fire on SL hits, panic exits, strategy deployment

---

## New Tabs (6 added — total 25 tabs)

### Tab 20: OI Flow (📊 Open Interest Analytics)
- Real-time PCR (OI + Volume) with session trend chart
- Call Wall / Put Wall identification from OI concentration
- Max Pain calculator with pain curve visualization
- Unusual OI accumulation detection (>2σ statistical spike)
- OI distribution bar chart (CE/PE by strike, ±7% from spot)
- Historical PCR intraday trend line

### Tab 21: Trade Journal (📓 Trade Journal & P&L Attribution)
- MTD / WTD / Today P&L metrics
- Win rate, profit factor, expectancy
- Streak tracker (current + best/worst historical)
- Daily P&L bar chart (green/red by day)
- Strategy-type performance table (win%, avg win, avg loss, PF, hold time)
- Recent trades table (last 15 closed trades)

### Tab 22: Monitor (🩺 Live Strategy Health Monitor)
- Portfolio health score (0-100 composite)
- Per-strategy expandable cards with 5-component scoring:
  - SL Proximity (30% weight)
  - P&L Health (25% weight)
  - Delta Drift (20% weight)
  - DTE Risk (15% weight)
  - Time Risk (10% weight)
- Color-coded health labels: EXCELLENT / GOOD / CAUTION / DANGER / CRITICAL
- Recommended actions: HOLD / CLOSE_50 / CLOSE_ALL / ADJUST_DELTA / ROLL / MONITOR
- Per-leg detail table with SL buffer % per leg

### Tab 23: Payoff (📈 Payoff Diagram Builder)
- Interactive payoff diagram: expiry + BS now curves
- IV shock overlays: +5% and -5% IV scenarios
- Break-even point vertical lines
- Scenario analysis table: -10% to +10% spot moves
- Combined portfolio payoff (button-triggered)
- KPIs: max profit, max loss, R:R, PoP %, expected move

### Tab 24: Margin (💰 Margin & Capital Efficiency)
- Capital utilization progress bar
- SPAN margin estimation per strategy (SPAN + exposure formula)
- Defined-risk discount for IC/butterfly (70% / 65% of SPAN)
- Per-strategy margin table: SPAN, exposure, total, ROM, annualized ROM
- Margin pie chart (allocation by strategy)
- Smart alerts: critical/warn when free margin < 10%/20%
- Capital efficiency = premium/margin ratio

### Tab 25: Regime (🌡️ Market Regime — 4 subtabs)
- **Regime subtab**: vol regime classification, IV rank/percentile, IV/HV ratio,
  trend regime, term structure shape, strategy recommendations + avoid list
- **Rollover subtab**: per-strategy roll analysis, urgency classification,
  roll cost/credit, gamma risk alerts
- **Position Sizer subtab**: Kelly criterion + margin + risk-based lot sizing,
  input: capital, win rate, strategy type
- **IV Surface subtab**: full IV surface table, term structure chart, skew metrics

---

## Health Scoring Algorithm (strategy_monitor.py)



## PCR Interpretation (oi_analytics.py)



## SPAN Margin Estimates (margin_ledger.py)


