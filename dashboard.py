"""
Project Regime-Master — AI Risk & Performance Dashboard
Streamlit-based monitoring UI with regime traffic light, equity curve,
confidence gauge, and emergency kill switch.

Run:  streamlit run dashboard.py
"""
import os
import json
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

import config

# ─── Page Config ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Regime-Master AI Dashboard",
    page_icon="🤖",
    layout="wide",
)

# ─── Custom Styling ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    }
    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 16px;
    }
    [data-testid="stMetricLabel"] { color: #a0a0b0; }
    [data-testid="stMetricValue"] { color: #ffffff; font-weight: 700; }
    .regime-bull { color: #00e676; font-size: 28px; font-weight: bold; }
    .regime-bear { color: #ff5252; font-size: 28px; font-weight: bold; }
    .regime-chop { color: #ffab40; font-size: 28px; font-weight: bold; }
    .regime-crash { color: #ff1744; font-size: 28px; font-weight: bold; animation: blink 1s infinite; }
    @keyframes blink { 50% { opacity: 0.3; } }
    h1 { color: #e0e0ff !important; }
    h2, h3 { color: #b0b0d0 !important; }
</style>
""", unsafe_allow_html=True)


# ─── Data Loaders ────────────────────────────────────────────────────────────────

def load_bot_state():
    """Load current bot state from JSON."""
    if not os.path.exists(config.STATE_FILE):
        return {
            "regime": "WAITING",
            "confidence": 0,
            "action": "NOT_STARTED",
            "timestamp": "N/A",
            "trade_count": 0,
            "paper_mode": True,
        }
    try:
        with open(config.STATE_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {"regime": "ERROR", "confidence": 0, "action": "READ_FAIL"}


def load_trade_log():
    """Load trade history CSV."""
    if not os.path.exists(config.TRADE_LOG_FILE):
        return pd.DataFrame()
    try:
        return pd.read_csv(config.TRADE_LOG_FILE, parse_dates=["timestamp"])
    except Exception:
        return pd.DataFrame()


# ─── Regime Color Helper ────────────────────────────────────────────────────────

REGIME_STYLES = {
    "BULLISH":       ("regime-bull",  "🟢"),
    "BEARISH":       ("regime-bear",  "🔴"),
    "SIDEWAYS/CHOP": ("regime-chop",  "🟡"),
    "CRASH/PANIC":   ("regime-crash", "💀"),
    "WAITING":       ("regime-chop",  "⏳"),
    "ERROR":         ("regime-crash", "❌"),
}


# ─── Header ──────────────────────────────────────────────────────────────────────
st.title("🤖 Regime-Master: AI Crypto Bot Monitor")
st.caption("Real-time HMM regime classification and trade monitoring")
st.divider()

# ─── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚡ Bot Controls")

    if st.button("🚨 EMERGENCY KILL SWITCH", type="primary", use_container_width=True):
        try:
            os.makedirs(config.DATA_DIR, exist_ok=True)
            with open(config.COMMANDS_FILE, "w") as f:
                json.dump({"command": "KILL", "timestamp": datetime.utcnow().isoformat()}, f)
            st.error("⚠️ Kill command sent! All positions will be closed.")
        except Exception as e:
            st.error(f"Failed: {e}")

    st.divider()

    if st.button("🔄 Reset Kill Switch", use_container_width=True):
        try:
            with open(config.COMMANDS_FILE, "w") as f:
                json.dump({"command": "RESET", "timestamp": datetime.utcnow().isoformat()}, f)
            st.success("✅ Reset command sent.")
        except Exception as e:
            st.error(f"Failed: {e}")

    st.divider()
    st.subheader("⚙️ Configuration")
    st.code(f"Symbol:    {config.PRIMARY_SYMBOL}")
    st.code(f"Testnet:   {config.TESTNET}")
    st.code(f"Paper:     {config.PAPER_TRADE}")
    st.code(f"Max Lev:   {config.LEVERAGE_HIGH}x")
    st.code(f"Conf Req:  {config.CONFIDENCE_HIGH * 100:.0f}%")

    if st.button("🔄 Refresh Data", use_container_width=True):
        st.rerun()

# ─── Load Data ───────────────────────────────────────────────────────────────────
state = load_bot_state()
trade_log = load_trade_log()

# ─── Load Multi-Coin Data ────────────────────────────────────────────────────────
multi_state = None
multi_state_file = os.path.join(config.DATA_DIR, "multi_bot_state.json")
if os.path.exists(multi_state_file):
    try:
        with open(multi_state_file, "r") as f:
            multi_state = json.load(f)
    except Exception:
        pass

# ─── Status Row ──────────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

regime_name = state.get("regime", "WAITING")
style_class, emoji = REGIME_STYLES.get(regime_name, ("regime-chop", "❓"))

with col1:
    st.markdown(f"### {emoji} BTC Regime")
    st.markdown(f'<span class="{style_class}">{regime_name}</span>', unsafe_allow_html=True)

with col2:
    confidence = state.get("confidence", 0)
    conf_pct = confidence * 100 if confidence <= 1 else confidence
    st.metric("🧠 AI Confidence", f"{conf_pct:.1f}%")

with col3:
    if multi_state:
        st.metric("🎯 Active Trades", f"{len(multi_state.get('active_positions', {}))}/{config.MAX_CONCURRENT_POSITIONS}")
    else:
        action = state.get("action", "N/A")
        st.metric("🎯 Last Action", action)

with col4:
    mode_label = "📝 PAPER" if state.get("paper_mode", True) else "🔴 LIVE"
    st.metric("Mode", mode_label)

# ── Multi-Coin Scan Stats ──
if multi_state:
    st.divider()
    st.subheader("⚡ Multi-Coin Trading Engine")
    mcol1, mcol2, mcol3, mcol4, mcol5 = st.columns(5)
    mcol1.metric("🔍 Coins Scanned", multi_state.get("coins_scanned", 0))
    mcol2.metric("✅ Eligible", multi_state.get("eligible_count", 0))
    mcol3.metric("🚀 Deployed", multi_state.get("deployed_count", 0))
    mcol4.metric("📊 Total Trades", multi_state.get("total_trades", 0))
    mcol5.metric("🔄 Cycle #", multi_state.get("cycle", 0))

    # Active Positions table
    active = multi_state.get("active_positions", {})
    if active:
        st.markdown("#### 🟢 Active Positions")
        pos_data = []
        for sym, info in active.items():
            pos_data.append({
                "Symbol": sym,
                "Side": info.get("side", "?"),
                "Regime": info.get("regime", "?"),
                "Confidence": f"{info.get('confidence', 0)*100:.1f}%",
                "Leverage": f"{info.get('leverage', 1)}x",
                "Entry Time": info.get("entry_time", "?")[:19],
            })
        st.dataframe(pd.DataFrame(pos_data), use_container_width=True, height=250)

    # Coin-by-coin status from latest scan
    coin_states = multi_state.get("coin_states", {})
    if coin_states:
        st.markdown("#### 📋 Latest Scan Results")
        scan_data = []
        for sym, info in coin_states.items():
            scan_data.append({
                "Symbol": sym,
                "Regime": info.get("regime", "?"),
                "Confidence": f"{info.get('confidence', 0)*100:.1f}%",
                "Price": f"${info.get('price', 0):,.4f}",
                "Action": info.get("action", "?"),
            })

        df_scan = pd.DataFrame(scan_data)

        def color_action(val):
            if "ELIGIBLE" in str(val):
                return "background-color: rgba(0,230,118,0.25); color: #00e676"
            elif "SKIP" in str(val) or "LOW" in str(val):
                return "background-color: rgba(255,171,64,0.15); color: #ffab40"
            elif "CRASH" in str(val):
                return "background-color: rgba(255,23,68,0.25); color: #ff1744"
            return ""

        st.dataframe(
            df_scan.style.map(color_action, subset=["Action"]),
            use_container_width=True,
            height=500,
        )

st.divider()

# ─── Confidence Gauge ───────────────────────────────────────────────────────────
st.subheader("🔬 AI Confidence Gauge")

gauge = go.Figure(go.Indicator(
    mode="gauge+number+delta",
    value=conf_pct,
    domain={"x": [0, 1], "y": [0, 1]},
    title={"text": "HMM State Probability", "font": {"color": "#b0b0d0"}},
    gauge={
        "axis": {"range": [0, 100], "tickcolor": "#555"},
        "bar": {"color": "#7c4dff"},
        "bgcolor": "rgba(0,0,0,0)",
        "steps": [
            {"range": [0, 50], "color": "rgba(255,82,82,0.2)"},
            {"range": [50, 65], "color": "rgba(255,171,64,0.2)"},
            {"range": [65, 85], "color": "rgba(255,213,79,0.2)"},
            {"range": [85, 100], "color": "rgba(0,230,118,0.2)"},
        ],
        "threshold": {
            "line": {"color": "#00e676", "width": 3},
            "thickness": 0.8,
            "value": config.CONFIDENCE_HIGH * 100,
        },
    },
    number={"suffix": "%", "font": {"color": "#ffffff"}},
))
gauge.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    height=280,
    margin=dict(l=30, r=30, t=60, b=20),
)
st.plotly_chart(gauge, use_container_width=True)

st.divider()

# ─── Equity Curve / Trade History ────────────────────────────────────────────────
st.subheader("📈 Performance Tracking")

if not trade_log.empty and "timestamp" in trade_log.columns:
    tab1, tab2 = st.tabs(["📊 Trade Timeline", "📋 Trade Log"])

    with tab1:
        fig = px.scatter(
            trade_log,
            x="timestamp",
            y="entry_price",
            color="side",
            symbol="regime",
            size_max=12,
            color_discrete_map={"BUY": "#00e676", "SELL": "#ff5252"},
            title="Trade Entries Over Time",
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0.1)",
            font_color="#b0b0d0",
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        display_cols = [c for c in trade_log.columns
                        if c not in ["mode"]]
        st.dataframe(
            trade_log[display_cols].sort_values("timestamp", ascending=False),
            use_container_width=True,
            height=400,
        )

    # Summary metrics
    mcol1, mcol2, mcol3 = st.columns(3)
    mcol1.metric("Total Trades", len(trade_log))
    buys = len(trade_log[trade_log["side"] == "BUY"])
    sells = len(trade_log[trade_log["side"] == "SELL"])
    mcol2.metric("🟢 Longs", buys)
    mcol3.metric("🔴 Shorts", sells)
else:
    st.info("No trades logged yet. Start the bot to see data here.")

st.divider()

# ─── Regime Distribution ────────────────────────────────────────────────────────
st.subheader("🧠 Regime Distribution (From Trade History)")

if not trade_log.empty and "regime" in trade_log.columns:
    regime_counts = trade_log["regime"].value_counts()
    fig_pie = px.pie(
        values=regime_counts.values,
        names=regime_counts.index,
        color=regime_counts.index,
        color_discrete_map={
            "BULLISH": "#00e676",
            "BEARISH": "#ff5252",
            "SIDEWAYS/CHOP": "#ffab40",
            "CRASH/PANIC": "#ff1744",
        },
        title="Trade Distribution by Regime",
    )
    fig_pie.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#b0b0d0",
    )
    fig_pie.update_traces(textfont_color="#ffffff")
    st.plotly_chart(fig_pie, use_container_width=True)
else:
    st.info("Regime distribution will appear after trades are logged.")

# ─── Top 50 Coins Scanner ───────────────────────────────────────────────────────
st.subheader("🔍 Top 50 Coins by Volume — Regime Heatmap")

from coin_scanner import load_scanner_state

scanner_data = load_scanner_state()

if scanner_data and scanner_data.get("coins"):
    coins = scanner_data["coins"]
    scan_time = scanner_data.get("last_scan", "N/A")
    st.caption(f"Last scan: {scan_time} UTC  •  {len(coins)} coins classified")

    # Summary row
    scol1, scol2, scol3, scol4 = st.columns(4)
    bull_n = sum(1 for c in coins if c["regime_name"] == "BULLISH")
    bear_n = sum(1 for c in coins if c["regime_name"] == "BEARISH")
    chop_n = sum(1 for c in coins if c["regime_name"] == "SIDEWAYS/CHOP")
    crash_n = sum(1 for c in coins if c["regime_name"] == "CRASH/PANIC")
    scol1.metric("🟢 Bullish", bull_n)
    scol2.metric("🔴 Bearish", bear_n)
    scol3.metric("🟡 Sideways", chop_n)
    scol4.metric("💀 Crash", crash_n)

    # Build styled table
    df_coins = pd.DataFrame(coins)

    # Color-code regime column
    def color_regime(val):
        colors = {
            "BULLISH": "background-color: rgba(0,230,118,0.25); color: #00e676",
            "BEARISH": "background-color: rgba(255,82,82,0.25); color: #ff5252",
            "SIDEWAYS/CHOP": "background-color: rgba(255,171,64,0.25); color: #ffab40",
            "CRASH/PANIC": "background-color: rgba(255,23,68,0.25); color: #ff1744",
        }
        return colors.get(val, "")

    display_df = df_coins[["rank", "symbol", "regime_name", "confidence", "price"]].copy()
    display_df.columns = ["#", "Symbol", "Regime", "Confidence", "Price"]
    display_df["Confidence"] = (display_df["Confidence"] * 100).round(1).astype(str) + "%"
    display_df["Price"] = display_df["Price"].apply(lambda x: f"${x:,.4f}")

    st.dataframe(
        display_df.style.map(color_regime, subset=["Regime"]),
        use_container_width=True,
        height=600,
    )

    # Regime distribution bar chart
    regime_dist = df_coins["regime_name"].value_counts()
    fig_bar = px.bar(
        x=regime_dist.index,
        y=regime_dist.values,
        color=regime_dist.index,
        color_discrete_map={
            "BULLISH": "#00e676",
            "BEARISH": "#ff5252",
            "SIDEWAYS/CHOP": "#ffab40",
            "CRASH/PANIC": "#ff1744",
        },
        title="Market Regime Distribution (Top 50 Coins)",
        labels={"x": "Regime", "y": "Count"},
    )
    fig_bar.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0.1)",
        font_color="#b0b0d0",
        showlegend=False,
    )
    st.plotly_chart(fig_bar, use_container_width=True)

else:
    st.info("No scanner data yet. Run: `python3 coin_scanner.py` to scan top 50 coins.")

# ─── Footer ──────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<center style='color:#555'>Project Regime-Master v1.0 | "
    f"Last State Update: {state.get('timestamp', 'N/A')}</center>",
    unsafe_allow_html=True,
)

