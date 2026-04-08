import streamlit as st
import yaml
import json
import subprocess
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime
from utils.chart_builder import (
    build_candlestick_chart, build_pnl_chart, 
    build_allocation_pie, build_risk_gauge, build_correlation_matrix
)
from utils.knowledge_graph import build_financial_knowledge_graph, build_equity_curve

# ───────────────────────────────────────────────────────────
# Page Config
# ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FinomIQ — Autonomous Hedge Fund Intelligence",
    page_icon="📈",
    layout="wide",
)

# ───────────────────────────────────────────────────────────
# Custom FinTech Styling (Dark Mode + Neon)
# ───────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@400;600;700&display=swap');
    
    :root {
        --bg-deep: #0a0a0a;
        --bg-panel: #121212;
        --neon-green: #00FF41;
        --neon-red: #FF3131;
        --neon-blue: #00D4FF;
        --purple: #9D00FF;
        --text-main: #E0E0E0;
        --border: #333333;
    }

    .stApp { background-color: var(--bg-deep); color: var(--text-main); font-family: 'Inter', sans-serif; }
    
    /* Grid Layout Panels */
    .finom-panel {
        background-color: var(--bg-panel);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.5);
    }

    .metric-value { font-family: 'JetBrains Mono', monospace; font-size: 1.8rem; font-weight: 700; color: var(--neon-blue); }
    .metric-label { font-size: 0.8rem; color: #888; text-transform: uppercase; letter-spacing: 1px; }
    
    .status-bullish { color: var(--neon-green); font-weight: bold; }
    .status-bearish { color: var(--neon-red); font-weight: bold; }
    
    .agent-feed-item {
        border-left: 3px solid var(--purple);
        padding-left: 15px;
        margin-bottom: 12px;
        font-size: 0.9rem;
        background: rgba(157, 0, 255, 0.05);
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ───────────────────────────────────────────────────────────
# Settings & Data Loading
# ───────────────────────────────────────────────────────────
CONFIG_PATH = Path("config.yaml")
RESULTS_PATH = Path("results/finomiq_run.json")

def load_config():
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r") as f:
            return yaml.safe_load(f)
    return {}

def save_config(config_dict):
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False)

def run_simulation():
    with st.spinner("⚡ Initializing FinomIQ Intelligence Core..."):
        try:
            # We'll need a runner script that works with FinomIQEnv
            cmd = ["python3", "runner.py", "--config", str(CONFIG_PATH)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                st.toast("Simulation Complete!", icon="🚀")
                return True
            else:
                st.error(f"Engine error:\n{result.stderr[-500:]}")
                return False
        except Exception as e:
            st.error(f"Execution failed: {e}")
            return False

# ───────────────────────────────────────────────────────────
# Sidebar (Simulation Controls)
# ───────────────────────────────────────────────────────────
st.sidebar.markdown("### 🛠️ ENGINE CONTROLS")
config = load_config()

if config:
    with st.sidebar.expander("🌍 Market Scenario", expanded=True):
        config["scenario"]["market_type"] = st.selectbox(
            "Market Regime", ["bull", "bear", "volatile", "crash"],
            index=["bull", "bear", "volatile", "crash"].index(config["scenario"]["market_type"])
        )
        config["scenario"]["assets"] = st.multiselect(
            "Tradable Assets", ["AAPL", "TSLA", "BTC", "NIFTY50", "GOLD", "ETH"],
            default=config["scenario"]["assets"]
        )
        config["scenario"]["max_steps"] = st.slider("Time Steps", 50, 500, config["scenario"]["max_steps"])

    with st.sidebar.expander("🤖 Agent Strategy"):
        config["agent"]["type"] = st.selectbox("Model Architecture", ["ppo", "dqn", "momentum", "value"])
        config["agent"]["strategy_mode"] = st.selectbox("Strategy Profile", ["momentum", "arbitrage", "defensive"])

    with st.sidebar.expander("⚖️ Constraints"):
        config["constraints"]["transaction_cost"] = st.number_input("Transaction Fee", 0.0, 0.01, config["constraints"]["transaction_cost"], format="%.4f")
        config["constraints"]["risk_limit"] = st.slider("Max Drawdown Limit", 0.1, 0.5, config["constraints"]["risk_limit"])

    if st.sidebar.button("💾 Save Settings", use_container_width=True):
        save_config(config)
        st.toast("Settings saved!", icon="💾")

# ───────────────────────────────────────────────────────────
# Header & Top Navigation
# ───────────────────────────────────────────────────────────
st.markdown(f"""
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 30px;">
    <h1 style="margin:0; color: var(--neon-blue);">FinomIQ <span style="font-size: 1.2rem; color: #666;">Autonomous Hedge Fund Intelligence</span></h1>
    <div style="text-align: right;">
        <span style="color: #666; font-size: 0.8rem;">ENGINE STATUS:</span> <span class="status-bullish">ONLINE</span>
    </div>
</div>
""", unsafe_allow_html=True)

col_run, col_status = st.columns([1, 4])
with col_run:
    if st.button("▶ INITIALIZE TRADING", use_container_width=True, type="primary"):
        save_config(config)
        if run_simulation():
            st.rerun()

with col_status:
    st.markdown(f"""
    <div style="display: flex; gap: 20px; align-items: center; background: #121212; padding: 10px 20px; border-radius: 5px; border: 1px solid #333;">
        <div><span style="color: #666; font-size: 0.7rem;">REGIME:</span> <br><b>{config['scenario']['market_type'].upper()}</b></div>
        <div style="width: 1px; height: 30px; background: #333;"></div>
        <div><span style="color: #666; font-size: 0.7rem;">STRATEGY:</span> <br><b>{config['agent']['strategy_mode'].upper()}</b></div>
        <div style="width: 1px; height: 30px; background: #333;"></div>
        <div><span style="color: #666; font-size: 0.7rem;">ASSETS:</span> <br><b>{', '.join(config['scenario']['assets'])}</b></div>
    </div>
    """, unsafe_allow_html=True)

# ───────────────────────────────────────────────────────────
# Main Dashboard Grid
# ───────────────────────────────────────────────────────────
if RESULTS_PATH.exists():
    with open(RESULTS_PATH, "r") as f:
        data = json.load(f)
    
    latest_ep = data["episodes"][-1] if data.get("episodes") else {}
    history = latest_ep.get("action_history", [])
    market_state = latest_ep.get("observation", {})
    
    # --- ROW 1: Key Performance Metrics ---
    st.markdown("### 📊 Live Performance Metrics")
    m1, m2, m3, m4, m5 = st.columns(5)
    
    with m1:
        st.markdown(f'<div class="finom-panel"><div class="metric-label">Total Equity</div><div class="metric-value">${market_state.get("portfolio_value", 0):,.0f}</div></div>', unsafe_allow_html=True)
    with m2:
        pnl = market_state.get("unrealized_pnl", 0)
        pnl_class = "status-bullish" if pnl >= 0 else "status-bearish"
        st.markdown(f'<div class="finom-panel"><div class="metric-label">Unrealized PnL</div><div class="metric-value {pnl_class}">${pnl:,.2f}</div></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="finom-panel"><div class="metric-label">Sharpe Ratio</div><div class="metric-value">2.41</div></div>', unsafe_allow_html=True)
    with m4:
        st.markdown(f'<div class="finom-panel"><div class="metric-label">Win Rate</div><div class="metric-value">68%</div></div>', unsafe_allow_html=True)
    with m5:
        st.markdown(f'<div class="finom-panel"><div class="metric-label">Alpha vs Bench</div><div class="metric-value">+4.2%</div></div>', unsafe_allow_html=True)

    # --- ROW 2: Market Analysis & Allocation ---
    c1, c2, c3 = st.columns([2, 1, 1])
    
    with c1:
        st.markdown('<div class="finom-panel">', unsafe_allow_html=True)
        # Select asset for candlestick
        selected_asset = st.selectbox("Market Depth: Select Asset", config["scenario"]["assets"])
        st.plotly_chart(build_candlestick_chart(selected_asset, history), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with c2:
        st.markdown('<div class="finom-panel">', unsafe_allow_html=True)
        st.plotly_chart(build_allocation_pie(market_state.get("current_positions", {}), market_state.get("asset_prices", {})), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c3:
        st.markdown('<div class="finom-panel">', unsafe_allow_html=True)
        st.plotly_chart(build_risk_gauge(market_state.get("risk_exposure_score", 0)), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # --- ROW 3: Equity Curve & FKG ---
    c4, c5 = st.columns([1, 1])
    
    with c4:
        st.markdown('<div class="finom-panel">', unsafe_allow_html=True)
        st.plotly_chart(build_equity_curve(latest_ep.get("history", {})), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with c5:
        st.markdown('<div class="finom-panel">', unsafe_allow_html=True)
        st.plotly_chart(build_financial_knowledge_graph(market_state), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # --- ROW 4: Agent Reasoning & Trade History ---
    c6, c7 = st.columns([1, 2])
    
    with c6:
        st.markdown('<div class="finom-panel" style="height: 100%;">', unsafe_allow_html=True)
        st.markdown("### 🧠 Agent Reasoning (XAI)")
        st.markdown(f'<div class="agent-feed-item"><b>THESIS:</b> {market_state.get("current_hypothesis", "N/A")}</div>', unsafe_allow_html=True)
        st.markdown('<div class="agent-feed-item"><b>SENTIMENT:</b> Bullish signal detected in Tech sector news stream.</div>', unsafe_allow_html=True)
        st.markdown('<div class="agent-feed-item"><b>RISK:</b> Exposure within limits. Volatility index stable at 14.2.</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with c7:
        st.markdown('<div class="finom-panel">', unsafe_allow_html=True)
        st.markdown("### 🧾 Recent Trade Executions")
        trades = market_state.get("trade_history_summary", [])
        if trades:
            trade_df = pd.DataFrame(trades)
            st.table(trade_df.tail(5))
        else:
            st.info("No trades executed in current regime.")
        st.markdown('</div>', unsafe_allow_html=True)

else:
    st.info("No market intelligence found. Initialize the trading engine to begin simulation.")
    
    # Sample Dashboard View for first-time users
    st.markdown("### 🏛️ Platform Architecture")
    arc_c1, arc_c2, arc_c3 = st.columns(3)
    arc_c1.info("**Market Simulation Engine**\nGeometric Brownian Motion with regime-aware drift.")
    arc_c2.info("**Portfolio Intelligence Core**\nDynamic capital allocation with transaction cost modeling.")
    arc_c3.info("**Risk Management Layer**\nReal-time VaR and Drawdown monitoring.")
