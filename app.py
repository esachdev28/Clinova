"""
FinomIQ — Hugging Face Edition.
Autonomous Financial Strategy Intelligence Platform.
"""

import json
import os
import subprocess
import yaml
import pandas as pd
from pathlib import Path
import gradio as gr
from app_theme import FinomIQTheme
from utils.chart_builder import (
    build_candlestick_chart, build_pnl_chart, 
    build_allocation_pie, build_risk_gauge, build_correlation_matrix,
    build_drawdown_chart, build_risk_return_scatter, build_sector_heatmap
)
from utils.knowledge_graph import build_financial_knowledge_graph, build_equity_curve
from utils.summary_engine import generate_rule_based_summary, generate_llm_summary
from utils.experiment_panels import (
    build_xai_thinking_advanced, 
    build_advanced_rl_stats, build_strategy_lab_status,
    build_rl_step_monitor
)
from utils.explainability import explain_decision_advanced

# ── Constants ─────────────────────────────────────────────────────────────────

RESULTS_PATH = "results/finomiq_intelligence_run.json"
CONFIG_PATH = "config.yaml"
LOG_PATH = "logs/finomiq.log"

# Ensure directories exist
os.makedirs("results", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_results():
    try:
        if os.path.exists(RESULTS_PATH):
            with open(RESULTS_PATH, "r") as f:
                return json.load(f)
        return None
    except:
        return None

def load_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            return yaml.safe_load(f)
    # Default fallback config
    return {
        "scenario": {"market_type": "bull"},
        "agent": {"type": "ppo"},
        "strategy_sandbox": {"default_rules": "", "hybrid_mode": True},
        "visualization": {"persistence_path": RESULTS_PATH}
    }

def run_intelligence_simulation(m_type, a_type, rules_text, hybrid):
    cfg = load_config()
    cfg["scenario"]["market_type"] = m_type
    cfg["agent"]["type"] = a_type
    cfg["strategy_sandbox"]["default_rules"] = rules_text
    cfg["strategy_sandbox"]["hybrid_mode"] = hybrid
    cfg["visualization"]["persistence_path"] = RESULTS_PATH
    
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(cfg, f)
        
    cmd = ["python3", "runner.py", "--config", CONFIG_PATH]
    # In HF environment, we run the runner script which generates results
    subprocess.run(cmd, capture_output=True)
    return load_results()

# ── Strategy Library ─────────────────────────────────────────────────────────

STRATEGY_TEMPLATES = {
    "Defensive Alpha": "IF sentiment < 0.4 AND vix > 25:\n  HEDGE 20% GOLD\nIF drawdown > 0.05:\n  SELL 50% BTC",
    "Momentum Chaser": "IF trend > 0.02 AND sentiment > 0.6:\n  BUY 15% TSLA\nIF trend < -0.01:\n  SELL 80% TSLA",
    "Conservative Growth": "IF vix < 15 AND sentiment > 0.5:\n  BUY 5% AAPL\nIF vix > 30:\n  HEDGE 30% GOLD",
    "Crypto Speculator": "IF sentiment > 0.8:\n  BUY 20% BTC\nIF drawdown > 0.10:\n  SELL 100% BTC"
}

# ── UI Data Mapping ───────────────────────────────────────────────────────────

def get_intelligence_data(summary_type="Rule-based"):
    data = load_results()
    if not data or "episodes" not in data or not data["episodes"]:
        # If no data yet, run a default simulation to populate UI
        data = run_intelligence_simulation("bull", "ppo", "", True)
        if not data:
            return [None] * 18
    
    latest_ep = data["episodes"][-1]
    history = latest_ep.get("action_history", [])
    obs = latest_ep.get("observation", {})
    results_all = data.get("episodes", [])
    
    # 1. Advanced Charts
    asset = list(obs["asset_prices"].keys())[0] if obs.get("asset_prices") else "AAPL"
    fig_candle = build_candlestick_chart(asset, history)
    fig_pnl = build_pnl_chart(history)
    fig_pie = build_allocation_pie(obs.get("current_positions", {}), obs.get("asset_prices", {}))
    fig_risk = build_risk_gauge(obs.get("risk_exposure_score", 0))
    fig_fkg = build_financial_knowledge_graph(obs)
    fig_equity = build_equity_curve(latest_ep.get("history", {}))
    fig_dd = build_drawdown_chart(history)
    fig_scatter = build_risk_return_scatter(results_all)
    fig_sector = build_sector_heatmap()
    
    # 2. Intel Panels
    rl_step_html = build_rl_step_monitor(history)
    rl_stats_html = build_advanced_rl_stats(obs)
    
    xai_data = explain_decision_advanced(asset, obs, history[-1]["action"] if history else 2)
    xai_html = build_xai_thinking_advanced(xai_data)
    
    cfg = load_config()
    from env.strategy_engine import StrategyDSLEngine
    engine = StrategyDSLEngine()
    rules = engine.parse_rules(cfg["strategy_sandbox"].get("default_rules", ""))
    lab_html = build_strategy_lab_status(rules, cfg["strategy_sandbox"].get("hybrid_mode", True))
    
    # Summary Generation
    if summary_type == "LLM-powered":
        summary_text = generate_llm_summary(data)
    else:
        summary_text = generate_rule_based_summary(data)
        
    # 3. Trade History
    trade_df = pd.DataFrame(obs.get("trade_history_summary", []))
    
    return (
        obs.get("portfolio_value", 0), obs.get("unrealized_pnl", 0), obs.get("volatility_index", 0),
        rl_step_html, xai_html, rl_stats_html, lab_html,
        fig_candle, fig_pnl, fig_pie, fig_risk, fig_fkg, fig_equity, fig_dd, fig_scatter, fig_sector,
        trade_df, summary_text
    )

# ── Main UI ───────────────────────────────────────────────────────────────────

with gr.Blocks(title="FinomIQ Terminal") as demo:
    gr.Markdown("# FinomIQ - Autonomous Financial Strategy Intelligence Platform")
    
    with gr.Row():
        # --- LEFT: Strategy Lab ---
        with gr.Column(scale=1):
            gr.Markdown("### Strategy Intelligence Lab")
            
            with gr.Tabs():
                with gr.Tab("Visual Builder"):
                    gr.Markdown("<small>Construct strategic rules via interface</small>")
                    with gr.Row():
                        v_indicator = gr.Dropdown(label="Indicator", choices=["sentiment", "vix", "trend", "drawdown", "liquidity"], value="sentiment")
                        v_operator = gr.Dropdown(label="Operator", choices=[">", "<", "=="], value=">")
                        v_value = gr.Number(label="Value", value=0.7)
                    with gr.Row():
                        v_action = gr.Dropdown(label="Action", choices=["BUY", "SELL", "HEDGE"], value="BUY")
                        v_amount = gr.Number(label="Amount %", value=10)
                        v_asset = gr.Dropdown(label="Asset", choices=["AAPL", "TSLA", "BTC", "GOLD", "ETH"], value="AAPL")
                    add_rule_btn = gr.Button("Add Rule to Strategy", size="sm")

                with gr.Tab("Strategy Library"):
                    template_sel = gr.Dropdown(label="Templates", choices=list(STRATEGY_TEMPLATES.keys()))
                    apply_template_btn = gr.Button("Load Template", size="sm")

            gr.Markdown("#### Active Strategy Rules (DSL)")
            rules_editor = gr.Code(
                label="Strategy Engine DSL",
                value="IF sentiment > 0.7 AND trend > 0.01:\n  BUY 10% AAPL\nIF drawdown > 0.05:\n  SELL 50% BTC",
                language="python",
                lines=10
            )
            clear_rules_btn = gr.Button("Clear All Rules", size="sm", variant="secondary")

            gr.Markdown("#### Simulation Scenario")
            m_type = gr.Dropdown(label="Market Regime", choices=["bull", "bear", "volatile", "crash"], value="bull")
            a_type = gr.Dropdown(label="AI Decision Core", choices=["ppo", "dqn", "hybrid"], value="ppo")
            hybrid_toggle = gr.Checkbox(label="Enable Hybrid Intelligence", value=True)
            
            run_btn = gr.Button("EXECUTE STRATEGY SIMULATION", variant="primary")
            
            gr.Markdown("---")
            summary_mode = gr.Radio(label="Summary Intelligence", choices=["Rule-based", "LLM-powered"], value="Rule-based")
            summary_display = gr.Markdown(label="Strategy Summary")
            
            gr.Markdown("---")
            lab_status_panel = gr.HTML()
            rl_stats_panel = gr.HTML()

        # --- RIGHT: Intelligence Terminal ---
        with gr.Column(scale=3):
            # Top Ribbon
            with gr.Row():
                equity_metric = gr.Number(label="Total Equity (USD)", precision=0)
                pnl_metric = gr.Number(label="Strategy PnL (USD)", precision=2)
                vix_metric = gr.Number(label="Market Volatility (VIX)", precision=1)
            
            rl_monitor_panel = gr.HTML()
            
            with gr.Tabs():
                with gr.Tab("Strategy Analytics"):
                    with gr.Row():
                        fig_equity = gr.Plot(label="Portfolio Equity Curve")
                        fig_pnl = gr.Plot(label="PnL Trajectory")
                    with gr.Row():
                        fig_dd = gr.Plot(label="Drawdown Analysis")
                        fig_scatter = gr.Plot(label="Risk-Return Efficiency")

                with gr.Tab("Market Depth"):
                    with gr.Row():
                        with gr.Column(scale=2):
                            fig_candle = gr.Plot(label="Asset Price Action")
                        with gr.Column(scale=1):
                            fig_risk = gr.Plot(label="VaR Risk Gauge")
                    with gr.Row():
                        fig_sector = gr.Plot(label="Sector Heatmap")
                        fig_pie = gr.Plot(label="Current Allocation")

                with gr.Tab("Neural Reasoning (XAI)"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            xai_panel = gr.HTML()
                        with gr.Column(scale=1):
                            fig_fkg = gr.Plot(label="Financial Knowledge Graph")
                    gr.Markdown("### What-If Analysis Engine")
                    gr.Info("Adjust parameters below to evaluate strategy performance under alternative market conditions.")
                    with gr.Row():
                        gr.Slider(label="Simulated Sentiment Shift", minimum=-0.5, maximum=0.5, value=0)
                        gr.Slider(label="Simulated Volatility Spike", minimum=0, maximum=50, value=0)

                with gr.Tab("Execution Log"):
                    trade_table = gr.DataFrame(label="Institutional Trade History")
                    log_viewer = gr.Code(label="Engine Console Output", lines=15)

    def get_logs():
        if os.path.exists(LOG_PATH):
            with open(LOG_PATH, "r") as f:
                return f.read()[-5000:]
        return "No logs found."

    def add_visual_rule(rules, indicator, op, val, action, amount, asset):
        new_rule = f"IF {indicator} {op} {val}:\n  {action} {amount}% {asset}"
        if rules.strip():
            return f"{rules}\n{new_rule}"
        return new_rule

    def load_template(template_name):
        return STRATEGY_TEMPLATES.get(template_name, "")

    def clear_rules():
        return ""

    def update_terminal(m, a, r, h, s_mode):
        data = run_intelligence_simulation(m, a, r, h)
        results = get_intelligence_data(s_mode)
        logs = get_logs()
        return (*results, logs)

    # Initial load
    demo.load(
        fn=lambda s_mode: get_intelligence_data(s_mode) + (get_logs(),),
        inputs=[summary_mode],
        outputs=[
            equity_metric, pnl_metric, vix_metric, 
            rl_monitor_panel, xai_panel, rl_stats_panel, lab_status_panel,
            fig_candle, fig_pnl, fig_pie, fig_risk, fig_fkg, fig_equity, fig_dd, fig_scatter, fig_sector,
            trade_table, summary_display, log_viewer
        ]
    )

    run_btn.click(
        fn=update_terminal,
        inputs=[m_type, a_type, rules_editor, hybrid_toggle, summary_mode],
        outputs=[
            equity_metric, pnl_metric, vix_metric, 
            rl_monitor_panel, xai_panel, rl_stats_panel, lab_status_panel,
            fig_candle, fig_pnl, fig_pie, fig_risk, fig_fkg, fig_equity, fig_dd, fig_scatter, fig_sector,
            trade_table, summary_display, log_viewer
        ]
    )

    add_rule_btn.click(
        fn=add_visual_rule,
        inputs=[rules_editor, v_indicator, v_operator, v_value, v_action, v_amount, v_asset],
        outputs=rules_editor
    )

    apply_template_btn.click(
        fn=load_template,
        inputs=template_sel,
        outputs=rules_editor
    )

    clear_rules_btn.click(
        fn=clear_rules,
        outputs=rules_editor
    )

if __name__ == "__main__":
    demo.launch()
