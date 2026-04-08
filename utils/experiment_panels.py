"""
FinomIQ — Strategy Intelligence Panels.
Advanced UI components for decision analysis and strategy lab.
"""

from typing import Dict, List, Any

def build_strategy_lab_status(rules: List[Dict], hybrid_mode: bool) -> str:
    """Strategy Sandbox Status Panel."""
    mode_label = "HYBRID (AI + RULES)" if hybrid_mode else "RULE-BASED"
    mode_color = "#9D00FF" if hybrid_mode else "#00D4FF"
    # Match badge background to the mode color (10% alpha)
    badge_bg = "rgba(157, 0, 255, 0.1)" if hybrid_mode else "rgba(0, 212, 255, 0.1)"
    
    rules_html = "".join([f"<li style='margin-bottom: 5px; color: #E0E0E0;'>{r['condition']}</li>" for r in rules]) or "<li style='color: #888;'>No active rules</li>"
    
    return f"""
    <div style="padding: 20px; background: #121212; border: 1px solid #333; border-radius: 8px; font-family: 'Inter', sans-serif;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
            <span style="font-size: 0.8rem; font-weight: 700; color: {mode_color}; text-transform: uppercase; letter-spacing: 1px;">Strategy Engine: {mode_label}</span>
            <span style="padding: 4px 8px; background: {badge_bg}; color: {mode_color}; border-radius: 4px; font-size: 0.7rem;">ACTIVE</span>
        </div>
        <div style="margin-top: 10px;">
            <div style="font-size: 0.7rem; color: #888; text-transform: uppercase; margin-bottom: 10px;">Loaded Strategy Rules</div>
            <ul style="font-size: 0.85rem; padding-left: 15px;">
                {rules_html}
            </ul>
        </div>
    </div>
    """

def build_advanced_rl_stats(obs: Dict) -> str:
    """Institutional Metrics Panel (Alpha, Beta, Sharpe)."""
    return f"""
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; font-family: 'Inter', sans-serif;">
        <div style="padding: 15px; background: #121212; border: 1px solid #333; border-radius: 8px; text-align: center;">
            <div style="font-size: 0.65rem; color: #888; text-transform: uppercase;">Alpha (vs Bench)</div>
            <div style="font-size: 1.5rem; font-weight: 700; color: #00FF41;">{obs.get('alpha', 0):+.2%}</div>
        </div>
        <div style="padding: 15px; background: #121212; border: 1px solid #333; border-radius: 8px; text-align: center;">
            <div style="font-size: 0.65rem; color: #888; text-transform: uppercase;">Portfolio Beta</div>
            <div style="font-size: 1.5rem; font-weight: 700; color: #00D4FF;">{obs.get('beta', 0):.2f}</div>
        </div>
        <div style="padding: 15px; background: #121212; border: 1px solid #333; border-radius: 8px; text-align: center;">
            <div style="font-size: 0.65rem; color: #888; text-transform: uppercase;">Sharpe Ratio</div>
            <div style="font-size: 1.5rem; font-weight: 700; color: #9D00FF;">{obs.get('sharpe_ratio', 0):.2f}</div>
        </div>
        <div style="padding: 15px; background: #121212; border: 1px solid #333; border-radius: 8px; text-align: center;">
            <div style="font-size: 0.65rem; color: #888; text-transform: uppercase;">Sortino Ratio</div>
            <div style="font-size: 1.5rem; font-weight: 700; color: #E0E0E0;">{obs.get('sortino_ratio', 0):.2f}</div>
        </div>
    </div>
    """

def build_xai_thinking_advanced(xai: Dict) -> str:
    """Advanced XAI thinking panel with confidence and counterfactuals."""
    factors = xai.get("factors", [])
    factor_html = ""
    for f in factors:
        score_pct = f['score'] * 100
        factor_html += f"""
        <div style="margin-bottom: 12px;">
            <div style="display: flex; justify-content: space-between; font-size: 0.8rem; margin-bottom: 4px;">
                <span>{f['label']}</span>
                <span style="color: #888;">{f['reason']}</span>
            </div>
            <div style="height: 4px; background: #222; border-radius: 2px; overflow: hidden;">
                <div style="width: {score_pct}%; height: 100%; background: #00D4FF;"></div>
            </div>
        </div>
        """
        
    return f"""
    <div style="padding: 20px; background: #121212; border: 1px solid #333; border-radius: 8px; font-family: 'Inter', sans-serif;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
            <span style="font-size: 0.8rem; font-weight: 700; color: #9D00FF; text-transform: uppercase;">Decision Intelligence</span>
            <span style="font-size: 0.8rem; color: #00FF41; font-weight: bold;">{xai.get('confidence', 0):.0%} Conf.</span>
        </div>
        <p style="font-size: 0.95rem; color: #E0E0E0; line-height: 1.5; margin-bottom: 15px; border-left: 2px solid #9D00FF; padding-left: 10px;">
            {xai.get('summary', '')}
        </p>
        <div style="font-size: 0.7rem; color: #888; text-transform: uppercase; margin-bottom: 10px;">Feature Weights</div>
        {factor_html}
        <div style="margin-top: 15px; padding-top: 15px; border-top: 1px solid #333;">
            <div style="font-size: 0.7rem; color: #888; text-transform: uppercase; margin-bottom: 5px;">What-If Analysis (Counterfactual)</div>
            <div style="font-size: 0.85rem; color: #FF3131; font-style: italic;">{xai.get('counterfactual', '')}</div>
        </div>
    </div>
    """

def build_rl_step_monitor(history: List[Dict]) -> str:
    """Visualizes the RL agent's step-by-step decision trace."""
    rows_html = ""
    # Show last 10 steps
    for item in history[-10:]:
        act_type = item.get("action", 2)
        act_map = {0: "BUY", 1: "SELL", 2: "HOLD", 3: "REBAL", 4: "ANALYZ", 5: "HEDGE", 6: "LIQUID"}
        act_name = act_map.get(act_type, "HOLD")
        
        # Color based on action
        color = "#00FF41" if act_type == 0 else "#FF3131" if act_type == 1 else "#00D4FF" if act_type == 5 else "#888"
        reward = item.get("reward", 0.5)
        pnl = item.get("unrealized_pnl", 0)
        
        rows_html += f"""
        <tr style="border-bottom: 1px solid #222;">
            <td style="padding: 8px; font-family: 'JetBrains Mono'; font-size: 0.75rem; color: #888;">{item.get('step', 0)}</td>
            <td style="padding: 8px; font-weight: 700; color: {color}; font-size: 0.8rem;">{act_name}</td>
            <td style="padding: 8px; font-size: 0.8rem; color: #E0E0E0;">{item.get('asset', 'N/A')}</td>
            <td style="padding: 8px; font-family: 'JetBrains Mono'; font-size: 0.8rem; color: {'#00FF41' if reward > 0.5 else '#FF3131'};">{reward:.4f}</td>
            <td style="padding: 8px; font-family: 'JetBrains Mono'; font-size: 0.8rem; color: {'#00FF41' if pnl > 0 else '#FF3131'};">${pnl:,.0f}</td>
        </tr>
        """

    return f"""
    <div style="background: rgba(18, 18, 18, 0.6); backdrop-filter: blur(10px); border: 1px solid #333; border-radius: 8px; overflow: hidden;">
        <div style="padding: 10px 15px; background: rgba(0, 212, 255, 0.1); border-bottom: 1px solid #333; display: flex; justify-content: space-between; align-items: center;">
            <span style="font-size: 0.7rem; font-weight: 700; color: #00D4FF; text-transform: uppercase;">RL Policy Trace (Live)</span>
            <span style="font-size: 0.6rem; color: #888;">Showing last 10 steps</span>
        </div>
        <table style="width: 100%; border-collapse: collapse; text-align: left;">
            <thead>
                <tr style="background: #1a1a1a; font-size: 0.65rem; color: #666; text-transform: uppercase;">
                    <th style="padding: 8px;">Step</th>
                    <th style="padding: 8px;">Decision</th>
                    <th style="padding: 8px;">Asset</th>
                    <th style="padding: 8px;">Reward</th>
                    <th style="padding: 8px;">PnL</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
    </div>
    """
