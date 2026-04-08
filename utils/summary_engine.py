"""
FinomIQ — Institutional Summary Engine.
Generates strategy performance reports using rule-based logic or simulated LLM synthesis.
"""

from typing import Dict, List, Any

def generate_rule_based_summary(data: Dict) -> str:
    """Generates a technical performance summary based on quantitative thresholds."""
    latest_ep = data["episodes"][-1]
    pnl = latest_ep.get("unrealized_pnl", 0)
    steps = latest_ep.get("steps", 0)
    obs = latest_ep.get("observation", {})
    
    alpha = obs.get("alpha", 0)
    sharpe = obs.get("sharpe_ratio", 0)
    max_dd = obs.get("max_drawdown", 0)
    
    performance = "POSITIVE" if pnl > 0 else "NEGATIVE"
    risk_profile = "CONSERVATIVE" if max_dd < 0.05 else "AGGRESSIVE"
    
    summary = f"STRATEGY REPORT: {performance} alpha generation observed over {steps} steps. "
    summary += f"The portfolio achieved an unrealized PnL of ${pnl:,.2f} with a Sharpe ratio of {sharpe:.2f}. "
    summary += f"Risk management was {risk_profile} with a maximum drawdown of {max_dd:.2%}. "
    
    if alpha > 0.01:
        summary += "Strategy successfully exploited market inefficiencies to generate excess returns."
    else:
        summary += "Strategy performance was largely driven by beta exposure; alpha decay detected."
        
    return summary

def generate_llm_summary(data: Dict) -> str:
    """Simulates an LLM 'Market Analyst' summary with qualitative reasoning."""
    latest_ep = data["episodes"][-1]
    obs = latest_ep.get("observation", {})
    pnl = latest_ep.get("unrealized_pnl", 0)
    
    market_regime = obs.get("regime", "Unknown")
    sentiment = obs.get("news_sentiment_score", 0.5)
    
    regime_desc = "bullish breakout" if market_regime == "bull" else "bearish correction" if market_regime == "bear" else "high volatility"
    sentiment_desc = "optimistic news flow" if sentiment > 0.6 else "pessimistic headlines" if sentiment < 0.4 else "neutral commentary"
    
    summary = f"### Institutional Analyst Review\n\n"
    summary += f"Our proprietary LLM-driven core has analyzed the strategy execution during the recent **{regime_desc}**. "
    summary += f"Despite the **{sentiment_desc}**, the agent maintained a total PnL of **${pnl:,.2f}**.\n\n"
    summary += "#### Strategic Insight\n"
    summary += "The RL agent demonstrated high adaptive capacity, rotating capital into defensive assets during peak volatility. "
    summary += "We noticed a strong correlation between news sentiment shifts and the agent's rebalancing frequency. "
    summary += "Recommendation: Increase the risk limit to allow the PPO core more 'exploration space' in low-vix environments."
    
    return summary
