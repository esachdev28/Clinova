"""
FinomIQ — Multi-Agent Competition System.
Multiple hedge fund agents competing in the same market environment.
"""

import numpy as np
from typing import Dict
from env.agents import BaseAgent
from env.models import Action
from env.logging_config import setup_logger

logger = setup_logger("MultiAgent")

class ArbitrageAgent(BaseAgent):
    """Focuses on high-frequency price discrepancies and liquidity."""
    def choose_action(self, observation: Dict) -> Action:
        step = observation.get("time_step", 0)
        prices = observation.get("asset_prices", {})
        liquidity = observation.get("liquidity_index", 0.9)
        
        if liquidity > 0.9:
            # Simple arbitrage logic: buy lowest price asset
            target = min(prices, key=prices.get)
            amount = (observation.get("cash_reserve", 0) * 0.05) / prices[target]
            logger.info(f"[Arbitrage] Step {step} | Buying {target} (liquidity={liquidity:.2f})")
            return Action(action_type=0, asset_name=target, amount=amount)
        
        return Action(action_type=2)

class ValueAgent(BaseAgent):
    """Focuses on macro indicators and long-term fundamental value."""
    def choose_action(self, observation: Dict) -> Action:
        step = observation.get("time_step", 0)
        macro = observation.get("macro_indicators", {})
        inflation = macro.get("inflation", 0.03)
        
        if inflation < 0.04:
            # Bullish on value assets
            target = "AAPL"
            amount = (observation.get("cash_reserve", 0) * 0.1) / observation["asset_prices"][target]
            logger.info(f"[Value] Step {step} | Bullish on {target} (Inflation={inflation:.2%})")
            return Action(action_type=0, asset_name=target, amount=amount)
            
        return Action(action_type=2)

class RiskManagerAgent(BaseAgent):
    """Focuses on capital preservation and defensive positioning."""
    def choose_action(self, observation: Dict) -> Action:
        step = observation.get("time_step", 0)
        vix = observation.get("volatility_index", 15.0)
        drawdown = observation.get("risk_exposure_score", 0.0)
        
        if vix > 25 or drawdown > 0.1:
            logger.info(f"[RiskManager] Step {step} | CRITICAL: VIX={vix:.1f}, Drawdown={drawdown:.2%}. Hedging.")
            return Action(action_type=5) # Hedge
            
        return Action(action_type=2)
