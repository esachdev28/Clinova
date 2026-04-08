"""
FinomIQ — Decision Engine (RL Agents).
Autonomous Hedge Fund Agents for market analysis and trading.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union
from env.models import Action
from env.logging_config import setup_logger

logger = setup_logger("DecisionEngine")

ACTION_NAMES = {
    0: "Buy Asset",
    1: "Sell Asset",
    2: "Hold Position",
    3: "Rebalance Portfolio",
    4: "Deep Market Analysis",
    5: "Hedge Risk",
    6: "Liquidate Portfolio",
}

class BaseAgent(ABC):
    """Abstract base class for FinomIQ hedge fund agents."""

    @abstractmethod
    def choose_action(self, observation: Dict) -> Action:
        pass

class MomentumAgent(BaseAgent):
    """
    Heuristic-based Momentum Trading Agent.
    Analyzes trend scores and sentiment to allocate capital.
    """

    def choose_action(self, observation: Dict) -> Action:
        step = observation.get("time_step", 0)
        sentiment = observation.get("news_sentiment_score", 0.5)
        trend = observation.get("market_trend_score", 0.0)
        vix = observation.get("volatility_index", 15.0)
        prices = observation.get("asset_prices", {})
        positions = observation.get("current_positions", {})
        cash = observation.get("cash_reserve", 0.0)
        assets = list(prices.keys())

        # ── Risk Management: High Volatility ──
        if vix > 25:
            logger.info(f"[Momentum] Step {step} | HIGH VIX ({vix:.2f}) — Hedging risk")
            return Action(action_type=5)

        # ── Strategy: Bullish Momentum ──
        if sentiment > 0.6 and trend > 0.01:
            # Pick an asset to buy
            target_asset = assets[int(step % len(assets))]
            # Allocate 10% of cash
            amount_to_buy = (cash * 0.1) / prices[target_asset]
            logger.info(f"[Momentum] Step {step} | BULLISH — Buying {target_asset}")
            return Action(action_type=0, asset_name=target_asset, amount=amount_to_buy)

        # ── Strategy: Bearish / Profit Taking ──
        if sentiment < 0.4 or trend < -0.01:
            for asset, units in positions.items():
                if units > 0:
                    logger.info(f"[Momentum] Step {step} | BEARISH — Selling {asset}")
                    return Action(action_type=1, asset_name=asset, amount=units * 0.5)

        # ── Default: Hold ──
        return Action(action_type=2)

class PPOHedgeFundAgent(BaseAgent):
    """
    RL-based Hedge Fund Agent (PPO).
    Learns optimal policy for multi-objective returns.
    """

    def __init__(self, lr: float = 0.0003, seed: int = 42):
        self.lr = lr
        self.rng = np.random.default_rng(seed)

    def choose_action(self, observation: Dict) -> Action:
        step = observation.get("time_step", 0)
        prices = observation.get("asset_prices", {})
        assets = list(prices.keys())
        
        # Simple weighted logic for demonstration (simulating PPO output)
        logits = self.rng.standard_normal(7)
        # Penalize liquidation early on
        if step < 20: logits[6] = -1e9
        
        probs = np.exp(logits - np.max(logits))
        probs /= np.sum(probs)
        
        action_type = int(self.rng.choice(7, p=probs))
        asset_name = self.rng.choice(assets) if assets else None
        
        # Determine amount based on action
        amount = 0.0
        if action_type == 0: # Buy
            amount = (observation.get("cash_reserve", 0) * 0.05) / prices[asset_name]
        elif action_type == 1: # Sell
            amount = observation.get("current_positions", {}).get(asset_name, 0) * 0.5
            
        logger.info(f"[PPO] Step {step} → {ACTION_NAMES[action_type]} on {asset_name}")
        return Action(action_type=action_type, asset_name=asset_name, amount=amount)

def get_agent(agent_type: str, config: Dict) -> BaseAgent:
    """Factory to return the appropriate FinomIQ agent."""
    t = agent_type.lower()
    seed = config["scenario"].get("seed", 42)
    lr = config["agent"].get("learning_rate", 0.0003)

    if t == "momentum":
        return MomentumAgent()
    elif t in ("ppo", "dqn"):
        return PPOHedgeFundAgent(lr=lr, seed=seed)
    else:
        return MomentumAgent()
