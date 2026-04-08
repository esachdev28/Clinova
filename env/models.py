"""
FinomIQ — Autonomous Hedge Fund Intelligence Platform models.
"""

from pydantic import BaseModel
from typing import Dict, List, Optional


class Observation(BaseModel):
    """Expanded Decision Intelligence Observation Space."""

    asset_prices: Dict[str, float]
    volatility_index: float
    market_trend_score: float
    liquidity_index: float
    portfolio_value: float
    cash_reserve: float
    current_positions: Dict[str, float]
    unrealized_pnl: float
    risk_exposure_score: float
    macro_indicators: Dict[str, float]
    news_sentiment_score: float
    correlation_matrix: List[List[float]]
    trade_history_summary: List[Dict]
    time_step: int
    remaining_budget: int
    alpha: float = 0.0
    beta: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0


class Action(BaseModel):
    """Decision Space for FinomIQ Agents."""

    action_type: int  # 0:Buy, 1:Sell, 2:Hold, 3:Rebalance, 4:Analyze, 5:Hedge, 6:Liquidate
    asset_name: Optional[str] = None
    amount: Optional[float] = None
    strategy_id: Optional[str] = "ai_agent"  # Tracks which strategy triggered the action


class Reward(BaseModel):
    """Multi-objective reward for financial performance."""

    value: float  # Strictly between (0.0001, 0.9999)
    reason: str
    metrics: Dict[str, float]  # profit_growth, sharpe_ratio, etc.


class StepResult(BaseModel):
    """Result returned by env.step()."""

    observation: Observation
    reward: float
    done: bool
    info: Dict
