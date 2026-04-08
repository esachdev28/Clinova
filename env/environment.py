"""
FinomIQ — Autonomous Financial Strategy Intelligence Platform.
Advanced Market Simulation, Portfolio Core, and Strategy Sandbox.
"""

import os
import numpy as np
import yaml
import pandas as pd
from typing import Dict, List, Any, Optional
from env.models import Action, Observation, StepResult
from env.logging_config import setup_logger
from env.strategy_engine import StrategyDSLEngine

from env.graders import get_grader

logger = setup_logger("FinomIQEnv")

class FinomIQEnv:
    """Production-grade RL platform for financial strategy intelligence."""

    def __init__(self, config_path: str = "config.yaml", task_id: str = "bull_market_growth") -> None:
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.task_id = task_id
        self.assets = self.config["scenario"]["assets"]
        self.initial_market_type = self._get_market_regime_for_task(task_id)
        self.max_steps = self.config["scenario"]["max_steps"]
        self.base_seed = self.config["scenario"]["seed"]
        
        self.transaction_cost = self.config["constraints"]["transaction_cost"]
        self.slippage = self.config["constraints"]["slippage"]
        self.risk_limit = self.config["constraints"]["risk_limit"]
        self.capital_limit = self.config["constraints"]["capital_limit"]

        self.dsl_engine = StrategyDSLEngine()
        self.user_rules = self.dsl_engine.parse_rules(self.config["strategy_sandbox"].get("default_rules", ""))
        
        self.episode_count = 0
        self._reset_state()

    def _get_market_regime_for_task(self, task_id: str) -> str:
        task_regimes = {
            "bull_market_growth": "bull",
            "volatile_market_navigation": "volatile",
            "market_crash_survival": "crash"
        }
        return task_regimes.get(task_id, "bull")

    def _reset_state(self) -> None:
        self.rng = np.random.default_rng(seed=self.base_seed + self.episode_count)
        self.step_count = 0
        self.done = False
        self.market_type = self.initial_market_type

        # Portfolio
        self.cash = float(self.capital_limit)
        self.positions = {asset: 0.0 for asset in self.assets}
        self.portfolio_value = self.cash
        self.initial_portfolio_value = self.cash
        self.trade_history = []
        
        # Market
        self.asset_prices = {asset: self._get_initial_price(asset) for asset in self.assets}
        self.vix = 15.0
        self.sentiment = 0.5
        self.liquidity = 0.95
        self.market_trend = self._get_regime_drift()
        
        # Performance Tracking
        self.history = {
            "portfolio_value": [self.portfolio_value],
            "returns": [],
            "drawdowns": [0.0],
            "market_returns": []
        }

        logger.info(f"FINOMIQ INTEL RESET | Ep#{self.episode_count} | Regime: {self.market_type}")

    def _get_initial_price(self, asset: str) -> float:
        prices = {"AAPL": 150.0, "TSLA": 250.0, "BTC": 45000.0, "GOLD": 2000.0, "ETH": 2500.0}
        return prices.get(asset, 100.0) * (1 + self.rng.uniform(-0.05, 0.05))

    def _get_regime_drift(self) -> float:
        drifts = {"bull": 0.05, "bear": -0.05, "volatile": 0.01, "crash": -0.15}
        return drifts.get(self.market_type, 0.0)

    def _simulate_market(self) -> None:
        """Advanced Market Engine: GBM + Jump Diffusion + Regime Switching."""
        
        # 1. Regime Switching (Probabilistic)
        if self.rng.random() < 0.02: # 2% chance to switch regime each step
            regimes = ["bull", "bear", "volatile", "crash"]
            self.market_type = self.rng.choice(regimes)
            self.market_trend = self._get_regime_drift()
            logger.info(f"MARKET REGIME SWITCH -> {self.market_type}")

        # 2. Jump Diffusion Parameters
        jump_lambda = 0.01 # Jump frequency
        jump_sigma = 0.05   # Jump magnitude
        
        vol_scale = 0.01 if self.market_type == "bull" else 0.03 if self.market_type == "volatile" else 0.05
        
        for asset in self.assets:
            # GBM component
            drift = self.market_trend / self.max_steps
            diffusion = self.rng.normal(0, vol_scale)
            
            # Jump component
            jump = 0
            if self.rng.random() < jump_lambda:
                jump = self.rng.normal(0, jump_sigma)
            
            # Update price
            self.asset_prices[asset] *= (1 + drift + diffusion + jump)
            self.asset_prices[asset] = max(0.01, self.asset_prices[asset])

        # 3. Update Indicators
        self.vix = np.clip(self.vix + self.rng.normal(0, 1), 10, 50)
        self.sentiment = np.clip(self.sentiment + self.rng.normal(0, 0.05), 0, 1)
        self.liquidity = np.clip(self.liquidity + self.rng.normal(0, 0.01), 0.5, 1.0)

    async def step(self, action: Action) -> dict:
        if self.done:
            return StepResult(observation=self._get_observation(), reward=0.5, done=True, info={}).model_dump()

        self.step_count += 1
        
        # 1. Market Evolution
        self._simulate_market()

        # 2. User Strategy Execution (DSL)
        triggered_rules = self.dsl_engine.evaluate(self.user_rules, self._get_raw_obs())
        for rule_act in triggered_rules:
            self._execute_action(rule_act, strategy_id="user_rules")

        # 3. AI Agent Action Execution
        self._execute_action(action.model_dump(), strategy_id="ai_agent")

        # 4. Update Portfolio Value
        asset_val = sum(self.positions[a] * self.asset_prices[a] for a in self.assets)
        self.portfolio_value = self.cash + asset_val
        
        # 5. Metrics & History
        ret = (self.portfolio_value / self.history["portfolio_value"][-1]) - 1
        self.history["portfolio_value"].append(self.portfolio_value)
        self.history["returns"].append(ret)
        
        peak = max(self.history["portfolio_value"])
        drawdown = (peak - self.portfolio_value) / peak
        self.history["drawdowns"].append(drawdown)

        # 6. Reward Calculation
        reward_val, reward_metrics = self._calculate_reward()

        # 7. Termination
        if self.step_count >= self.max_steps or drawdown > self.risk_limit:
            self.done = True

        return StepResult(
            observation=self._get_observation(),
            reward=reward_val,
            done=self.done,
            info={"regime": self.market_type, "metrics": reward_metrics}
        ).model_dump()

    def _execute_action(self, act: Dict, strategy_id: str) -> None:
        a_type = act.get("action_type", 2)
        asset = act.get("asset_name")
        amount_pct = act.get("amount_pct", 0.0)
        
        if a_type == 0: # BUY
            if not asset or asset not in self.assets: return
            price = self.asset_prices[asset] * (1 + self.slippage)
            # If amount_pct provided (from DSL), use % of cash. Else use amount from Action.
            if amount_pct > 0:
                cost = self.cash * amount_pct
            else:
                cost = act.get("amount", 0) * price
            
            total_cost = cost * (1 + self.transaction_cost)
            if self.cash >= total_cost and total_cost > 0:
                self.cash -= total_cost
                units = cost / price
                self.positions[asset] += units
                self.trade_history.append({"step": self.step_count, "type": "BUY", "asset": asset, "units": units, "price": price, "strategy": strategy_id})

        elif a_type == 1: # SELL
            if not asset or asset not in self.assets: return
            price = self.asset_prices[asset] * (1 - self.slippage)
            if amount_pct > 0:
                units = self.positions[asset] * amount_pct
            else:
                units = min(act.get("amount", 0), self.positions[asset])
            
            if units > 0:
                revenue = units * price
                self.cash += revenue * (1 - self.transaction_cost)
                self.positions[asset] -= units
                self.trade_history.append({"step": self.step_count, "type": "SELL", "asset": asset, "units": units, "price": price, "strategy": strategy_id})

        elif a_type == 5: # HEDGE (Simulated defensive move)
            if not asset or asset not in self.assets: return
            # Defensive assets like GOLD get 20% allocation automatically
            self._execute_action({"action_type": 0, "asset_name": asset, "amount_pct": amount_pct or 0.2}, strategy_id=f"{strategy_id}_hedge")

        elif a_type == 6: # LIQUIDATE
            for a in self.assets:
                if self.positions[a] > 0:
                    self._execute_action({"action_type": 1, "asset_name": a, "amount_pct": 1.0}, strategy_id=f"{strategy_id}_liq")

    def _calculate_reward(self) -> (float, dict):
        total_return = (self.portfolio_value / self.initial_portfolio_value) - 1
        
        # Sharpe
        rets = self.history["returns"]
        sharpe = (np.mean(rets) / np.std(rets)) if len(rets) > 5 and np.std(rets) > 0 else 0
        
        # Stability (Negative of returns variance)
        stability = 1.0 - np.clip(np.std(rets) * 10, 0, 1) if rets else 1.0
        
        # Drawdown Penalty
        dd = self.history["drawdowns"][-1]
        
        # Diversification
        non_zero = sum(1 for p in self.positions.values() if p > 0)
        div = non_zero / len(self.assets)

        weights = self.config["rewards"]
        reward = (
            weights["profit_growth_weight"] * np.tanh(total_return * 5) +
            weights["sharpe_ratio_weight"] * np.clip(sharpe, -1, 1) +
            weights["stability_weight"] * stability +
            weights["diversification_weight"] * div -
            weights["drawdown_penalty_weight"] * dd
        )
        
        # STRICT RULE: Normalize to (0, 1) and clip [0.0001, 0.9999]
        reward = (np.tanh(reward) + 1) / 2
        reward = np.clip(reward, 0.0001, 0.9999)
        
        return float(reward), {"return": total_return, "sharpe": sharpe, "stability": stability}

    def _get_raw_obs(self) -> Dict:
        return {
            "sentiment": self.sentiment,
            "market_trend": self.market_trend,
            "volatility_index": self.vix,
            "liquidity_index": self.liquidity,
            "portfolio_value": self.portfolio_value,
            "cash_reserve": self.cash,
            "risk_exposure_score": self.history["drawdowns"][-1]
        }

    def _get_observation(self) -> Observation:
        raw = self._get_raw_obs()
        return Observation(
            asset_prices=self.asset_prices,
            volatility_index=self.vix,
            market_trend_score=self.market_trend,
            liquidity_index=self.liquidity,
            portfolio_value=self.portfolio_value,
            cash_reserve=self.cash,
            current_positions=self.positions,
            unrealized_pnl=self.portfolio_value - self.initial_portfolio_value,
            risk_exposure_score=self.history["drawdowns"][-1],
            macro_indicators={"inflation": 0.03, "interest_rate": 0.05},
            news_sentiment_score=self.sentiment,
            correlation_matrix=[[1.0 if i==j else 0.5 for j in range(len(self.assets))] for i in range(len(self.assets))],
            trade_history_summary=self.trade_history[-5:],
            time_step=self.step_count,
            remaining_budget=self.max_steps - self.step_count,
            alpha=0.02, # Simulated alpha
            beta=0.85,  # Simulated beta
            sharpe_ratio=1.5,
            sortino_ratio=1.8,
            max_drawdown=max(self.history["drawdowns"])
        )

    async def reset(self) -> dict:
        self.episode_count += 1
        self._reset_state()
        return self._get_observation().model_dump()

    async def state(self) -> dict:
        obs = self._get_observation().model_dump()
        obs.update({"history": self.history, "trade_history": self.trade_history})
        return obs

    def grade(self) -> float:
        """Evaluate performance for the current task."""
        grader = get_grader(self.task_id)
        return grader(self.history)
