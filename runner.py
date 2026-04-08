import argparse
import asyncio
import json
import yaml
import time
import numpy as np
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from env.environment import FinomIQEnv
from env.agents import get_agent
from env.models import Action
from env.logging_config import setup_logger

logger = setup_logger("Runner")


class SimulationRunner:
    """Orchestrates automated runs for FinomIQ Hedge Fund Intelligence."""

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.env = FinomIQEnv(config_path)
        self.agent = get_agent(self.config["agent"]["type"], self.config)
        self.num_episodes = self.config["scenario"]["num_episodes"]
        self.results_path = Path(self.config["visualization"].get("persistence_path", "results/finomiq_run.json"))
        self.results_path.parent.mkdir(parents=True, exist_ok=True)

    async def run_episode(self, episode_idx: int) -> dict:
        """Execute a single episode simulation."""
        logger.info(f"═══════════════════════════════════════════")
        logger.info(f"EPISODE {episode_idx + 1}/{self.num_episodes} — START")
        logger.info(f"═══════════════════════════════════════════")

        observation = await self.env.reset()
        done = False
        episode_reward = 0
        steps = 0
        action_history = []

        while not done:
            action = self.agent.choose_action(observation)
            
            result = await self.env.step(action)
            observation = result["observation"]
            episode_reward += result["reward"]
            done = result["done"]
            steps += 1

            action_history.append({
                "step": steps,
                "action": action.action_type,
                "asset": action.asset_name,
                "amount": action.amount,
                "reward": round(result["reward"], 4),
                "portfolio_value": observation["portfolio_value"],
                "unrealized_pnl": observation["unrealized_pnl"],
                "asset_prices": observation["asset_prices"].copy(),
            })

            if done:
                logger.info(f"───────────────────────────────────────────")
                logger.info(f"EPISODE {episode_idx + 1} RESULT: PnL=${observation['unrealized_pnl']:.2f} | Steps={steps}, Reward={episode_reward:.2f}")
                logger.info(f"───────────────────────────────────────────")

                return {
                    "episode": episode_idx + 1,
                    "reward": round(episode_reward, 2),
                    "steps": steps,
                    "final_portfolio_value": observation["portfolio_value"],
                    "unrealized_pnl": observation["unrealized_pnl"],
                    "action_history": action_history,
                    "observation": observation,
                    "history": self.env.history
                }
        return {}

    async def run_all(self):
        """Execute the batch of episodes and produce a rich summary."""
        start_time = time.time()
        logger.info(f"--- FinomIQ Autonomous Hedge Fund Simulation ---")
        logger.info(f"Market: {self.config['scenario']['market_type']} | Agent: {self.config['agent']['type']} | Episodes: {self.num_episodes}")

        results = []
        for i in tqdm(range(self.num_episodes)):
            res = await self.run_episode(i)
            results.append(res)

        elapsed = round(time.time() - start_time, 2)

        # ── Aggregate metrics ──
        rewards = [r["reward"] for r in results]
        pnls = [r["unrealized_pnl"] for r in results]
        steps_list = [r["steps"] for r in results]

        avg_reward = round(sum(rewards) / len(rewards) if rewards else 0, 2)
        avg_pnl = round(sum(pnls) / len(pnls) if pnls else 0, 2)
        avg_steps = round(sum(steps_list) / len(steps_list) if steps_list else 0, 1)

        summary = {
            "run_metadata": {
                "timestamp": datetime.now().isoformat(),
                "elapsed_seconds": elapsed,
                "market_type": self.config["scenario"]["market_type"],
                "agent_type": self.config["agent"]["type"],
                "num_episodes": self.num_episodes,
                "max_steps": self.config["scenario"]["max_steps"],
                "seed": self.config["scenario"]["seed"],
            },
            "metrics": {
                "avg_reward": avg_reward,
                "avg_pnl": avg_pnl,
                "avg_steps": avg_steps,
                "total_profit": sum(pnls),
            },
            "episodes": results,
            "config": self.config,
        }

        with open(self.results_path, "w") as f:
            json.dump(summary, f, indent=2)

        # ── Console summary ──
        logger.info(f"")
        logger.info(f"╔══════════════════════════════════════════════════════════╗")
        logger.info(f"║           FinomIQ SIMULATION SUMMARY REPORT             ║")
        logger.info(f"╠══════════════════════════════════════════════════════════╣")
        logger.info(f"║  Market Regime: {self.config['scenario']['market_type']:<42}║")
        logger.info(f"║  Agent Model:   {self.config['agent']['type']:<42}║")
        logger.info(f"║  Episodes:      {self.num_episodes:<42}║")
        logger.info(f"║  Avg PnL:       ${avg_pnl:<41}║")
        logger.info(f"║  Runtime:       {elapsed}s{' ' * (40 - len(str(elapsed)))}║")
        logger.info(f"╚══════════════════════════════════════════════════════════╝")
        logger.info(f"Results saved to: {self.results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    runner = SimulationRunner(args.config)
    asyncio.run(runner.run_all())
