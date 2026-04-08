---
title: FinomIQ — Autonomous Financial Strategy Intelligence Platform
emoji: none
colorFrom: indigo
colorTo: blue
sdk: docker
pinned: true
license: mit
tags:
  - openenv
---

# FinomIQ — Autonomous Financial Strategy Intelligence Platform

**FinomIQ** is a production-grade decision intelligence system for autonomous financial strategy design, simulation, and evolution. It provides a comprehensive Strategy Sandbox where users can construct complex, rule-based strategies that interact with AI-driven market environments.

---

## System Architecture

FinomIQ is built on a modular, multi-layered architecture:

### 1. Strategy Sandbox Engine (CORE)
*   Strategy DSL: A rule-based Domain-Specific Language for defining financial logic.
*   Hybrid Execution: Integrates user-defined rules with state-of-the-art RL agents (PPO, DQN).
*   What-If Analysis: Evaluates strategy performance under alternative market conditions.

### 2. Market Simulation Engine
*   GBM + Jump Diffusion: High-fidelity price simulation with stochastic jumps and mean reversion.
*   Regime Switching: Dynamic transitions between Bull, Bear, Volatile, and Crash regimes.

### 3. Portfolio & Risk Intelligence
*   Institutional Metrics: Real-time tracking of Alpha, Beta, Sharpe Ratio, and Sortino Ratio.
*   Advanced Analytics: Peak-to-trough Drawdown analysis and Risk-Return Efficiency profiling.

---

## 📊 Observation & Action Spaces

### Observation Space
The agent receives a rich state vector representing institutional market data:
*   `asset_prices`: Current price of all tradable assets.
*   `volatility_index`: VIX-like market uncertainty indicator.
*   `market_trend_score`: Momentum signal.
*   `portfolio_value`: Total valuation of holdings and cash.
*   `risk_exposure_score`: Current peak-to-trough drawdown.
*   `news_sentiment_score`: NLP-derived alternative data signal.
*   `macro_indicators`: Inflation and Interest Rate proxies.

### Action Space
The agent can execute seven institutional-grade financial operations:
*   `0: BUY`: Allocate capital to a specific asset.
*   `1: SELL`: Exit or reduce positions.
*   `2: HOLD`: Maintain current allocation.
*   `3: REBALANCE`: Equal-weight portfolio realignment.
*   `4: ANALYZE`: Spend budget to improve signal clarity.
*   `5: HEDGE`: Execute defensive moves using non-correlated assets.
*   `6: LIQUIDATE`: Complete exit to cash reserve.

---

## 🧪 OpenEnv Tasks

| Task ID | Name | Difficulty | Objective |
| :--- | :--- | :--- | :--- |
| `bull_market_growth` | Bull Market Growth | **Easy** | Maximize profit in a steady upward trending market. |
| `volatile_market_navigation` | Volatile Navigation | **Medium** | Achieve high Sharpe ratio despite extreme price swings. |
| `market_crash_survival` | Crash Survival | **Hard** | Protect capital and survive a severe market liquidation event. |

---

## 📈 Baseline Performance

Results from the `inference.py` baseline script:

| Task | Baseline Score (0-1) | Status |
| :--- | :--- | :--- |
| Bull Market Growth | 0.85 | PASS |
| Volatile Navigation | 0.62 | PASS |
| Crash Survival | 0.45 | EVAL |

---

## 🚀 Execution

### Institutional Terminal (Gradio)
```bash
python3 gradio_app.py
```

### OpenEnv Inference Script
```bash
python3 inference.py
```

---

**FinomIQ** — *The future of autonomous financial strategy intelligence.*
