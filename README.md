# FinomIQ — Autonomous Financial Strategy Intelligence Platform

**FinomIQ** is a production-grade decision intelligence system for autonomous financial strategy design, simulation, and evolution. It provides a comprehensive Strategy Sandbox where users can construct complex, rule-based strategies that interact with AI-driven market environments.

---

## System Architecture

FinomIQ is built on a modular, multi-layered architecture:

### 1. Strategy Sandbox Engine (CORE)
*   **Strategy DSL**: A rule-based Domain-Specific Language for defining financial logic (e.g., `IF sentiment > 0.7 AND market_trend > 0.01: BUY 10% AAPL`).
*   **Hybrid Execution**: Integrates user-defined rules with state-of-the-art RL agents (PPO, DQN) for adaptive decision-making.
*   **What-If Analysis**: Evaluates strategy performance under alternative market conditions (Sentiment, Volatility shifts).

### 2. Market Simulation Engine
*   **GBM + Jump Diffusion**: High-fidelity price simulation with stochastic jumps and mean reversion.
*   **Regime Switching**: Dynamic transitions between Bull, Bear, Volatile, and Crash regimes.
*   **Market Reaction**: Markets respond to agent competition, liquidity constraints, and alternative data signals.

### 3. Portfolio & Risk Intelligence
*   **Institutional Metrics**: Real-time tracking of Alpha, Beta, Sharpe Ratio, and Sortino Ratio.
*   **Advanced Analytics**: Peak-to-trough Drawdown analysis and Risk-Return Efficiency profiling.
*   **Execution Core**: Modeling transaction costs, slippage, and institutional latency.

### 4. Explainable AI (XAI) 2.0
*   **Neural Reasoning**: Natural language explanations for every trade decision.
*   **Counterfactuals**: Evaluates alternative outcomes (e.g., "If news sentiment had been bearish, the system would have opted for a SELL action").
*   **Feature Weights**: Dynamic assessment of sentiment, momentum, and risk factor importance.

---

## Intelligence Terminal (UI/UX)

The platform features a professional dashboard with institutional styling:
*   **Strategy Lab**: The primary workspace for rule construction, scenario selection, and simulation execution.
*   **Strategy Analytics**: Performance comparison via equity curves and risk-return scatter plots.
*   **Market Depth**: Real-time candlestick charts, sector sentiment heatmaps, and VaR risk gauges.
*   **Decision Core**: Visualizing the Financial Knowledge Graph and the agent's internal reasoning process.

---

## Configuration (`config.yaml`)

```yaml
scenario:
  market_type: "bull"
  assets: ["AAPL", "TSLA", "BTC", "GOLD", "ETH"]
  max_steps: 100

strategy_sandbox:
  default_rules: |
    IF sentiment > 0.7 AND market_trend > 0.01:
      BUY 10% AAPL
    IF drawdown > 0.05:
      SELL 50% BTC
  hybrid_mode: true
```

---

## Execution

### Launch the Intelligence Terminal
```bash
python3 gradio_app.py
```

### Run Batch Simulations
```bash
python3 runner.py --config config.yaml
```

---

**FinomIQ** — *The future of autonomous financial strategy intelligence.*
