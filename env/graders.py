"""
FinomIQ — Task Graders.
Programmatic evaluation of hedge fund agent performance for each task.
"""

from typing import Dict, List
import numpy as np

# Scoring bounds: (0, 1) exclusive
EPSILON = 1e-6
LOWER_BOUND = EPSILON
UPPER_BOUND = 1.0 - EPSILON

def grade_bull_market_growth(history: Dict) -> float:
    """Grader for Easy Task: Maximize profit in a bull market."""
    pv = history.get("portfolio_value", [])
    if not pv: return LOWER_BOUND
    initial_pv = pv[0]
    final_pv = pv[-1]
    
    total_return = (final_pv / initial_pv) - 1
    # Normalized score: 0.0 at 0% return, 1.0 at 10% return
    score = np.clip(total_return / 0.1, LOWER_BOUND, UPPER_BOUND)
    return float(score)

def grade_volatile_market_navigation(history: Dict) -> float:
    """Grader for Medium Task: High Sharpe ratio in a volatile market."""
    rets = history.get("returns", [])
    if not rets: return LOWER_BOUND
    
    mean_ret = np.mean(rets)
    std_ret = np.std(rets)
    
    if std_ret == 0:
        return UPPER_BOUND if mean_ret > 0 else LOWER_BOUND
    sharpe = mean_ret / std_ret
    
    # Normalized score: 0.0 at sharpe <= 0, 1.0 at sharpe >= 1.0
    score = np.clip(sharpe, LOWER_BOUND, UPPER_BOUND)
    return float(score)

def grade_market_crash_survival(history: Dict) -> float:
    """Grader for Hard Task: Minimize drawdown in a market crash."""
    dds = history.get("drawdowns", [])
    if not dds: return LOWER_BOUND
    
    max_dd = max(dds)
    # Normalized score: 1.0 at 0% drawdown, 0.0 at 30% drawdown
    score = np.clip(1.0 - (max_dd / 0.3), LOWER_BOUND, UPPER_BOUND)
    return float(score)

def get_grader(task_id: str):
    graders = {
        "bull_market_growth": grade_bull_market_growth,
        "volatile_market_navigation": grade_volatile_market_navigation,
        "market_crash_survival": grade_market_crash_survival
    }
    return graders.get(task_id, grade_bull_market_growth)
