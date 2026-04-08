"""
FinomIQ — Task Graders.
Programmatic evaluation of hedge fund agent performance for each task.
"""

from typing import Dict, List
import numpy as np

def grade_bull_market_growth(history: Dict) -> float:
    """Grader for Easy Task: Maximize profit in a bull market."""
    pv = history.get("portfolio_value", [])
    if not pv: return 0.0
    initial_pv = pv[0]
    final_pv = pv[-1]
    
    total_return = (final_pv / initial_pv) - 1
    # Normalized score: 0.0 at 0% return, 1.0 at 10% return
    score = np.clip(total_return / 0.1, 0.0, 1.0)
    return float(score)

def grade_volatile_market_navigation(history: Dict) -> float:
    """Grader for Medium Task: High Sharpe ratio in a volatile market."""
    rets = history.get("returns", [])
    if not rets: return 0.0
    
    mean_ret = np.mean(rets)
    std_ret = np.std(rets)
    
    if std_ret == 0: return 0.0
    sharpe = mean_ret / std_ret
    
    # Normalized score: 0.0 at sharpe <= 0, 1.0 at sharpe >= 1.0
    score = np.clip(sharpe, 0.0, 1.0)
    return float(score)

def grade_market_crash_survival(history: Dict) -> float:
    """Grader for Hard Task: Minimize drawdown in a market crash."""
    dds = history.get("drawdowns", [])
    if not dds: return 0.0
    
    max_dd = max(dds)
    # Normalized score: 1.0 at 0% drawdown, 0.0 at 30% drawdown
    score = np.clip(1.0 - (max_dd / 0.3), 0.0, 1.0)
    return float(score)

def get_grader(task_id: str):
    graders = {
        "bull_market_growth": grade_bull_market_growth,
        "volatile_market_navigation": grade_volatile_market_navigation,
        "market_crash_survival": grade_market_crash_survival
    }
    return graders.get(task_id, grade_bull_market_growth)
