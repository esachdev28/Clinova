"""
FinomIQ — Strategy Sandbox Engine.
Parses and executes user-defined financial rules (DSL).
"""

import re
from typing import Dict, List, Any, Optional

class StrategyDSLEngine:
    """Engine to parse and execute custom trading rules."""

    def __init__(self):
        self.operators = {'>': lambda a, b: a > b, '<': lambda a, b: a < b, '==': lambda a, b: a == b}

    def parse_rules(self, dsl_text: str) -> List[Dict]:
        """
        Parses DSL text into executable rule objects.
        Example DSL:
        IF sentiment > 0.7 AND market_trend > 0.01:
          BUY 10% AAPL
        """
        rules = []
        lines = [line.strip() for line in dsl_text.split('\n') if line.strip()]
        
        current_rule = None
        for line in lines:
            if line.startswith("IF "):
                # Parse condition
                condition_part = line[3:].split(':')[0]
                current_rule = {"condition": condition_part, "actions": []}
                rules.append(current_rule)
            elif current_rule and (line.startswith("BUY") or line.startswith("SELL") or line.startswith("HEDGE") or line.startswith("HOLD")):
                # Parse action
                current_rule["actions"].append(line)
        
        return rules

    def evaluate(self, rules: List[Dict], observation: Dict) -> List[Dict]:
        """Evaluates rules against current market observation."""
        triggered_actions = []
        
        for rule in rules:
            if self._check_condition(rule["condition"], observation):
                for action_str in rule["actions"]:
                    action = self._parse_action(action_str)
                    if action:
                        triggered_actions.append(action)
        
        return triggered_actions

    def _check_condition(self, condition: str, obs: Dict) -> bool:
        """Simple evaluator for 'var > val AND var2 < val2'."""
        # Split by AND/OR (case insensitive)
        parts = re.split(r'\s+AND\s+', condition, flags=re.IGNORECASE)
        for part in parts:
            match = re.search(r'(\w+)\s*([><=]+)\s*([\d\.]+)', part)
            if not match: continue
            
            var, op, val = match.groups()
            val = float(val)
            
            # Map DSL vars to observation keys
            obs_val = obs.get(var, 0)
            if var == "sentiment": obs_val = obs.get("news_sentiment_score", 0)
            elif var == "drawdown": obs_val = obs.get("risk_exposure_score", 0)
            elif var == "trend": obs_val = obs.get("market_trend_score", 0)
            elif var == "vix": obs_val = obs.get("volatility_index", 0)
            elif var == "liquidity": obs_val = obs.get("liquidity_index", 0)
            
            if not self.operators.get(op, lambda a, b: False)(obs_val, val):
                return False
        return True

    def _parse_action(self, action_str: str) -> Optional[Dict]:
        """Parses 'BUY 10% AAPL' into dict."""
        match = re.search(r'(BUY|SELL|HEDGE|HOLD)\s*(\d+)%\s*(\w+)', action_str, re.IGNORECASE)
        if not match: return None
        
        act_type, amount_pct, asset = match.groups()
        type_map = {"BUY": 0, "SELL": 1, "HOLD": 2, "HEDGE": 5}
        
        return {
            "action_type": type_map.get(act_type.upper(), 2),
            "amount_pct": float(amount_pct) / 100.0,
            "asset_name": asset.upper()
        }
