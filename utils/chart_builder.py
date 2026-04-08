"""
FinomIQ — FinTech Chart Builders.
Modern dark-themed analytics for hedge fund performance.
"""

import plotly.graph_objects as go
import numpy as np
import pandas as pd
from typing import List, Dict, Any

# ── FinTech Dark Palette ──────────────────────────────────────────────────────

_BG = "rgba(10,10,10,1)"
_PAPER = "rgba(18,18,18,1)"
_GRID = "#333333"
_TEXT = "#E0E0E0"
_NEON_GREEN = "#00FF41"  # Bullish
_NEON_RED = "#FF3131"    # Bearish
_NEON_BLUE = "#00D4FF"   # Analytics
_PURPLE = "#9D00FF"      # Strategy

_LAYOUT = dict(
    plot_bgcolor=_BG,
    paper_bgcolor=_BG,
    font=dict(family="Inter, sans-serif", color=_TEXT, size=12),
    margin=dict(l=50, r=30, t=60, b=50),
    xaxis=dict(gridcolor=_GRID, zerolinecolor=_GRID),
    yaxis=dict(gridcolor=_GRID, zerolinecolor=_GRID),
)

def _base_layout(**overrides) -> dict:
    layout = dict(_LAYOUT)
    if "height" not in overrides:
        layout["height"] = 400
    layout.update(overrides)
    return layout

def build_candlestick_chart(asset_name: str, history: List[Dict]) -> go.Figure:
    """Real-time candlestick chart for a specific asset."""
    # Simulated OHLC data from price history
    prices = [h["asset_prices"].get(asset_name, 100) for h in history]
    if not prices:
        return go.Figure().update_layout(**_base_layout(title="Waiting for market data..."))

    df = pd.DataFrame({"close": prices})
    df["open"] = df["close"].shift(1).fillna(df["close"])
    df["high"] = df[["open", "close"]].max(axis=1) * (1 + np.random.random(len(df)) * 0.005)
    df["low"] = df[["open", "close"]].min(axis=1) * (1 - np.random.random(len(df)) * 0.005)
    
    fig = go.Figure(data=[go.Candlestick(
        x=list(range(len(df))),
        open=df["open"], high=df["high"],
        low=df["low"], close=df["close"],
        increasing_line_color=_NEON_GREEN,
        decreasing_line_color=_NEON_RED,
        name=asset_name
    )])
    
    fig.update_layout(**_base_layout( 
        title=dict(text=f"{asset_name} Price Action", font=dict(size=18, color=_NEON_BLUE, weight=700)),
        xaxis_rangeslider_visible=False,
        xaxis_title="Time Steps", yaxis_title="Price ($)"
    ))
    return fig

def build_pnl_chart(history: List[Dict]) -> go.Figure:
    """Portfolio PnL curve with neon accents."""
    pnl = [h["unrealized_pnl"] for h in history]
    steps = list(range(len(pnl)))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=steps, y=pnl, mode="lines",
        line=dict(color=_NEON_GREEN if (pnl[-1] if pnl else 0) >= 0 else _NEON_RED, width=3),
        fill="tozeroy",
        fillcolor="rgba(0, 255, 65, 0.1)" if (pnl[-1] if pnl else 0) >= 0 else "rgba(255, 49, 49, 0.1)",
        name="PnL"
    ))
    
    fig.update_layout(**_base_layout(
        title=dict(text="Portfolio PnL Trajectory", font=dict(size=18, weight=700)),
        xaxis_title="Steps", yaxis_title="PnL ($)"
    ))
    return fig

def build_allocation_pie(positions: Dict[str, float], prices: Dict[str, float]) -> go.Figure:
    """Portfolio allocation pie chart."""
    labels = []
    values = []
    for asset, units in positions.items():
        if units > 0:
            labels.append(asset)
            values.append(units * prices.get(asset, 0))
    
    if not values:
        labels = ["Cash"]
        values = [1.0]

    fig = go.Figure(data=[go.Pie(
        labels=labels, values=values,
        hole=0.6,
        marker=dict(colors=[_NEON_BLUE, _PURPLE, _NEON_GREEN, _NEON_RED], line=dict(color=_BG, width=2)),
        textinfo="label+percent"
    )])
    
    fig.update_layout(**_base_layout(
        title=dict(text="Asset Allocation", font=dict(size=18, weight=700)),
        showlegend=False
    ))
    return fig

def build_risk_gauge(risk_score: float) -> go.Figure:
    """Risk exposure gauge (0 to 1)."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Risk Exposure (%)", 'font': {'size': 16, 'color': _TEXT}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': _TEXT},
            'bar': {'color': _NEON_BLUE},
            'bgcolor': _PAPER,
            'borderwidth': 2,
            'bordercolor': _GRID,
            'steps': [
                {'range': [0, 30], 'color': 'rgba(0, 255, 65, 0.2)'},
                {'range': [30, 70], 'color': 'rgba(0, 212, 255, 0.2)'},
                {'range': [70, 100], 'color': 'rgba(255, 49, 49, 0.2)'}
            ],
            'threshold': {
                'line': {'color': _NEON_RED, 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    # Use _base_layout to ensure no duplicate height argument
    fig.update_layout(**_base_layout(height=250))
    return fig

def build_correlation_matrix(assets: List[str]) -> go.Figure:
    """Heatmap showing asset correlations."""
    n = len(assets)
    z = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j: z[i,j] = 1.0
            else: z[i,j] = 0.4 + np.random.random() * 0.5
            
    fig = go.Figure(data=go.Heatmap(
        z=z, x=assets, y=assets,
        colorscale=[[0, _BG], [0.5, _PURPLE], [1.0, _NEON_BLUE]],
        zmin=0, zmax=1
    ))
    
    fig.update_layout(**_base_layout(
        title=dict(text="Correlation Matrix", font=dict(size=18, weight=700)),
        height=400
    ))
    return fig

def build_drawdown_chart(history: List[Dict]) -> go.Figure:
    """Visualize peak-to-trough drawdown."""
    dds = [h.get("risk_exposure_score", 0) for h in history]
    steps = list(range(len(dds)))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=steps, y=dds, mode="lines",
        line=dict(color=_NEON_RED, width=2),
        fill="tozeroy",
        fillcolor="rgba(255, 49, 49, 0.1)",
        name="Drawdown"
    ))
    
    fig.update_layout(**_base_layout(
        title=dict(text="Peak-to-Trough Drawdown", font=dict(size=18, weight=700)),
        xaxis_title="Steps", yaxis_title="Drawdown (%)"
    ))
    return fig

def build_risk_return_scatter(results: List[Dict]) -> go.Figure:
    """Scatter plot of risk (std) vs return across episodes."""
    rets = [r.get("unrealized_pnl", 0) for r in results]
    risks = [np.std(r.get("history", {}).get("returns", [0])) for r in results]
    
    fig = go.Figure(data=go.Scatter(
        x=risks, y=rets, mode="markers+text",
        marker=dict(size=12, color=_NEON_BLUE, opacity=0.8),
        text=[f"Ep {i+1}" for i in range(len(rets))],
        textposition="top center"
    ))
    
    fig.update_layout(**_base_layout(
        title=dict(text="Risk-Return Efficiency Profile", font=dict(size=18, weight=700)),
        xaxis_title="Volatility (Risk)", yaxis_title="Returns ($)"
    ))
    return fig

def build_sector_heatmap() -> go.Figure:
    """Simulated Sector Performance Heatmap."""
    sectors = ["Tech", "Finance", "Crypto", "Energy", "Healthcare"]
    performance = np.random.uniform(-0.05, 0.05, size=len(sectors))
    
    fig = go.Figure(data=go.Heatmap(
        z=[performance], x=sectors, y=["Performance"],
        colorscale=[[0, _NEON_RED], [0.5, _PAPER], [1.0, _NEON_GREEN]],
        zmin=-0.05, zmax=0.05
    ))
    
    fig.update_layout(**_base_layout(
        title=dict(text="Sector Sentiment Heatmap", font=dict(size=18, weight=700)),
        height=200
    ))
    return fig
