"""
FinomIQ — Financial Knowledge Graph (FKG).
Interactive network diagrams for asset correlations and market dependencies.
"""

import plotly.graph_objects as go
import numpy as np
from collections import defaultdict
from typing import List, Dict

# ── FinTech Dark Palette ──────────────────────────────────────────────────────

_BG = "rgba(10,10,10,1)"
_PAPER = "rgba(18,18,18,1)"
_GRID = "#333333"
_TEXT = "#E0E0E0"
_NEON_GREEN = "#00FF41"  # Bullish
_NEON_RED = "#FF3131"    # Bearish
_NEON_BLUE = "#00D4FF"   # Analytics
_PURPLE = "#9D00FF"      # Strategy

def build_financial_knowledge_graph(market_state: Dict) -> go.Figure:
    """Build an interactive FKG showing asset correlations and sector linkages."""
    assets = list(market_state.get("asset_prices", {}).keys())
    sectors = ["Tech", "Finance", "Crypto"]
    
    nodes = assets + sectors
    n = len(nodes)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    radius = 1.2
    
    x_pos = {node: float(radius * np.cos(angles[i])) for i, node in enumerate(nodes)}
    y_pos = {node: float(radius * np.sin(angles[i])) for i, node in enumerate(nodes)}

    # Edges: Sector Linkages & Correlations
    edge_traces = []
    
    # Asset to Sector Edges
    for asset in assets:
        sector = "Tech" if asset in ["AAPL", "TSLA"] else "Crypto" if asset == "BTC" else "Finance"
        edge_traces.append(go.Scatter(
            x=[x_pos[asset], x_pos[sector], None],
            y=[y_pos[asset], y_pos[sector], None],
            mode="lines",
            line=dict(width=1, color="rgba(0, 212, 255, 0.3)"),
            hoverinfo="none",
            showlegend=False
        ))

    # Asset to Asset Correlations (Simulated)
    for i, a1 in enumerate(assets):
        for a2 in assets[i+1:]:
            correlation = 0.5 + np.random.random() * 0.4
            edge_traces.append(go.Scatter(
                x=[x_pos[a1], x_pos[a2], None],
                y=[y_pos[a1], y_pos[a2], None],
                mode="lines",
                line=dict(width=correlation * 3, color=f"rgba(157, 0, 255, {correlation*0.5})"),
                hoverinfo="text",
                text=f"{a1} ↔ {a2} Corr: {correlation:.2f}",
                showlegend=False
            ))

    # Nodes
    node_colors = [_NEON_BLUE if n in assets else _PURPLE for n in nodes]
    node_sizes = [25 if n in sectors else 15 for n in nodes]
    
    node_trace = go.Scatter(
        x=[x_pos[n] for n in nodes],
        y=[y_pos[n] for n in nodes],
        mode="markers+text",
        marker=dict(size=node_sizes, color=node_colors, 
                    line=dict(width=2, color=_GRID), opacity=0.9),
        text=nodes,
        textposition="top center",
        textfont=dict(size=12, color=_TEXT, family="JetBrains Mono, monospace"),
        hoverinfo="text",
        hovertext=[f"<b>{n}</b><br>{'Sector' if n in sectors else 'Asset'}" for n in nodes],
        showlegend=False
    )

    fig = go.Figure(data=edge_traces + [node_trace])
    fig.update_layout(
        paper_bgcolor=_BG, plot_bgcolor=_BG,
        font=dict(family="Inter, sans-serif", color=_TEXT, size=12),
        title=dict(text="Financial Knowledge Graph (FKG)", 
                   font=dict(size=18, color=_NEON_BLUE, weight=700)),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        margin=dict(l=20, r=20, t=60, b=20),
        height=520, showlegend=False,
        dragmode="pan"
    )
    return fig

def build_equity_curve(history: Dict) -> go.Figure:
    """Visualize portfolio equity curve and drawdown."""
    pv = history.get("portfolio_value", [])
    steps = list(range(len(pv)))
    
    fig = go.Figure()
    
    # Equity Curve
    fig.add_trace(go.Scatter(
        x=steps, y=pv, mode="lines",
        line=dict(color=_NEON_GREEN, width=3, shape="spline"),
        fill="tozeroy", fillcolor="rgba(0, 255, 65, 0.1)",
        name="Equity",
        hoverinfo="x+y"
    ))
    
    fig.update_layout(
        paper_bgcolor=_BG, plot_bgcolor=_BG,
        font=dict(family="Inter, sans-serif", color=_TEXT, size=12),
        title=dict(text="Portfolio Equity Curve", 
                   font=dict(size=18, color=_NEON_GREEN, weight=700)),
        xaxis=dict(title="Time Steps", gridcolor=_GRID, zeroline=False),
        yaxis=dict(title="Portfolio Value ($)", gridcolor=_GRID, zeroline=False),
        margin=dict(l=60, r=30, t=60, b=60),
        height=400, showlegend=False
    )
    return fig
