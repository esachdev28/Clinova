"""
FinomIQ — FastAPI server.
Exposes the Autonomous Hedge Fund Intelligence Platform over HTTP.
"""

from contextlib import asynccontextmanager
from typing import AsyncIterator, Dict, Optional, List
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from env.environment import FinomIQEnv
from env.models import Action

# ── Environment instances ─────────────────────────────────────────────────────

env: Optional[FinomIQEnv] = None

@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Initialize the FinomIQ environment on startup."""
    global env
    env = FinomIQEnv(config_path="config.yaml")
    yield
    env = None

app = FastAPI(
    title="FinomIQ",
    description="Autonomous Hedge Fund Intelligence API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request / response schemas ────────────────────────────────────────────────

class ResetRequest(BaseModel):
    market_type: Optional[str] = None
    assets: Optional[List[str]] = None
    seed: Optional[int] = None

class StepRequest(BaseModel):
    action_type: int
    asset_name: Optional[str] = None
    amount: Optional[float] = None

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/reset")
async def reset_env(body: ResetRequest | None = None):
    """Reset the market simulation."""
    if body:
        if body.market_type:
            env.config["scenario"]["market_type"] = body.market_type
            env.market_type = body.market_type
        if body.assets:
            env.config["scenario"]["assets"] = body.assets
            env.assets = body.assets
        if body.seed:
            env.config["scenario"]["seed"] = body.seed
            env.base_seed = body.seed
            
    result = await env.reset()
    return result

@app.post("/step")
async def step_env(body: StepRequest):
    """Take one financial action in the market."""
    if body.action_type < 0 or body.action_type > 6:
        raise HTTPException(
            status_code=422,
            detail=f"action_type must be 0–6, got {body.action_type}",
        )
    action = Action(
        action_type=body.action_type,
        asset_name=body.asset_name,
        amount=body.amount
    )
    result = await env.step(action)
    return result

@app.get("/state")
async def get_state():
    """Return the full internal state of the portfolio and market."""
    return await env.state()

@app.get("/health")
async def health_check():
    """Health-check endpoint."""
    return {"status": "ok", "engine": "FinomIQ-v1"}

def main() -> None:
    """Run the server."""
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)

if __name__ == "__main__":
    main()
