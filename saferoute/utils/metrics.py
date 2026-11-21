"""Evaluation metrics.

Two kinds of metrics, matching the two kinds of safety in the README:

    Training metrics    : computed from the episode history (what happened while learning)
    Deployment metrics  : computed by flying the final greedy policy once (epsilon = 0)

The environment is deterministic, so one flight of the final policy is
enough: it would follow exactly the same route every time.
"""

import numpy as np

from saferoute.environement.grid_map import Cell
from saferoute.planning.value_iteration import follow_policy


# ----------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------

def risk_counts(grid_map, route):
    """Number of low-risk and medium-risk cells entered along a route."""
    cells = [grid_map.cell(pos) for pos in route[1:]]       # route[0] is the start
    return cells.count(Cell.LOW_RISK), cells.count(Cell.MEDIUM_RISK)


def route_risk_cost(grid_map, route, rewards):
    """Risk cost of a route: |low_risk| per low cell + |medium_risk| per medium cell."""
    low, medium = risk_counts(grid_map, route)
    return low * abs(rewards["low_risk"]) + medium * abs(rewards["medium_risk"])


def route_return(reference_env, route):
    """Total (undiscounted) reward of a route, measured with a reference reward.

    All agents are scored with the SAME reward (the risk-aware one), so their
    returns can be compared even if they were trained with different rewards.
    """
    return sum(reference_env.reward(pos) for pos in route[1:])


# ----------------------------------------------------------------------
# Training metrics (from the saved history)
# ----------------------------------------------------------------------

def training_metrics(history):
    success = history["success"]
    return {
        "training_crashes": int(history["crash"].sum()),
        "first_success": int(np.argmax(success)) if success.any() else -1,
        "training_success_rate": float(success.mean()),
    }


# ----------------------------------------------------------------------
# Deployment metrics (fly the final greedy policy once)
# ----------------------------------------------------------------------

def deployment_metrics(env, agent, reference_env, optimal_return):
    route = follow_policy(env, agent.policy())
    final = route[-1]
    low, medium = risk_counts(env.map, route)
    ret = route_return(reference_env, route)

    return {
        "success": final == env.map.goal,
        "deployment_crash": env.map.is_high_risk(final),
        "path_length": len(route) - 1,
        "low_cells": low,
        "medium_cells": medium,
        "risk_cost": route_risk_cost(env.map, route, env.rewards),
        "return": ret,
        "gap_to_optimal": optimal_return - ret,
        "route": route,
    }