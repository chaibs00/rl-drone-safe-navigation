"""
   Units tests for value iteration planning algorithm.

   # Run all test 
        python -m pytest tests -v
   
      # Run individual test
           python -m pytest tests/test_value_iteration.py -v 

"""


from pathlib import Path

from saferoute.environement.grid_map import GridMap, Cell
from saferoute.environement.saferoute_env import SafeRouteEnv
from saferoute.planning.value_iteration import follow_policy, value_iteration

ROOT = Path(__file__).resolve().parents[1]
MAP_PATH = ROOT / "maps" / "saferoute_16x16.txt"
REWARDS = {"goal": 100, "step": -1, "crash": -100, "medium_risk": -5, "low_risk": -2}


def solve(risk_costs):
    env = SafeRouteEnv(GridMap(MAP_PATH), REWARDS, risk_costs=risk_costs)
    V, policy, _ = value_iteration(env, gamma=0.99, theta=1e-6)
    return env, V, follow_policy(env, policy)


def risk_cells(env, route):
    return sum(env.map.cell(p) in (Cell.LOW_RISK, Cell.MEDIUM_RISK) for p in route)


def test_optimal_route_is_shortest_and_safe():
    for risk_costs in (False, True):
        env, _, route = solve(risk_costs)
        assert route[-1] == env.map.goal                      # reaches the goal
        assert len(route) - 1 == 26                           # shortest possible
        assert not any(env.map.is_high_risk(p) for p in route)  # never crashes


def test_risk_costs_reduce_risk_exposure():
    env, _, base_route = solve(risk_costs=False)
    _, _, risk_route = solve(risk_costs=True)
    assert risk_cells(env, risk_route) == 2                   # minimum possible
    assert risk_cells(env, risk_route) < risk_cells(env, base_route)


def test_value_of_cell_next_to_goal():
    env, V, _ = solve(risk_costs=False)
    assert V[env.map.to_state((13, 14))] == 100               # one step: +100, then nothing