"""
    Tests for the risk-aware agent.

    # Run all test 
        python -m pytest tests -v
    
       # Run individual test
            python -m pytest tests/test_risk_aware.py -v


"""


from pathlib import Path

import yaml

from saferoute.agents.factory import build_agent
from saferoute.agents.q_learning import QLearningAgent
from saferoute.agents.sarsa import SarsaAgent
from saferoute.environement.grid_map import Cell
from saferoute.planning.value_iteration import follow_policy

ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load(open("config.yaml"))
CONFIG["env"]["map_path"] = str(ROOT / CONFIG["env"]["map_path"])   # works from any folder


def route_cost(env, route):
    """Risk cost of a route: 2 per low-risk cell, 5 per medium-risk cell."""
    cells = [env.map.cell(p) for p in route]
    return 2 * cells.count(Cell.LOW_RISK) + 5 * cells.count(Cell.MEDIUM_RISK)


def test_factory_builds_the_right_combinations():
    expected = {
        "q_learning": (QLearningAgent, False),
        "sarsa": (SarsaAgent, False),
        "risk_aware": (QLearningAgent, True),
    }
    for name, (agent_class, risk_costs) in expected.items():
        env, agent = build_agent(name, CONFIG, seed=0)
        assert type(agent) is agent_class
        assert env.risk_costs == risk_costs


def test_risk_aware_agent_takes_less_risk():
    results = {}
    for name in ("q_learning", "risk_aware"):
        env, agent = build_agent(name, CONFIG, seed=0)
        agent.train(env, CONFIG["training"]["episodes"])
        route = follow_policy(env, agent.policy())
        assert route[-1] == env.map.goal
        results[name] = (route, env)

    risk_route, env = results["risk_aware"]
    base_route, _ = results["q_learning"]

    medium = [env.map.cell(p) for p in risk_route].count(Cell.MEDIUM_RISK)
    assert medium == 2                                          # the unavoidable minimum
    assert route_cost(env, risk_route) < route_cost(env, base_route)