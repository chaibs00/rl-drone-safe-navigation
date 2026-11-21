"""
   Units tests for the shield (agent 4).

   # Run all test 
        python -m pytest tests -v

   # Run individual test
        python -m pytest tests/test_metrics.py -v

"""



import copy
from pathlib import Path

import numpy as np
import yaml

from saferoute.environement.saferoute_env import SafeRouteEnv
from saferoute.utils.metrics import risk_counts, route_return, route_risk_cost, training_metrics
from scripts.evaluate import evaluate_all, optimal_reference
from scripts.train import train_all

ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load(open("config.yaml"))
CONFIG["env"]["map_path"] = str(ROOT / CONFIG["env"]["map_path"])

# Hand-made 26-step route: down column 2, right along row 11, down column 14
ROW_11_ROUTE = ([(1, 1), (2, 1)] + [(2, r) for r in range(2, 12)]
                + [(c, 11) for c in range(3, 15)] + [(14, r) for r in range(12, 15)])


def test_risk_counts_and_cost():
    env = SafeRouteEnv.from_config(CONFIG, risk_costs=True)
    assert risk_counts(env.map, ROW_11_ROUTE) == (0, 2)              # crosses (7,11) and (8,11)
    assert route_risk_cost(env.map, ROW_11_ROUTE, CONFIG["rewards"]) == 10


def test_route_return_with_risk_aware_reward():
    env = SafeRouteEnv.from_config(CONFIG, risk_costs=True)
    # 23 normal steps (-1), 2 medium steps (-6), then the goal (+100)
    assert route_return(env, ROW_11_ROUTE) == 100 - 23 - 12


def test_optimal_return_is_65():
    _, _, optimal_return = optimal_reference(CONFIG)
    assert optimal_return == 65


def test_training_metrics():
    history = {
        "crash": np.array([1, 1, 0, 0, 0]),
        "success": np.array([0, 0, 0, 1, 1]),
    }
    metrics = training_metrics(history)
    assert metrics["training_crashes"] == 2
    assert metrics["first_success"] == 3
    assert np.isclose(metrics["training_success_rate"], 0.4)


def test_evaluate_all_on_saved_runs(tmp_path):
    config = copy.deepcopy(CONFIG)
    config["training"]["episodes"] = 3000
    config["training"]["seeds"] = [0]
    train_all(config, results_dir=tmp_path)

    rows = evaluate_all(config, results_dir=tmp_path)
    by_agent = {row["agent"]: row for row in rows}

    assert set(by_agent) == set(config["agents"])
    for row in rows:
        assert row["success"] and not row["deployment_crash"]
        assert row["gap_to_optimal"] >= 0                          # nobody beats the optimum
    assert by_agent["shielded"]["training_crashes"] == 0