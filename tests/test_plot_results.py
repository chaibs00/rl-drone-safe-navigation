"""
   Units tests for testing the plotting of results.

   # Run all test 
        python -m pytest tests -v

   # Run individual test
        python -m pytest tests/test_plot_results.py -v

"""

import copy
from pathlib import Path

import matplotlib

matplotlib.use("Agg")                     # draw to files, never open a window

import yaml

from scripts.evaluate import evaluate_all, optimal_reference
from scripts.plot_results_final import (
    figure_learning_curves, figure_routes, figure_training_crashes, figure_values, load_histories,
)
from scripts.train import train_all

ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load(open("config.yaml"))
CONFIG["env"]["map_path"] = str(ROOT / CONFIG["env"]["map_path"])


def test_all_figures_are_created(tmp_path):
    config = copy.deepcopy(CONFIG)
    config["training"]["episodes"] = 200          # small but longer than the 100-episode window
    config["training"]["seeds"] = [0, 1]
    results_dir = tmp_path / "results"
    train_all(config, results_dir=results_dir)

    histories = load_histories(config, results_dir)
    rows = evaluate_all(config, results_dir)
    optimal_env, optimal_route, _ = optimal_reference(config)

    figure_training_crashes(histories, tmp_path / "1.png")
    figure_learning_curves(histories, tmp_path / "2.png")
    figure_routes(rows, optimal_env, optimal_route, tmp_path / "3.png")
    figure_values(config, results_dir, tmp_path / "4.png")

    for name in ("1.png", "2.png", "3.png", "4.png"):
        assert (tmp_path / name).stat().st_size > 0