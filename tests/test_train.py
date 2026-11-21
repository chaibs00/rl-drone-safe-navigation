"""
   Units tests for the shield (agent 4).

   # Run all test 
        python -m pytest tests -v

   # Run individual test
        python -m pytest tests/test_train.py -v

"""

import copy
from pathlib import Path

import numpy as np
import yaml

from scripts.train import train_all
from saferoute.utils.io import load_run

ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load(open("config.yaml"))
CONFIG["env"]["map_path"] = str(ROOT / CONFIG["env"]["map_path"])


def small_config():
    """Same config, but tiny, so the test runs in a second."""
    config = copy.deepcopy(CONFIG)
    config["training"]["episodes"] = 50
    config["training"]["seeds"] = [0, 1]
    return config


def test_every_run_is_saved(tmp_path):
    config = small_config()
    train_all(config, results_dir=tmp_path)

    for name in config["agents"]:
        for seed in config["training"]["seeds"]:
            history, q_table = load_run(tmp_path, name, seed)
            assert len(history["reward"]) == 50
            assert q_table.shape == (256, 4)


def test_same_seed_gives_same_results(tmp_path):
    config = small_config()
    train_all(config, results_dir=tmp_path / "run_a")
    train_all(config, results_dir=tmp_path / "run_b")

    for name in config["agents"]:
        _, q_a = load_run(tmp_path / "run_a", name, 0)
        _, q_b = load_run(tmp_path / "run_b", name, 0)
        assert np.array_equal(q_a, q_b)


def test_different_seeds_give_different_results(tmp_path):
    config = small_config()
    train_all(config, results_dir=tmp_path)
    _, q_seed0 = load_run(tmp_path, "q_learning", 0)
    _, q_seed1 = load_run(tmp_path, "q_learning", 1)
    assert not np.array_equal(q_seed0, q_seed1)