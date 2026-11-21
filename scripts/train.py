"""Train the 4 agents over all seeds and save every run.

    python main.py train
    python main.py train --config configs/default.yaml

For each agent in the config and each seed, this script:
    1. builds the agent and its environment (factory)
    2. trains it for `episodes` episodes
    3. saves its episode history and Q-table in results/<agent>/seed_<n>/
A copy of the config is saved in results/ so every result can be traced
back to the exact settings that produced it.
"""

import shutil
import time
from pathlib import Path

import numpy as np
import yaml

from saferoute.agents.factory import build_agent
from saferoute.utils.io import save_run


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def train_all(config, results_dir=None):
    """Train every agent on every seed. Returns {agent_name: [history per seed]}."""
    results_dir = Path(results_dir or config["paths"]["results_dir"])
    episodes = config["training"]["episodes"]
    seeds = config["training"]["seeds"]

    all_histories = {}
    for name in config["agents"]:
        all_histories[name] = []
        for seed in seeds:
            start = time.time()

            env, agent = build_agent(name, config, seed=seed)
            history = agent.train(env, episodes)
            save_run(results_dir, name, seed, history, agent.Q)
            all_histories[name].append(history)

            crashes = sum(h["crash"] for h in history)
            success = np.mean([h["success"] for h in history[-100:]])
            print(f"  {name:11s} seed {seed} | training crashes = {crashes:4d} | "
                  f"success (last 100) = {success:4.0%} | {time.time() - start:.1f}s")

    return all_histories


def main(config_path="configs/default.yaml"):
    config = load_config(config_path)
    results_dir = Path(config["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Training {len(config['agents'])} agents x {len(config['training']['seeds'])} seeds "
          f"x {config['training']['episodes']} episodes\n")

    start = time.time()
    train_all(config, results_dir)
    shutil.copy(config_path, results_dir / "config.yaml")     # keep the exact settings

    print(f"\nDone in {time.time() - start:.0f}s. Results saved in {results_dir}/")


if __name__ == "__main__":
    main()