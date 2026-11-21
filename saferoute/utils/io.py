"""Save and load training runs.

One run = one agent trained with one seed. Its files go in:

    results/<agent>/seed_<n>/
        history.npz    one value per episode: reward, steps, success, crash, risky_steps
        q_table.npy    the learned Q-table (256 x 4)
"""

from pathlib import Path

import numpy as np

HISTORY_KEYS = ("reward", "steps", "success", "crash", "risky_steps")


def run_dir(results_dir, agent_name, seed):
    return Path(results_dir) / agent_name / f"seed_{seed}"


def save_run(results_dir, agent_name, seed, history, q_table):
    """Save the episode history (list of dicts) and the Q-table of one run."""
    folder = run_dir(results_dir, agent_name, seed)
    folder.mkdir(parents=True, exist_ok=True)

    arrays = {key: np.array([episode[key] for episode in history]) for key in HISTORY_KEYS}
    np.savez(folder / "history.npz", **arrays)
    np.save(folder / "q_table.npy", q_table)
    return folder


def load_run(results_dir, agent_name, seed):
    """Return (history, q_table). history is a dict of arrays, one value per episode."""
    folder = run_dir(results_dir, agent_name, seed)
    with np.load(folder / "history.npz") as data:
        history = {key: data[key] for key in HISTORY_KEYS}
    q_table = np.load(folder / "q_table.npy")
    return history, q_table