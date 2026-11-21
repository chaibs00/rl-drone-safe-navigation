"""Generate the result figures from the saved runs.

    python main.py plot

Reads results/ (written by `python main.py train`) and saves 4 figures in
assets/figures/, so the README can show them:

    1_training_crashes.png   cumulative crashes while learning (the headline result)
    2_learning_curves.png    success rate over training
    3_routes.png             final route of each agent next to the optimal route
    4_values_qlearning_vs_sarsa.png   why SARSA crashes less
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from saferoute.agents.factory import build_agent
from saferoute.utils.io import load_run
from saferoute.utils.plotting import (
    AGENT_STYLE, draw_map, draw_values, moving_average, plot_mean_and_band, style_axis,
)
from scripts.evaluate import evaluate_all, optimal_reference
from scripts.train import load_config

FIGURES_DIR = Path("assets/figures")
WINDOW = 100          # moving-average window for learning curves


def load_histories(config, results_dir):
    """{agent: [history of seed 0, history of seed 1, ...]}"""
    return {
        name: [load_run(results_dir, name, seed)[0] for seed in config["training"]["seeds"]]
        for name in config["agents"]
    }


def figure_training_crashes(histories, path):
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, runs in histories.items():
        plot_mean_and_band(ax, [np.cumsum(h["crash"]) for h in runs], name)
    style_axis(ax, "Cumulative crashes during training (mean of 5 seeds, min-max band)",
               "Episode", "Crashes")
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)


def figure_learning_curves(histories, path):
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, runs in histories.items():
        plot_mean_and_band(ax, [100 * moving_average(h["success"], WINDOW) for h in runs], name)
    style_axis(ax, f"Success rate during training (moving average over {WINDOW} episodes)",
               "Episode", "Success rate (%)")
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)


def figure_routes(rows, optimal_env, optimal_route, path, seed=0):
    panels = [("Optimal (value iteration)", optimal_route, optimal_env)]
    for name, (label, _) in AGENT_STYLE.items():
        row = next(r for r in rows if r["agent"] == name and r["seed"] == seed)
        panels.append((f"{label}\nrisk cost {row['risk_cost']}", row["route"], optimal_env))

    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 6.5))
    for ax, (title, route, env) in zip(axes, panels):
        draw_map(env.map, route=route, ax=ax, title=title, show_legend=False)
    fig.suptitle(f"Final routes (seed {seed}); optimal risk cost = 10", fontsize=16)
    fig.savefig(path, bbox_inches="tight", dpi=100)
    plt.close(fig)


def figure_values(config, results_dir, path, seed=0):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.5))
    for ax, name in zip(axes, ("q_learning", "sarsa")):
        env, agent = build_agent(name, config, seed=seed)
        _, agent.Q = load_run(results_dir, name, seed)
        draw_values(env.map, agent.Q.max(axis=1), agent.policy(), ax=ax,
                    title=f"{AGENT_STYLE[name][0]}: learned values max Q(s, a)")
    fig.suptitle("Cells next to the market: valuable for Q-Learning, avoided by SARSA",
                 fontsize=15)
    fig.savefig(path, bbox_inches="tight", dpi=100)
    plt.close(fig)


def main(config_path="configs/default.yaml"):
    config = load_config(config_path)
    results_dir = Path(config["paths"]["results_dir"])
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    histories = load_histories(config, results_dir)
    rows = evaluate_all(config, results_dir)
    optimal_env, optimal_route, _ = optimal_reference(config)

    figure_training_crashes(histories, FIGURES_DIR / "1_training_crashes.png")
    figure_learning_curves(histories, FIGURES_DIR / "2_learning_curves.png")
    figure_routes(rows, optimal_env, optimal_route, FIGURES_DIR / "3_routes.png")
    figure_values(config, results_dir, FIGURES_DIR / "4_values_qlearning_vs_sarsa.png")

    for path in sorted(FIGURES_DIR.glob("*.png")):
        print(f"Saved {path}")


if __name__ == "__main__":
    main()