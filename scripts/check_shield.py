"""
    A quick script to check that the shielded agent is working as expected on the saferoute environment.

    # Run in root 
        python -m scripts.check_shield
"""

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.patches import Rectangle

from saferoute.agents.factory import build_agent
from saferoute.environement.grid_map import Cell
from saferoute.planning.value_iteration import follow_policy
from saferoute.utils.plotting import draw_map

config = yaml.safe_load(open("./config.yaml"))
t = config["training"]
COLORS = {"q_learning": "#1F6FEB", "sarsa": "#E36209", "risk_aware": "#8250DF", "shielded": "#1A7F37"}


def risk_cost(env, route):
    cells = [env.map.cell(p) for p in route]
    return 2 * cells.count(Cell.LOW_RISK) + 5 * cells.count(Cell.MEDIUM_RISK)


results = {}
for name in config["agents"]:
    crash_curves, reward_curves, costs = [], [], []
    for seed in t["seeds"]:
        env, agent = build_agent(name, config, seed=seed)
        history = agent.train(env, t["episodes"])
        crash_curves.append(np.cumsum([h["crash"] for h in history]))
        reward_curves.append([h["reward"] for h in history])
        route = follow_policy(env, agent.policy())
        costs.append(risk_cost(env, route))
        if seed == 0:
            first_env, first_agent, first_route = env, agent, route
    results[name] = (np.array(crash_curves), np.array(reward_curves))
    print(f"{name:11s} training crashes per seed = {[int(c[-1]) for c in crash_curves]} | "
          f"route risk cost per seed = {costs}")

window = 50
fig, axes = plt.subplots(2, 2, figsize=(16, 14))

for name, (crashes, rewards) in results.items():
    axes[0, 0].plot(crashes.mean(axis=0), color=COLORS[name], label=name, linewidth=2)
    smooth = np.convolve(rewards.mean(axis=0), np.ones(window) / window, mode="valid")
    axes[0, 1].plot(smooth, color=COLORS[name], label=name)

axes[0, 0].set_title("Cumulative crashes during training (mean of 5 seeds)")
axes[0, 0].set_xlabel("Episode")
axes[0, 0].set_ylabel("Crashes")
axes[0, 0].legend()
axes[0, 1].set_title(f"Episode reward (mean of 5 seeds, moving average {window})")
axes[0, 1].set_xlabel("Episode")
axes[0, 1].set_ylabel("Reward")
axes[0, 1].legend()

# Where the shield acts: cells with at least one blocked action
shield = first_agent.shield
draw_map(first_env.map, ax=axes[1, 0], show_legend=False,
         title="Where the shield acts (cells with at least one blocked move)")
for state in range(first_env.map.n_states):
    col, row = first_env.map.to_pos(state)
    if first_env.map.cell((col, row)) in (Cell.BUILDING, Cell.HIGH_RISK):
        continue
    if len(shield.allowed(state)) < 4:
        axes[1, 0].add_patch(Rectangle((col, row), 1, 1, fill=False,
                                       edgecolor="#1A7F37", linewidth=3, zorder=5))

draw_map(first_env.map, route=first_route, ax=axes[1, 1], show_legend=False,
         title=f"Shielded agent (seed 0): learned route, risk cost {risk_cost(first_env, first_route)}")

plt.tight_layout()
plt.savefig("results/shield_check.png", bbox_inches="tight", dpi=80)
plt.show()