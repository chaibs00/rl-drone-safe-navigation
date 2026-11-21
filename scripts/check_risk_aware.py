"""
    A quick script to check that the risk-aware agent is working as expected on the saferoute environment.

    # Run in root 
        python -m scripts.check_risk_aware
"""



import matplotlib.pyplot as plt
import numpy as np
import yaml

from saferoute.agents.factory import build_agent
from saferoute.environement.grid_map import Cell
from saferoute.environement.saferoute_env import SafeRouteEnv
from saferoute.planning.value_iteration import follow_policy, value_iteration
from saferoute.utils.plotting import draw_map

config = yaml.safe_load(open("./config.yaml"))
t = config["training"]


def describe(env, route):
    cells = [env.map.cell(p) for p in route]
    low, medium = cells.count(Cell.LOW_RISK), cells.count(Cell.MEDIUM_RISK)
    return low, medium, 2 * low + 5 * medium


routes = {}
for name in ("q_learning", "risk_aware"):
    print(f"--- {name} ---")
    costs, crashes = [], []
    for seed in t["seeds"]:
        env, agent = build_agent(name, config, seed=seed)
        history = agent.train(env, t["episodes"])
        route = follow_policy(env, agent.policy())
        low, medium, cost = describe(env, route)
        costs.append(cost)
        crashes.append(sum(h["crash"] for h in history))
        print(f"seed {seed}: route = {len(route) - 1} steps | low = {low} | medium = {medium} | "
              f"risk cost = {cost:2d} | training crashes = {crashes[-1]}")
        if seed == 0:
            routes[name] = route
    print(f"mean risk cost = {np.mean(costs):.1f} | mean training crashes = {np.mean(crashes):.0f}\n")

# Optimal reference with risk costs
env = SafeRouteEnv.from_config(config, risk_costs=True)
_, policy, _ = value_iteration(env, gamma=t["gamma"], theta=config["value_iteration"]["theta"])
routes["optimal (value iteration)"] = follow_policy(env, policy)
low, medium, cost = describe(env, routes["optimal (value iteration)"])
print(f"optimal: low = {low} | medium = {medium} | risk cost = {cost}")

fig, axes = plt.subplots(1, 3, figsize=(24, 8))
for ax, (name, route) in zip(axes, routes.items()):
    low, medium, cost = describe(env, route)
    draw_map(env.map, route=route, ax=ax, show_legend=False,
             title=f"{name} (seed 0)\n{low} low + {medium} medium = risk cost {cost}"
             if "optimal" not in name else f"{name}\n{low} low + {medium} medium = risk cost {cost}")

plt.tight_layout()
plt.savefig("results/risk_aware_check.png", bbox_inches="tight", dpi=80)
plt.show()