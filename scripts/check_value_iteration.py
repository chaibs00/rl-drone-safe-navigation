"""
    A quick script to check that value iteration is working as expected on the saferoute environment.

    # Run in root 
        python -m scripts.check_value_iteration
"""


import matplotlib.pyplot as plt
import yaml

from saferoute.environement.grid_map import Cell
from saferoute.environement.saferoute_env import SafeRouteEnv
from saferoute.planning.value_iteration import follow_policy, value_iteration
from saferoute.utils.plotting import draw_map, draw_values

config = yaml.safe_load(open("./config.yaml"))
gamma = config["training"]["gamma"]
theta = config["value_iteration"]["theta"]

fig, axes = plt.subplots(2, 2, figsize=(16, 16))

for col, risk_costs in enumerate([False, True]):
    env = SafeRouteEnv.from_config(config, risk_costs=risk_costs)
    V, policy, n_iter = value_iteration(env, gamma=gamma, theta=theta)
    route = follow_policy(env, policy)

    risky = sum(env.map.cell(p) in (Cell.LOW_RISK, Cell.MEDIUM_RISK) for p in route)
    name = "Risk-aware reward" if risk_costs else "Base reward"
    print(f"{name}: converged in {n_iter} iterations | "
          f"route = {len(route) - 1} steps | risk cells crossed = {risky} | "
          f"V(start) = {V[env.map.to_state(env.map.start)]:.2f}")

    draw_values(env.map, V, policy, ax=axes[0, col], title=f"{name}: V(s) and policy")
    draw_map(env.map, ax=axes[1, col], route=route, show_legend=False,
             title=f"{name}: optimal route ({len(route) - 1} steps, {risky} risk cells)")

plt.savefig("results/value_iteration_check.png", bbox_inches="tight", dpi=80)
plt.show()