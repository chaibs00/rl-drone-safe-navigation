"""
    A quick script to check that value iteration is working as expected on the saferoute environment.

    # Run in root 
        python -m scripts.check_sarsa
"""


import matplotlib.pyplot as plt
import numpy as np
import yaml

from saferoute.agents.q_learning import QLearningAgent
from saferoute.agents.sarsa import SarsaAgent
from saferoute.environement.saferoute_env import SafeRouteEnv
from saferoute.planning.value_iteration import follow_policy
from saferoute.utils.plotting import draw_values

config = yaml.safe_load(open("./config.yaml"))
t = config["training"]
AGENTS = {"Q-Learning": (QLearningAgent, "#1F6FEB"), "SARSA": (SarsaAgent, "#E36209")}

results = {}
for name, (agent_class, _) in AGENTS.items():
    crash_curves, reward_curves, agents = [], [], []
    for seed in t["seeds"]:
        env = SafeRouteEnv.from_config(config, risk_costs=False)
        agent = agent_class(
            env.observation_space.n, env.action_space.n,
            alpha=t["alpha"], gamma=t["gamma"],
            epsilon_start=t["epsilon_start"], epsilon_end=t["epsilon_end"],
            epsilon_decay_episodes=t["epsilon_decay_episodes"],
            seed=seed,
        )
        history = agent.train(env, t["episodes"])
        crash_curves.append(np.cumsum([h["crash"] for h in history]))
        reward_curves.append([h["reward"] for h in history])
        agents.append(agent)

        route = follow_policy(env, agent.policy())
        print(f"{name:10s} seed {seed}: training crashes = {crash_curves[-1][-1]:4d} | "
              f"learned route = {len(route) - 1} steps")

    results[name] = (np.array(crash_curves), np.array(reward_curves), agents[0], env)
    print(f"{name:10s} mean training crashes = {np.mean([c[-1] for c in crash_curves]):.0f}\n")

window = 50
fig, axes = plt.subplots(2, 2, figsize=(16, 14))

for name, (crashes, rewards, _, _) in results.items():
    color = AGENTS[name][1]
    mean_crashes = crashes.mean(axis=0)
    axes[0, 0].plot(mean_crashes, color=color, label=name)
    axes[0, 0].fill_between(range(len(mean_crashes)), crashes.min(axis=0), crashes.max(axis=0),
                            color=color, alpha=0.2)
    smooth = np.convolve(rewards.mean(axis=0), np.ones(window) / window, mode="valid")
    axes[0, 1].plot(smooth, color=color, label=name)

axes[0, 0].set_title("Cumulative crashes during training (mean of 5 seeds, min-max band)")
axes[0, 0].set_xlabel("Episode")
axes[0, 0].set_ylabel("Crashes")
axes[0, 0].legend()
axes[0, 1].set_title(f"Episode reward (mean of 5 seeds, moving average {window})")
axes[0, 1].set_xlabel("Episode")
axes[0, 1].set_ylabel("Reward")
axes[0, 1].legend()

for ax, name in zip(axes[1], results):
    _, _, agent, env = results[name]
    draw_values(env.map, agent.Q.max(axis=1), agent.policy(), ax=ax,
                title=f"{name} (seed 0): learned values max Q(s, a)")

plt.tight_layout()
plt.savefig("results/sarsa_check.png", bbox_inches="tight", dpi=80)
plt.show()