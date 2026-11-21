"""
    A quick script to check that value iteration is working as expected on the saferoute environment.

    # Run in root 
        python -m scripts.check_q_learning
"""



import matplotlib.pyplot as plt
import numpy as np
import yaml

from saferoute.agents.q_learning import QLearningAgent
from saferoute.environement.saferoute_env import SafeRouteEnv
from saferoute.planning.value_iteration import follow_policy
from saferoute.utils.plotting import draw_map, draw_values

config = yaml.safe_load(open("./config.yaml"))
t = config["training"]

env = SafeRouteEnv.from_config(config, risk_costs=False)
agent = QLearningAgent(
    env.observation_space.n, env.action_space.n,
    alpha=t["alpha"], gamma=t["gamma"],
    epsilon_start=t["epsilon_start"], epsilon_end=t["epsilon_end"],
    epsilon_decay_episodes=t["epsilon_decay_episodes"],
    seed=0,
)
history = agent.train(env, t["episodes"])

rewards = np.array([h["reward"] for h in history])
crashes = np.array([h["crash"] for h in history])
successes = np.array([h["success"] for h in history])
route = follow_policy(env, agent.policy())

print(f"Training crashes : {crashes.sum()} / {len(history)} episodes")
print(f"First success    : episode {successes.argmax()}")
print(f"Success rate, last 500 episodes: {successes[-500:].mean():.0%}")
print(f"Learned route    : {len(route) - 1} steps, ends at {route[-1]}, "
      f"risk cells crossed = {sum(env.map.cell(p).name.endswith('RISK') for p in route)}")

window = 50
smooth = np.convolve(rewards, np.ones(window) / window, mode="valid")

fig, axes = plt.subplots(2, 2, figsize=(16, 14))
axes[0, 0].plot(smooth)
axes[0, 0].set_title(f"Episode reward (moving average, {window} episodes)")
axes[0, 0].set_xlabel("Episode")
axes[0, 0].set_ylabel("Reward")

axes[0, 1].plot(np.cumsum(crashes), color="#DC2B2B")
axes[0, 1].set_title("Cumulative crashes during training")
axes[0, 1].set_xlabel("Episode")
axes[0, 1].set_ylabel("Crashes")

draw_values(env.map, agent.Q.max(axis=1), agent.policy(), ax=axes[1, 0],
            title="Learned values max Q(s, a) and policy")
draw_map(env.map, route=route, ax=axes[1, 1], show_legend=False,
         title=f"Learned route ({len(route) - 1} steps)")

plt.tight_layout()
plt.savefig("results/q_learning_check.png", bbox_inches="tight", dpi=80)
plt.show()