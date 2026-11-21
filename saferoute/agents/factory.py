"""Build any of the 4 agents (and its environment) from the config.

The config describes each agent as 1 algorithm + 2 switches:

    agents:
      q_learning: {algorithm: q_learning, risk_costs: false, shield: false}
      sarsa:      {algorithm: sarsa,      risk_costs: false, shield: false}
      risk_aware: {algorithm: q_learning, risk_costs: true,  shield: false}
      shielded:   {algorithm: q_learning, risk_costs: true,  shield: true}

So building an agent means: pick the algorithm, build the environment with
the right reward (risk_costs), and add the shield if asked.
"""

from saferoute.agents.q_learning import QLearningAgent
from saferoute.agents.sarsa import SarsaAgent
from saferoute.agents.shield import Shield
from saferoute.environement.saferoute_env import SafeRouteEnv

ALGORITHMS = {
    "q_learning": QLearningAgent,
    "sarsa": SarsaAgent,
}


def build_agent(name, config, seed=None):
    """Return (env, agent) for the agent called `name` in the config."""
    spec = config["agents"][name]
    t = config["training"]

    env = SafeRouteEnv.from_config(config, risk_costs=spec["risk_costs"])

    agent_class = ALGORITHMS[spec["algorithm"]]
    agent = agent_class(
        env.observation_space.n, env.action_space.n,
        alpha=t["alpha"], gamma=t["gamma"],
        epsilon_start=t["epsilon_start"], epsilon_end=t["epsilon_end"],
        epsilon_decay_episodes=t["epsilon_decay_episodes"],
        seed=seed,
        shield=Shield(env) if spec["shield"] else None,
    )

    return env, agent