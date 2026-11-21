"""
   Units tests for the shield (agent 4).

   # Run all test 
        python -m pytest tests -v

   # Run individual test
        python -m pytest tests/test_shield.py -v

"""



from pathlib import Path

import yaml

from saferoute.agents.factory import build_agent
from saferoute.agents.q_learning import QLearningAgent
from saferoute.agents.shield import Shield
from saferoute.environement.grid_map import GridMap
from saferoute.environement.saferoute_env import DOWN, LEFT, RIGHT, UP, SafeRouteEnv
from saferoute.planning.value_iteration import follow_policy

ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load(open("config.yaml"))
CONFIG["env"]["map_path"] = str(ROOT / CONFIG["env"]["map_path"])
REWARDS = CONFIG["rewards"]


def make_env():
    return SafeRouteEnv(GridMap(CONFIG["env"]["map_path"]), REWARDS, risk_costs=True)


def test_shield_blocks_only_moves_into_red():
    env = make_env()
    shield = Shield(env)
    next_to_red = env.map.to_state((13, 7))            # the market is on the left
    assert shield.allowed(next_to_red) == [UP, DOWN, RIGHT]
    far_away = env.map.to_state((2, 2))
    assert shield.allowed(far_away) == [UP, DOWN, LEFT, RIGHT]


def test_greedy_never_picks_a_blocked_action():
    env = make_env()
    agent = QLearningAgent(256, 4, shield=Shield(env), seed=0)
    state = env.map.to_state((13, 7))
    agent.Q[state] = [0, 0, 999, 0]                     # LEFT (into red) looks best...
    assert all(agent.greedy_action(state) != LEFT for _ in range(100))


def test_zero_crashes_even_when_acting_fully_at_random():
    env = make_env()
    agent = QLearningAgent(256, 4, shield=Shield(env), seed=0)
    agent.epsilon = 1.0                                 # pure random exploration
    crashes = sum(agent.run_episode(env, learn=False)["crash"] for _ in range(500))
    assert crashes == 0


def test_shielded_agent_never_crashes_during_training():
    env, agent = build_agent("shielded", CONFIG, seed=0)
    history = agent.train(env, CONFIG["training"]["episodes"])
    assert sum(h["crash"] for h in history) == 0

    route = follow_policy(env, agent.policy())
    assert route[-1] == env.map.goal
    assert len(route) - 1 == 26