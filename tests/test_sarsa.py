"""
   Units tests for value iteration planning algorithm.

   # Run all test 
        python -m pytest tests -v

   # Run individual test
        python -m pytest tests/test_sarsa.py -v

"""


from pathlib import Path

import numpy as np

from saferoute.agents.q_learning import QLearningAgent
from saferoute.agents.sarsa import SarsaAgent
from saferoute.environement.grid_map import GridMap
from saferoute.environement.saferoute_env import SafeRouteEnv
from saferoute.planning.value_iteration import follow_policy

ROOT = Path(__file__).resolve().parents[1]
MAP_PATH = ROOT / "maps" / "saferoute_16x16.txt"
REWARDS = {"goal": 100, "step": -1, "crash": -100, "medium_risk": -5, "low_risk": -2}


def test_update_uses_the_action_actually_taken():
    agent = SarsaAgent(n_states=2, n_actions=4, alpha=0.1, gamma=0.99)
    agent.Q[1] = [0, 10, 0, 0]                    # best next action (1) is worth 10...
    agent.update(state=0, action=2, reward=-1, next_state=1, next_action=0, terminated=False)
    # ...but the agent will actually take action 0, worth 0
    # target = -1 + 0.99 * 0 = -1   ->   Q = 0 + 0.1 * (-1 - 0) = -0.1
    assert np.isclose(agent.Q[0, 2], -0.1)


def test_same_update_as_q_learning_when_next_action_is_the_best():
    q_agent = QLearningAgent(2, 4)
    s_agent = SarsaAgent(2, 4)
    for agent in (q_agent, s_agent):
        agent.Q[1] = [0, 10, 0, 0]
        agent.update(state=0, action=2, reward=-1, next_state=1, next_action=1, terminated=False)
    assert np.isclose(q_agent.Q[0, 2], s_agent.Q[0, 2])


def train(agent_class, seed=0):
    env = SafeRouteEnv(GridMap(MAP_PATH), REWARDS)
    agent = agent_class(env.observation_space.n, env.action_space.n, seed=seed)
    history = agent.train(env, episodes=3000)
    return env, agent, history


def test_trained_sarsa_finds_shortest_route():
    env, agent, _ = train(SarsaAgent)
    route = follow_policy(env, agent.policy())
    assert route[-1] == env.map.goal
    assert len(route) - 1 == 26


def test_sarsa_crashes_less_than_q_learning_during_training():
    _, _, q_history = train(QLearningAgent)
    _, _, s_history = train(SarsaAgent)
    q_crashes = sum(h["crash"] for h in q_history)
    s_crashes = sum(h["crash"] for h in s_history)
    assert s_crashes < q_crashes