"""
    Tests for the Q-learning agent.

    # Run all test :
          python -m pytest tests -v

    # Run individual test
         python -m pytest tests.test_q_learning 


"""


from pathlib import Path

import numpy as np

from saferoute.agents.q_learning import QLearningAgent
from saferoute.environement.grid_map import GridMap
from saferoute.environement.saferoute_env import SafeRouteEnv
from saferoute.planning.value_iteration import follow_policy

ROOT = Path(__file__).resolve().parents[1]
MAP_PATH = ROOT / "maps" / "saferoute_16x16.txt"
REWARDS = {"goal": 100, "step": -1, "crash": -100, "medium_risk": -5, "low_risk": -2}


def test_update_uses_best_next_action():
    agent = QLearningAgent(n_states=2, n_actions=4, alpha=0.1, gamma=0.99)
    agent.Q[1] = [0, 10, 0, 0]                    # best next action is worth 10
    agent.update(state=0, action=2, reward=-1, next_state=1, next_action=0, terminated=False)
    # target = -1 + 0.99 * 10 = 8.9   ->   Q = 0 + 0.1 * (8.9 - 0) = 0.89
    assert np.isclose(agent.Q[0, 2], 0.89)


def test_update_on_terminal_ignores_future():
    agent = QLearningAgent(n_states=2, n_actions=4, alpha=0.1, gamma=0.99)
    agent.Q[1] = [50, 50, 50, 50]                 # must be ignored
    agent.update(state=0, action=0, reward=-100, next_state=1, next_action=0, terminated=True)
    assert np.isclose(agent.Q[0, 0], -10.0)       # 0.1 * (-100 - 0)


def test_epsilon_decays_linearly():
    agent = QLearningAgent(2, 4, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay_episodes=2000)
    agent.update_epsilon(0)
    assert np.isclose(agent.epsilon, 1.0)
    agent.update_epsilon(1000)
    assert np.isclose(agent.epsilon, 0.525)
    agent.update_epsilon(5000)
    assert np.isclose(agent.epsilon, 0.05)


def test_ties_are_broken_at_random():
    agent = QLearningAgent(2, 4, seed=0)          # Q is all zeros: 4-way tie
    chosen = {agent.greedy_action(0) for _ in range(100)}
    assert chosen == {0, 1, 2, 3}


def test_trained_agent_finds_shortest_route():
    env = SafeRouteEnv(GridMap(MAP_PATH), REWARDS)
    agent = QLearningAgent(env.observation_space.n, env.action_space.n, seed=0)
    history = agent.train(env, episodes=3000)

    route = follow_policy(env, agent.policy())
    assert route[-1] == env.map.goal
    assert len(route) - 1 == 26
    assert all(h["success"] for h in history[-100:])