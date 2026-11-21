"""Shared logic for the tabular agents (Q-Learning and SARSA).

What both agents have in common:
    - a Q-table of shape (n_states, n_actions), filled with zeros
    - an epsilon-greedy policy, with epsilon decreasing linearly over training
    - the episode loop (reset, act, step, learn)
    - an optional safety shield that restricts the allowed actions

What differs is only the update rule, written in the subclasses.
"""

import numpy as np


class BaseAgent:

    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay_episodes=2000,
                 seed=None, shield=None):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma

        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_episodes = epsilon_decay_episodes
        self.epsilon = epsilon_start

        self.rng = np.random.default_rng(seed)
        self.Q = np.zeros((n_states, n_actions))
        self.shield = shield                      # None = no shield

    # ------------------------------------------------------------------
    # Acting
    # ------------------------------------------------------------------

    def allowed_actions(self, state):
        """All actions, or only the safe ones if the agent has a shield."""
        if self.shield is None:
            return np.arange(self.n_actions)
        return np.array(self.shield.allowed(state))

    def greedy_action(self, state):
        """Best allowed action for this state. Ties are broken at random."""
        actions = self.allowed_actions(state)
        q_values = self.Q[state, actions]
        best = actions[q_values == q_values.max()]
        return int(self.rng.choice(best))

    def select_action(self, state):
        """Epsilon-greedy over the allowed actions."""
        if self.rng.random() < self.epsilon:
            return int(self.rng.choice(self.allowed_actions(state)))   # explore
        return self.greedy_action(state)                                # exploit

    def update_epsilon(self, episode):
        """Linear decay from epsilon_start to epsilon_end, then constant."""
        fraction = min(episode / self.epsilon_decay_episodes, 1.0)
        self.epsilon = self.epsilon_start + fraction * (self.epsilon_end - self.epsilon_start)

    def policy(self):
        """Greedy policy for every state (used to draw the learned route)."""
        return np.array([
            self.allowed_actions(s)[np.argmax(self.Q[s, self.allowed_actions(s)])]
            for s in range(self.n_states)
        ])

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def update(self, state, action, reward, next_state, next_action, terminated):
        """The learning rule. Each algorithm writes its own."""
        raise NotImplementedError

    def run_episode(self, env, learn=True):
        """Play one episode. Learn from every step if learn=True.

        Returns a dict of statistics about the episode.
        """
        state, info = env.reset()
        action = self.select_action(state)

        total_reward, steps, risky_steps = 0, 0, 0

        while True:
            next_state, reward, terminated, truncated, info = env.step(action)
            next_action = self.select_action(next_state)    # chosen now: SARSA needs it

            if learn:
                self.update(state, action, reward, next_state, next_action, terminated)

            total_reward += reward
            steps += 1
            risky_steps += info["risky"]

            if terminated or truncated:
                break
            state, action = next_state, next_action

        return {
            "reward": total_reward,
            "steps": steps,
            "success": info["success"],
            "crash": info["crash"],
            "risky_steps": risky_steps,
        }

    def train(self, env, episodes):
        """Train for a number of episodes. Returns the list of episode statistics."""
        history = []
        for episode in range(episodes):
            self.update_epsilon(episode)
            history.append(self.run_episode(env, learn=True))
        return history