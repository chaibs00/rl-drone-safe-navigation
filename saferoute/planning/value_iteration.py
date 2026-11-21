"""
    Value iteration: compute the optimal policy directly from the known map.

    - It uses the model of the environment (next_position, reward, is_terminal) 
    to compute the best possible route.

    Bellman optimality update (deterministic environment):

        V(s) <- max_a [ R(s, a, s') + gamma * V(s') ]

    with V(s') = 0 when s' is terminal (goal or crash).
"""

import numpy as np


# Function to compute the value of taking an action in a given state, then acting optimally. 
def q_value(env, V, pos, action, gamma):
    """Value of taking `action` in `pos`, then acting optimally: R + gamma * V(s')."""
    next_pos = env.next_position(pos, action)
    reward = env.reward(next_pos)
    if env.is_terminal(next_pos):
        return reward                     # no future after a terminal state
    return reward + gamma * V[env.map.to_state(next_pos)]


# Function to compute the optimal value function and policy using value iteration.
def value_iteration(env, gamma=0.99, theta=1e-6, max_iterations=10_000):
    """Return (V, policy, n_iterations).

    V      : array of 256 state values
    policy : array of 256 best actions (-1 for buildings and terminal cells)
    """
    grid_map = env.map
    n_states = grid_map.n_states
    n_actions = env.action_space.n

    # States where the drone can actually be and still has to decide
    decision_states = [
        s for s in range(n_states)
        if not grid_map.is_building(grid_map.to_pos(s))
        and not env.is_terminal(grid_map.to_pos(s))
    ]

    V = np.zeros(n_states)

    for iteration in range(1, max_iterations + 1):
        V_new = np.zeros(n_states)

        for state in decision_states:
            pos = grid_map.to_pos(state)
            V_new[state] = max(q_value(env, V, pos, a, gamma) for a in range(n_actions))

        delta = np.max(np.abs(V_new - V))   # biggest change in this sweep
        V = V_new
        if delta < theta:                   # values stopped changing: converged
            break

    # Extract the policy: in each state, take the action with the best value
    policy = np.full(n_states, -1)
    for state in decision_states:
        pos = grid_map.to_pos(state)
        values = [q_value(env, V, pos, a, gamma) for a in range(n_actions)]
        policy[state] = int(np.argmax(values))

    return V, policy, iteration


def follow_policy(env, policy, max_steps=None):
    """Play one episode with a fixed policy. Return the list of visited positions."""
    max_steps = max_steps or env.max_steps
    pos = env.map.start
    route = [pos]

    for _ in range(max_steps):
        action = policy[env.map.to_state(pos)]
        pos = env.next_position(pos, action)
        route.append(pos)
        if env.is_terminal(pos):
            break

    return route