"""Safety shield: blocks actions that would lead into a high-risk cell.

    A_safe(s) = { a in A : s'(s, a) is not a high-risk cell }

The shield sits between the agent and the environment. Before each action,
the agent may only choose among the safe actions of its current state.
Since the environment is deterministic, this guarantees zero crashes,
from the very first episode.

The shield knows WHERE the no-fly zones are (like published no-fly maps
for real drones). It does not know where the goal is or what the best
route is: the agent still has to learn that by itself.
"""


class Shield:

    def __init__(self, env):
        self.env = env
        grid_map = env.map
        n_actions = env.action_space.n

        # Pre-compute A_safe(s) for every state, once
        self.safe_actions = []
        for state in range(grid_map.n_states):
            pos = grid_map.to_pos(state)
            safe = [
                a for a in range(n_actions)
                if not grid_map.is_high_risk(env.next_position(pos, a))
            ]
            # Never leave the agent without a choice (cannot happen on this map)
            self.safe_actions.append(safe if safe else list(range(n_actions)))

    def allowed(self, state):
        """A_safe(s): the actions the agent is allowed to take in this state."""
        return self.safe_actions[state]

    def is_safe(self, state, action):
        return action in self.safe_actions[state]