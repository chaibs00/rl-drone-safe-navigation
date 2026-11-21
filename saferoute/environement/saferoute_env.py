"""
SafeRoute Gymnasium environment.

The drone moves on the grid with 4 actions. The environment is deterministic.

    State   : one number = row * width + col  (256 states)
    Actions : 0 = up, 1 = down, 2 = left, 3 = right

Rules:
    - Moving into a building or off the grid: the drone stays in place.
    - Entering the goal: +100, episode ends (success).
    - Entering a high-risk cell: -100, episode ends (crash).
    - Any other move: -1, minus the risk cost c(s') if risk_costs=True.
    - After max_steps moves the episode is cut (timeout).
"""

import gymnasium as gym
from gymnasium import spaces

from saferoute.environement.grid_map import Cell, GridMap

UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
ACTION_NAMES = ["up", "down", "left", "right"]


# (change in col, change in row) for each action. Row 0 is at the top.
MOVES = {
    UP:    (0, -1),
    DOWN:  (0, 1),
    LEFT:  (-1, 0),
    RIGHT: (1, 0),
}


class SafeRouteEnv(gym.Env):

    metadata = {"render_modes": []}

    def __init__(self, grid_map, rewards, max_steps=200, risk_costs=False):
        super().__init__()
        self.map = grid_map
        self.rewards = rewards
        self.max_steps = max_steps
        self.risk_costs = risk_costs

        self.observation_space = spaces.Discrete(grid_map.n_states)
        self.action_space = spaces.Discrete(len(MOVES))

        self.pos = None
        self.steps = 0

    @classmethod
    def from_config(cls, config, risk_costs=False):
        """Build the environment from the loaded default.yaml."""
        grid_map = GridMap(config["env"]["map_path"])
        return cls(
            grid_map,
            rewards=config["rewards"],
            max_steps=config["env"]["max_steps"],
            risk_costs=risk_costs,
        )

    # ------------------------------------------------------------------
    # The model of the world: P (next_position) and R (reward).
    # They are pure functions, so the shield and value iteration can
    # reuse them without moving the drone.
    # ------------------------------------------------------------------

    def next_position(self, pos, action):
        """P(s' | s, a): where the drone ends up. Deterministic."""
        d_col, d_row = MOVES[action]
        new_pos = (pos[0] + d_col, pos[1] + d_row)

        if not self.map.in_bounds(new_pos) or self.map.is_building(new_pos):
            return pos          # blocked: stay in place
        return new_pos

    def risk_cost(self, pos):
        """c(s'): extra cost for soft-risk cells (0 if risk costs are off)."""
        if not self.risk_costs:
            return 0
        cell = self.map.cell(pos)
        if cell == Cell.MEDIUM_RISK:
            return -self.rewards["medium_risk"]     # config stores -5, cost is 5
        if cell == Cell.LOW_RISK:
            return -self.rewards["low_risk"]
        return 0

    def reward(self, next_pos):
        """R(s, a, s'): reward for arriving in next_pos."""
        if next_pos == self.map.goal:
            return self.rewards["goal"]
        if self.map.is_high_risk(next_pos):
            return self.rewards["crash"]
        return self.rewards["step"] - self.risk_cost(next_pos)

    def is_terminal(self, pos):
        """T: goal or high-risk cell."""
        return pos == self.map.goal or self.map.is_high_risk(pos)

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.pos = self.map.start
        self.steps = 0
        return self.map.to_state(self.pos), self._info()

    def step(self, action):
        next_pos = self.next_position(self.pos, action)
        reward = self.reward(next_pos)

        self.pos = next_pos
        self.steps += 1

        terminated = self.is_terminal(next_pos)                       # goal or crash
        truncated = not terminated and self.steps >= self.max_steps   # timeout

        return self.map.to_state(self.pos), reward, terminated, truncated, self._info()

    def _info(self):
        cell = self.map.cell(self.pos)
        return {
            "pos": self.pos,
            "cell": cell.name,
            "success": self.pos == self.map.goal,
            "crash": cell == Cell.HIGH_RISK,
            "risky": cell in (Cell.LOW_RISK, Cell.MEDIUM_RISK),
        }