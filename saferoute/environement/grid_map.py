"""

    Load the SafeRoute map from a text file. 
    And converting the text-based map into a NumPy grid that an RL agent can interact with.

    Each character in the file is one cell:
        .  normal area       #  building
        L  low risk          M  medium risk       H  high risk
        S  start (normal)    G  goal (normal)

    Positions are written as (col, row), like in the README.
    The numpy grid is indexed as grid[row, col].

"""

from enum import IntEnum
from pathlib import Path

import numpy as np


class Cell(IntEnum):
    """The five cell types of the map."""
    NORMAL = 0
    BUILDING = 1
    LOW_RISK = 2
    MEDIUM_RISK = 3
    HIGH_RISK = 4


CHAR_TO_CELL = {
    ".": Cell.NORMAL,
    "#": Cell.BUILDING,
    "L": Cell.LOW_RISK,
    "M": Cell.MEDIUM_RISK,
    "H": Cell.HIGH_RISK,
    "S": Cell.NORMAL,   # the start cell is a normal cell
    "G": Cell.NORMAL,   # the goal cell is a normal cell
}


class GridMap:
    """The map: cell types, size, start and goal."""

    def __init__(self, path):
        lines = self._read_lines(path)

        self.height = len(lines)
        self.width = len(lines[0])
        self.grid = np.zeros((self.height, self.width), dtype=int)
        self.start = None
        self.goal = None

        for row, line in enumerate(lines):
            if len(line) != self.width:
                raise ValueError(f"Row {row} has {len(line)} cells, expected {self.width}")

            for col, char in enumerate(line):
                if char not in CHAR_TO_CELL:
                    raise ValueError(f"Unknown character '{char}' at (col={col}, row={row})")

                self.grid[row, col] = CHAR_TO_CELL[char]

                if char == "S":
                    if self.start is not None:
                        raise ValueError("The map has more than one start 'S'")
                    self.start = (col, row)
                elif char == "G":
                    if self.goal is not None:
                        raise ValueError("The map has more than one goal 'G'")
                    self.goal = (col, row)

        if self.start is None or self.goal is None:
            raise ValueError("The map needs exactly one 'S' and one 'G'")

    @staticmethod
    def _read_lines(path):
        """Read the file and drop empty lines and surrounding spaces."""
        text = Path(path).read_text()
        return [line.strip() for line in text.splitlines() if line.strip()]

    # ---------- questions about a position (col, row) ----------

    def in_bounds(self, pos):
        col, row = pos
        return 0 <= col < self.width and 0 <= row < self.height

    def cell(self, pos):
        col, row = pos
        return Cell(self.grid[row, col])

    def is_building(self, pos):
        return self.cell(pos) == Cell.BUILDING

    def is_high_risk(self, pos):
        return self.cell(pos) == Cell.HIGH_RISK

    # ---------- position <-> state number ----------

    @property
    def n_states(self):
        return self.width * self.height

    def to_state(self, pos):
        """(col, row) -> single number, used as the row index of the Q-table."""
        col, row = pos
        return row * self.width + col

    def to_pos(self, state):
        """Single number -> (col, row)."""
        row, col = divmod(state, self.width)
        return (col, row)