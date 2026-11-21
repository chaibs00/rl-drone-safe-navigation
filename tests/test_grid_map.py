"""
    Testing the GridMap implementation.

   # Run all test 
        python -m pytest tests -v

   # Run individual test
        python -m pytest tests/test_grid_map.py -v
"""




from pathlib import Path

from saferoute.environement.grid_map import Cell, GridMap

ROOT = Path(__file__).resolve().parents[1]
MAP_PATH = ROOT / "maps" / "saferoute_16x16.txt"


def load():
    return GridMap(MAP_PATH)


def test_size():
    m = load()
    assert (m.width, m.height) == (16, 16)
    assert m.n_states == 256


def test_start_and_goal():
    m = load()
    assert m.start == (1, 1)
    assert m.goal == (14, 14)


def test_some_cells_match_the_grid_image():
    m = load()
    assert m.cell((3, 0)) == Cell.BUILDING      # top buildings
    assert m.cell((7, 0)) == Cell.MEDIUM_RISK   # medium corridor
    assert m.cell((9, 8)) == Cell.HIGH_RISK     # market
    assert m.cell((4, 13)) == Cell.LOW_RISK     # park
    assert m.cell((13, 7)) == Cell.NORMAL       # open street


def test_cell_counts():
    m = load()
    assert (m.grid == Cell.HIGH_RISK).sum() == 30     # 6 columns x 5 rows
    assert (m.grid == Cell.LOW_RISK).sum() == 16      # 4 columns x 4 rows


def test_state_conversion_round_trip():
    m = load()
    for state in range(m.n_states):
        assert m.to_state(m.to_pos(state)) == state
    assert m.to_state((1, 1)) == 17                   # row 1 * 16 + col 1


def test_in_bounds():
    m = load()
    assert m.in_bounds((0, 0))
    assert m.in_bounds((15, 15))
    assert not m.in_bounds((-1, 0))
    assert not m.in_bounds((16, 3))