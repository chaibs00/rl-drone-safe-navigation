"""
    Testing the Environment Implmentation . 

    python -m pytest tests -v

"""



from pathlib import Path

from saferoute.environement.grid_map import GridMap
from saferoute.environement.saferoute_env import DOWN, LEFT, RIGHT, UP, SafeRouteEnv

ROOT = Path(__file__).resolve().parents[1]
MAP_PATH = ROOT / "maps" / "saferoute_16x16.txt"
REWARDS = {"goal": 100, "step": -1, "crash": -100, "medium_risk": -5, "low_risk": -2}


def make_env(risk_costs=False, max_steps=200):
    return SafeRouteEnv(GridMap(MAP_PATH), REWARDS, max_steps=max_steps, risk_costs=risk_costs)


def test_reset_starts_at_start():
    env = make_env()
    state, info = env.reset(seed=0)
    assert info["pos"] == (1, 1)
    assert state == 17


def test_normal_move():
    env = make_env()
    env.reset()
    _, reward, terminated, truncated, info = env.step(DOWN)
    assert info["pos"] == (1, 2)
    assert reward == -1
    assert not terminated and not truncated


def test_building_blocks_the_drone():
    env = make_env()
    env.reset()
    env.step(RIGHT)                       # (1,1) -> (2,1)
    _, reward, _, _, info = env.step(RIGHT)   # (3,1) is a building
    assert info["pos"] == (2, 1)
    assert reward == -1


def test_grid_edge_blocks_the_drone():
    env = make_env()
    env.reset()
    env.step(UP)                          # (1,1) -> (1,0)
    _, _, _, _, info = env.step(UP)       # off the grid
    assert info["pos"] == (1, 0)


def test_crash_in_high_risk_cell():
    env = make_env()
    env.reset()
    env.pos = (13, 7)                     # open street next to the market
    _, reward, terminated, _, info = env.step(LEFT)    # (12,7) is high risk
    assert reward == -100
    assert terminated
    assert info["crash"]


def test_reaching_the_goal():
    env = make_env()
    env.reset()
    env.pos = (13, 14)
    _, reward, terminated, _, info = env.step(RIGHT)
    assert reward == 100
    assert terminated
    assert info["success"]


def test_risk_costs_off_and_on():
    for risk_costs, expected in [(False, -1), (True, -6)]:
        env = make_env(risk_costs=risk_costs)
        env.reset()
        env.pos = (6, 11)
        _, reward, _, _, info = env.step(RIGHT)        # (7,11) is medium risk
        assert reward == expected
        assert info["risky"]

    env = make_env(risk_costs=True)
    env.reset()
    env.pos = (2, 13)
    _, reward, _, _, _ = env.step(RIGHT)               # (3,13) is low risk
    assert reward == -3


def test_timeout():
    env = make_env(max_steps=3)
    env.reset()
    for _ in range(2):
        _, _, terminated, truncated, _ = env.step(UP)
        assert not terminated and not truncated
    _, _, terminated, truncated, _ = env.step(UP)
    assert not terminated and truncated


def test_example_route_reaches_goal_in_26_steps():
    env = make_env()
    env.reset()
    actions = [RIGHT] + [DOWN] * 10 + [RIGHT] * 12 + [DOWN] * 3
    total = 0
    for action in actions:
        _, reward, terminated, _, info = env.step(action)
        total += reward
    assert terminated and info["success"]
    assert len(actions) == 26
    assert total == 100 - 25              # 25 normal steps, then the goal