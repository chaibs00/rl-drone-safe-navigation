"""
    Testing the GridMap implementation.

    # Run all tests
        python -m pytest tests -v
    
    # Run individual test
        python -m tests.test_draw_map
"""


import matplotlib.pyplot as plt

from saferoute.environement.grid_map import GridMap
from saferoute.utils.plotting import draw_map

grid_map = GridMap("maps/saferoute_16x16.txt")

# A hand-made example route (down column 2, right along row 11, down to the goal)
route = [(1, 1), (2, 1)]
route += [(2, r) for r in range(2, 12)]
route += [(c, 11) for c in range(3, 15)]
route += [(14, r) for r in range(12, 15)]

fig, axes = plt.subplots(1, 2, figsize=(16, 8))
draw_map(grid_map, ax=axes[0], title="Map", show_legend=False)
draw_map(grid_map, ax=axes[1], route=route, title=f"Example route ({len(route) - 1} steps)")

plt.savefig("results/map_check.png", bbox_inches="tight", dpi=100)
plt.show()