"""
    Function to draw the grid map and Any route on top of it . 
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

from saferoute.environement.grid_map import Cell

# Same colors as the grid image in the README
CELL_STYLE = {
    Cell.NORMAL:      {"color": "#FFFFFF", "hatch": None,  "label": "Normal area"},
    Cell.BUILDING:    {"color": "#2E3338", "hatch": None,  "label": "Building"},
    Cell.LOW_RISK:    {"color": "#F7E48C", "hatch": None,  "label": "Low risk (soft)"},
    Cell.MEDIUM_RISK: {"color": "#F2A324", "hatch": "..",  "label": "Medium risk (soft)"},
    Cell.HIGH_RISK:   {"color": "#DC2B2B", "hatch": "///", "label": "High risk (hard)"},
}


def draw_map(grid_map, ax=None, route=None, title=None, show_legend=True):
    """Draw the grid. Optionally draw a route on top of it.

    grid_map : GridMap
    ax       : matplotlib axis to draw on (a new figure is created if None)
    route    : list of (col, row) positions visited by the drone
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    # 1. One colored square per cell
    for row in range(grid_map.height):
        for col in range(grid_map.width):
            style = CELL_STYLE[grid_map.cell((col, row))]
            ax.add_patch(Rectangle(
                (col, row), 1, 1,
                facecolor=style["color"],
                hatch=style["hatch"],
                edgecolor="#C9D3DD",
                linewidth=0.6,
            ))

    # 2. Optional route, drawn through the cell centers
    if route:
        xs = [col + 0.5 for col, _ in route]
        ys = [row + 0.5 for _, row in route]
        ax.plot(xs, ys, color="#1F6FEB", linewidth=3, zorder=3)

    # 3. Start and goal markers
    s_col, s_row = grid_map.start
    g_col, g_row = grid_map.goal
    ax.scatter(s_col + 0.5, s_row + 0.5, s=300, marker="s",
               color="#1F4E9C", zorder=4, label="Start (drone)")
    ax.scatter(g_col + 0.5, g_row + 0.5, s=350, marker="o",
               color="#1E8E4E", zorder=4, label="Destination")

    # 4. Axes: column numbers on top, row 0 at the top
    ax.set_xlim(0, grid_map.width)
    ax.set_ylim(grid_map.height, 0)
    ax.set_xticks([c + 0.5 for c in range(grid_map.width)])
    ax.set_xticklabels(range(grid_map.width))
    ax.set_yticks([r + 0.5 for r in range(grid_map.height)])
    ax.set_yticklabels(range(grid_map.height))
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    ax.set_aspect("equal")

    if title:
        ax.set_title(title, pad=30)

    if show_legend:
        handles = [
            Patch(facecolor=s["color"], hatch=s["hatch"], edgecolor="#888888", label=s["label"])
            for s in CELL_STYLE.values()
        ]
        handles += ax.get_legend_handles_labels()[0]
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False)

    return ax