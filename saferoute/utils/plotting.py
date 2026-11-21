"""
    Function to draw the grid map and Any route on top of it . 
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import numpy as np

from saferoute.environement.grid_map import Cell


# Same colors and design as the grid image in the README
CELL_STYLE = {
    Cell.NORMAL:      {"color": "#FFFFFF", "hatch": None,  "label": "Normal area"},
    Cell.BUILDING:    {"color": "#2E3338", "hatch": None,  "label": "Building"},
    Cell.LOW_RISK:    {"color": "#F7E48C", "hatch": None,  "label": "Low risk (soft)"},
    Cell.MEDIUM_RISK: {"color": "#F2A324", "hatch": "..",  "label": "Medium risk (soft)"},
    Cell.HIGH_RISK:   {"color": "#DC2B2B", "hatch": "///", "label": "High risk (hard)"},
}


# Main function to draw the grid map and any route on top of it 
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



# =========================== After defining the value policy ==================================

# Arrow direction (d_col, d_row) for each action: up, down, left, right
ARROWS = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)}


def draw_values(grid_map, V, policy=None, ax=None, title=None):
    """Draw the state values V as a heatmap, with the policy as arrows.

    Buildings and high-risk cells are drawn in their own colors (no value).
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    values = V.reshape(grid_map.height, grid_map.width).astype(float)
    blocked = (grid_map.grid == Cell.BUILDING) | (grid_map.grid == Cell.HIGH_RISK)
    values[blocked] = np.nan                      # no color for these cells

    image = ax.imshow(values, cmap="viridis",
                      extent=(0, grid_map.width, grid_map.height, 0))
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="V(s)")

    # Buildings and high-risk cells on top, in the usual colors
    for row in range(grid_map.height):
        for col in range(grid_map.width):
            cell = grid_map.cell((col, row))
            if cell in (Cell.BUILDING, Cell.HIGH_RISK):
                style = CELL_STYLE[cell]
                ax.add_patch(Rectangle((col, row), 1, 1, facecolor=style["color"],
                                       hatch=style["hatch"], edgecolor="#C9D3DD", linewidth=0.6))

    # Policy arrows
    if policy is not None:
        for state, action in enumerate(policy):
            # if action < 0:
            #     continue
            # col, row = grid_map.to_pos(state)
            
            col, row = grid_map.to_pos(state)
            if action < 0 or blocked[row, col] or (col, row) == grid_map.goal:
                continue                              # no arrow where the drone never decides

            d_col, d_row = ARROWS[int(action)]
            ax.arrow(col + 0.5 - 0.25 * d_col, row + 0.5 - 0.25 * d_row,
                     0.4 * d_col, 0.4 * d_row,
                     head_width=0.2, head_length=0.12, color="white", length_includes_head=True)

    g_col, g_row = grid_map.goal
    ax.scatter(g_col + 0.5, g_row + 0.5, s=250, marker="o", color="#1E8E4E", zorder=4)

    ax.set_xticks([c + 0.5 for c in range(grid_map.width)])
    ax.set_xticklabels(range(grid_map.width))
    ax.set_yticks([r + 0.5 for r in range(grid_map.height)])
    ax.set_yticklabels(range(grid_map.height))
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    if title:
        ax.set_title(title, pad=30)
    return ax




# ========================== After defining all agents ==================================
# ----------------------------------------------------------------------
# Result plots (Step 11)
# ----------------------------------------------------------------------

# One label and color per agent, used in every result figure
AGENT_STYLE = {
    "q_learning": ("Q-Learning", "#1F6FEB"),
    "sarsa":      ("SARSA", "#E36209"),
    "risk_aware": ("Risk-aware Q-Learning", "#8250DF"),
    "shielded":   ("Shielded Q-Learning", "#1A7F37"),
}


def moving_average(values, window):
    return np.convolve(values, np.ones(window) / window, mode="valid")


def plot_mean_and_band(ax, curves, name):
    """Plot the mean of several seeds as a line, and the min-max range as a band."""
    label, color = AGENT_STYLE[name]
    curves = np.asarray(curves, dtype=float)
    x = np.arange(curves.shape[1])
    ax.plot(x, curves.mean(axis=0), color=color, label=label, linewidth=2)
    ax.fill_between(x, curves.min(axis=0), curves.max(axis=0), color=color, alpha=0.15)


def style_axis(ax, title, xlabel, ylabel):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)