"""Compute the metrics table from the saved training runs.

    python main.py evaluate

Reads results/<agent>/seed_<n>/ (written by `python main.py train`), computes
training and deployment metrics for every run, and writes:

    results/metrics_per_seed.csv   one row per agent and seed
    results/metrics_summary.md     mean (std) per agent, ready to paste in the README
"""

import csv
from pathlib import Path

import numpy as np

from saferoute.agents.factory import build_agent
from saferoute.environement.saferoute_env import SafeRouteEnv
from saferoute.planning.value_iteration import follow_policy, value_iteration
from saferoute.utils.io import load_run
from saferoute.utils.metrics import deployment_metrics, route_return, training_metrics
from scripts.train import load_config

# Columns of the summary table: (key, title, number format)
SUMMARY_COLUMNS = [
    ("training_crashes", "Training crashes", "{:.0f}"),
    ("first_success", "First success (episode)", "{:.0f}"),
    ("success", "Deployment success", "{:.0%}"),
    ("deployment_crash", "Deployment crashes", "{:.0f}"),
    ("path_length", "Path length", "{:.1f}"),
    ("risk_cost", "Route risk cost", "{:.1f}"),
    ("return", "Return", "{:.1f}"),
    ("gap_to_optimal", "Gap to optimal", "{:.1f}"),
]


def optimal_reference(config):
    """Value iteration with the risk-aware reward: the best possible return."""
    env = SafeRouteEnv.from_config(config, risk_costs=True)
    _, policy, _ = value_iteration(env, gamma=config["training"]["gamma"],
                                   theta=config["value_iteration"]["theta"])
    route = follow_policy(env, policy)
    return env, route, route_return(env, route)


def evaluate_all(config, results_dir=None):
    """Return a list of dicts, one per (agent, seed), with every metric."""
    results_dir = Path(results_dir or config["paths"]["results_dir"])
    reference_env, _, optimal_return = optimal_reference(config)

    rows = []
    for name in config["agents"]:
        for seed in config["training"]["seeds"]:
            history, q_table = load_run(results_dir, name, seed)

            env, agent = build_agent(name, config, seed=seed)   # same env and shield as in training
            agent.Q = q_table                                    # put the learned Q-table back

            row = {"agent": name, "seed": seed}
            row.update(training_metrics(history))
            row.update(deployment_metrics(env, agent, reference_env, optimal_return))
            rows.append(row)
    return rows


def summarize(rows, agent_names):
    """Mean and standard deviation over seeds, per agent and metric."""
    summary = {}
    for name in agent_names:
        agent_rows = [r for r in rows if r["agent"] == name]
        summary[name] = {
            key: (np.mean([float(r[key]) for r in agent_rows]),
                  np.std([float(r[key]) for r in agent_rows]))
            for key, _, _ in SUMMARY_COLUMNS
        }
    return summary


def summary_markdown(summary, optimal_return):
    header = "| Agent | " + " | ".join(title for _, title, _ in SUMMARY_COLUMNS) + " |"
    divider = "|---|" + "---:|" * len(SUMMARY_COLUMNS)
    lines = [header, divider]
    for name, metrics in summary.items():
        cells = []
        for key, _, fmt in SUMMARY_COLUMNS:
            mean, std = metrics[key]
            cell = fmt.format(mean)
            if std > 0 and key not in ("success", "deployment_crash"):
                cell += f" ({fmt.format(std)})"
            cells.append(cell)
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(f"Mean over seeds, standard deviation in parentheses. "
                 f"Return and gap are measured with the risk-aware reward for all agents "
                 f"(optimal return = {optimal_return:.0f}).")
    return "\n".join(lines)


def save_csv(rows, path):
    keys = [k for k in rows[0] if k != "route"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main(config_path="configs/default.yaml"):
    config = load_config(config_path)
    results_dir = Path(config["paths"]["results_dir"])

    rows = evaluate_all(config, results_dir)
    _, _, optimal_return = optimal_reference(config)
    summary = summarize(rows, list(config["agents"]))
    table = summary_markdown(summary, optimal_return)

    save_csv(rows, results_dir / "metrics_per_seed.csv")
    (results_dir / "metrics_summary.md").write_text(table + "\n")

    print(table)
    print(f"\nSaved {results_dir}/metrics_per_seed.csv and {results_dir}/metrics_summary.md")


if __name__ == "__main__":
    main()