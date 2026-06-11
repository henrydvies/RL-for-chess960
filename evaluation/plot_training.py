"""
Plot ep_rew_mean over total cumulative timesteps for each opponent type.
Usage: python -m evaluation.plot_training --model models/rl_agent_v1
Outputs PNG graphs to visualisation/<model_name>/
"""
import json
import os
import argparse
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
from collections import defaultdict


def load_log(model_folder):
    """
    Load training_log.json from model folder.
    """
    log_path = os.path.join(model_folder, "training_log.json")
    with open(log_path, "r") as f:
        return json.load(f)


def plot_training(model_folder):
    """
    Generate one graph per opponent type showing ep_rew_mean over total cumulative timesteps.
    X axis is total timesteps trained across all opponents at the time of each run.
    """
    log = load_log(model_folder)
    model_name = os.path.basename(model_folder)

    # Output folder
    output_folder = os.path.join("visualisation", model_name)
    os.makedirs(output_folder, exist_ok=True)

    # Calculate total cumulative timesteps at each run chronologically
    cumulative_at_timestamp = {}
    total = 0
    for run in log["runs"]:
        total += run["timesteps"]
        cumulative_at_timestamp[run["timestamp"]] = total

    # Group runs by opponent (only those with ep_rew_mean)
    opponent_runs = defaultdict(list)
    for run in log["runs"]:
        if run["ep_rew_mean"] is not None:
            opponent_runs[run["opponent"]].append(run)

    # Plot one graph per opponent
    for opponent_name, runs in opponent_runs.items():
        runs_sorted = sorted(runs, key=lambda r: r["timestamp"])

        x = [cumulative_at_timestamp[run["timestamp"]] for run in runs_sorted]
        y = [run["ep_rew_mean"] for run in runs_sorted]

        if not x:
            continue

        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(
            x,
            y,
            marker="o",
            markersize=3,
            linewidth=1.5,
            color="steelblue",
            label="ep_rew_mean"
        )

        ax.axhline(
            0,
            color="gray",
            linestyle="--",
            linewidth=0.8,
            label="Neutral (0.0)"
        )

        # Format x axis in millions
        ax.xaxis.set_major_formatter(FuncFormatter(format_timesteps))

        ax.set_title(
            f"ep_rew_mean vs {opponent_name} — {model_name}",
            fontsize=13
        )

        ax.set_xlabel(
            "Total Timesteps Trained (all opponents)",
            fontsize=11
        )

        ax.set_ylabel(
            "Mean Episode Reward",
            fontsize=11
        )

        # RL rewards are bounded between -1 and 1
        ax.set_ylim(-1, 1)
        ax.yaxis.set_major_locator(MultipleLocator(0.1))

        ax.grid(
            True,
            which="major",
            linestyle="--",
            linewidth=0.5,
            alpha=0.35
        )

        # Cleaner publication-style axes
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if len(y) >= 10:
            window = min(10, len(y))

            rolling = [
                sum(y[max(0, i - window):i]) / min(i, window)
                for i in range(1, len(y) + 1)
            ]

            ax.plot(
                x,
                rolling,
                color="orange",
                linewidth=2.5,
                linestyle="--",
                label=f"{window}-run rolling avg"
            )

        ax.legend(frameon=False)

        plt.tight_layout()

        output_path = os.path.join(
            output_folder,
            f"{opponent_name}.png"
        )

        plt.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        print(f"Saved: {output_path}")

    print(f"\nAll graphs saved to {output_folder}/")

def format_timesteps(value, _):
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.0f}K"
    return str(int(value))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        default="models/rl_agent_v4",
        help="Path to model folder"
    )

    args = parser.parse_args()

    plot_training(args.model)
    
    


