"""
Plot ep_rew_mean over total cumulative timesteps for each opponent type.
Usage: python -m evaluation.plot_training --model models/rl_agent_v1
Outputs PNG graphs to visualisation/<model_name>/
"""
import json
import os
import argparse
import matplotlib.pyplot as plt
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

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(x, y, marker="o", markersize=3, linewidth=1.5, color="steelblue", label="ep_rew_mean")
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, label="Neutral (0.0)")
        ax.set_title(f"ep_rew_mean vs {opponent_name} — {model_name}", fontsize=13)
        ax.set_xlabel("Total Timesteps Trained (all opponents)", fontsize=11)
        ax.set_ylabel("Mean Episode Reward", fontsize=11)
        ax.grid(True, alpha=0.3)

        if len(y) >= 5:
            window = min(5, len(y))
            rolling = [
                sum(y[max(0, i - window):i]) / min(i, window)
                for i in range(1, len(y) + 1)
            ]
            ax.plot(x, rolling, color="orange", linewidth=2, linestyle="--", label=f"{window}-run rolling avg")

        ax.legend()
        plt.tight_layout()

        output_path = os.path.join(output_folder, f"{opponent_name}.png")
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"Saved: {output_path}")

    print(f"\nAll graphs saved to {output_folder}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot training progress from log file")
    parser.add_argument("--model", type=str, default="models/rl_agent_v3", help="Path to model folder")
    args = parser.parse_args()
    plot_training(args.model)