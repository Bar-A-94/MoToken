#!/usr/bin/env python3
import argparse
import matplotlib.pyplot as plt
import numpy as np

def parse_data(file_path):
    """
    Read the diff data file and parse MSE differences into categories.
    Expected lines in the file:
      Man diff: 0.010694526135921478
      sheep diff: 0.02805955708026886
      ...
    """
    diffs = {"man": [], "sheep": [], "boxing": [], "swimming": []}
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            key_part, val_part = line.split(' diff:')
            key = key_part.lower().strip()
            if key == "swiming":  # correct the typo
                key = "swimming"
            value = float(val_part.strip())
            if key in diffs:
                diffs[key].append(value)
    return diffs

def plot_grouped_bars(diffs, output_path):
    """
    Generate and save a grouped bar chart of MSE differences per step for each category.
    """
    categories = ["man", "sheep", "boxing", "swimming"]
    labels = ["Man", "Sheep", "Boxing", "Swimming"]
    data = [diffs[c] for c in categories]
    
    n_steps = len(data[0])
    x = np.arange(n_steps)
    width = 0.2

    plt.figure(figsize=(12, 6))

    # Plot bars for each category
    for i, (d, label) in enumerate(zip(data, labels)):
        plt.bar(x + i * width, d, width=width, label=label)

    plt.xlabel("Step Index")
    plt.ylabel("MSE Difference")
    plt.title("Per-Step Prediction Differences\nSubjects vs. Motions")
    plt.xticks(x + width * 1.5, [str(i+1) for i in x], rotation=90)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser(
        description="Plot grouped bars per step of diffs to compare subjects vs. motions."
    )
    parser.add_argument(
        "input_file",
        help="Path to the text file containing lines like 'Man diff: 0.0106...'"
    )
    parser.add_argument(
        "--output", "-o",
        default="diff_grouped_bars.png",
        help="Output filename for the saved plot (PNG)."
    )
    args = parser.parse_args()

    diffs = parse_data(args.input_file)
    plot_grouped_bars(diffs, args.output)
    print(f"✔ Grouped bar chart saved to {args.output}")

if __name__ == "__main__":
    main()