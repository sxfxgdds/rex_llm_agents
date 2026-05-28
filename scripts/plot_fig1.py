"""Generate Fig. 1: Cooperation rate evolution under fixed temperature conditions."""

from __future__ import annotations

import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# --- Configuration ---
RESULTS_DIR = ROOT / "outputs" / "results"
FIGURES_DIR = ROOT / "outputs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

CONDITIONS = [
    ("fixed_low_temperature", r"Fixed $\tau = 0.2$", "#2196F3"),   # Blue
    ("fixed_mid_temperature", r"Fixed $\tau = 0.5$", "#FF9800"),   # Orange
    ("fixed_high_temperature", r"Fixed $\tau = 0.8$", "#E91E63"),  # Pink
]


def load_summary(condition: str) -> dict[int, tuple[float, float]]:
    """Load summary CSV, return {round: (mean, ci95)} for cooperation_rate."""
    path = RESULTS_DIR / f"{condition}_summary.csv"
    data: dict[int, tuple[float, float]] = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            r = int(float(row["round"]))
            mean = float(row["cooperation_rate_mean"])
            ci = float(row["cooperation_rate_ci95"])
            data[r] = (mean, ci)
    return data


def smooth(values: np.ndarray, window: int = 5) -> np.ndarray:
    """Simple moving average smoothing."""
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    smoothed = np.convolve(values, kernel, mode="same")
    # Preserve first and last values
    smoothed[0] = values[0]
    smoothed[-1] = values[-1]
    return smoothed


def main() -> None:
    # Load data
    all_data = {}
    for cond_key, _, _ in CONDITIONS:
        all_data[cond_key] = load_summary(cond_key)

    # Find common round range
    common_rounds = sorted(set.intersection(*[set(d.keys()) for d in all_data.values()]))

    # --- Plot ---
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 12,
        "axes.linewidth": 1.2,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.minor.width": 0.8,
        "ytick.minor.width": 0.8,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "0.8",
    })

    fig, ax = plt.subplots(figsize=(8, 5))

    for cond_key, label, color in CONDITIONS:
        data = all_data[cond_key]
        rounds = np.array(common_rounds)
        means = np.array([data[r][0] for r in common_rounds])
        cis = np.array([data[r][1] for r in common_rounds])

        # Smooth for cleaner visualization
        means_smooth = smooth(means, window=7)
        cis_smooth = smooth(cis, window=7)

        ax.plot(rounds, means_smooth, label=label, color=color, linewidth=2.2, zorder=3)
        ax.fill_between(rounds, means_smooth - cis_smooth, means_smooth + cis_smooth,
                        color=color, alpha=0.15, zorder=2)

    # Axis formatting
    ax.set_xlabel("Round", fontsize=13, labelpad=8)
    ax.set_ylabel("Cooperation Rate", fontsize=13, labelpad=8)
    ax.set_xlim(1, 300)
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(10))

    # Grid
    ax.grid(True, which="major", linestyle="--", alpha=0.3, zorder=0)
    ax.grid(True, which="minor", linestyle=":", alpha=0.15, zorder=0)

    # Legend
    legend = ax.legend(loc="lower right", fontsize=11, borderpad=0.8, handlelength=2.5)

    # Title (optional, can be removed for paper)
    # ax.set_title("Fig. 1: Cooperation Rate Under Fixed Temperature Conditions", fontsize=14, pad=12)

    fig.tight_layout()
    output_path = FIGURES_DIR / "fig1_cooperation_rate_fixed.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_path}")
    print(f"Saved: {output_path.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
