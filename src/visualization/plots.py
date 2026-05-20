"""Matplotlib plots for REx experiment outputs."""

from __future__ import annotations

import csv
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_MPL_CACHE_DIR = _PROJECT_ROOT / "outputs" / ".matplotlib_cache"
_MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


MetricSpec = tuple[str, str, str]


METRICS: list[MetricSpec] = [
    ("cooperation_rate", "Cooperation rate", "cooperation_rate_over_time.png"),
    ("avg_reputation", "Average reputation", "avg_reputation_over_time.png"),
    ("avg_temperature", "Average temperature", "avg_temperature_over_time.png"),
    ("social_welfare", "Social welfare", "social_welfare_over_time.png"),
    ("action_entropy", "Action entropy", "action_entropy_over_time.png"),
]


def load_round_results(results_dir: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(Path(results_dir).glob("*_round_results.csv")):
        with path.open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                rows.append(row)
    return rows


def create_all_plots(
    results_dir: str | Path = "outputs/results",
    figures_dir: str | Path = "outputs/figures",
) -> list[Path]:
    rows = load_round_results(results_dir)
    output_dir = Path(figures_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not rows:
        return []

    paths = []
    for metric, ylabel, filename in METRICS:
        paths.append(_plot_metric(rows, metric, ylabel, output_dir / filename))

    attack_rows = [
        row for row in rows if "attack" in str(row.get("condition", "")).lower()
    ]
    if attack_rows:
        paths.append(_plot_attack_recovery(attack_rows, output_dir / "reputation_attack_recovery.png"))
    return paths


def _plot_metric(rows: list[dict[str, Any]], metric: str, ylabel: str, output_path: Path) -> Path:
    grouped = _group_by_condition(rows)
    fig, ax = plt.subplots(figsize=(8, 5))
    for condition, condition_rows in grouped.items():
        ordered = sorted(condition_rows, key=lambda row: _as_int(row["round"]))
        rounds = [_as_int(row["round"]) for row in ordered]
        values = [_as_float(row[metric]) for row in ordered]
        ax.plot(rounds, values, label=condition, linewidth=1.8)
    ax.set_xlabel("Round")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} over time")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def _plot_attack_recovery(rows: list[dict[str, Any]], output_path: Path) -> Path:
    grouped = _group_by_condition(rows)
    fig, ax = plt.subplots(figsize=(8, 5))
    for condition, condition_rows in grouped.items():
        ordered = sorted(condition_rows, key=lambda row: _as_int(row["round"]))
        rounds = [_as_int(row["round"]) for row in ordered]
        reputation = [_as_float(row["avg_reputation"]) for row in ordered]
        cooperation = [_as_float(row["cooperation_rate"]) for row in ordered]
        ax.plot(rounds, reputation, label=f"{condition}: reputation", linewidth=1.8)
        ax.plot(rounds, cooperation, label=f"{condition}: cooperation", linewidth=1.4)
    ax.axvline(50, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
    ax.set_xlabel("Round")
    ax.set_ylabel("Rate / average reputation")
    ax.set_title("Reputation attack recovery")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def _group_by_condition(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("condition", "unknown"))].append(row)
    return dict(grouped)


def _as_float(value: Any) -> float:
    return float(value)


def _as_int(value: Any) -> int:
    return int(float(value))
