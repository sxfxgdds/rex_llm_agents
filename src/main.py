"""Command-line entry point for running REx experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.simulation.runner import run_conditions


def main() -> None:
    parser = argparse.ArgumentParser(description="Run REx simulation experiments.")
    parser.add_argument(
        "conditions",
        nargs="*",
        default=["reputation_piecewise"],
        help="Experiment condition names from config/experiments.yaml.",
    )
    parser.add_argument("--default-config", default="config/default.yaml")
    parser.add_argument("--experiments-config", default="config/experiments.yaml")
    args = parser.parse_args()

    results = run_conditions(
        default_path=Path(args.default_config),
        experiments_path=Path(args.experiments_config),
        conditions=list(args.conditions),
    )
    for result in results:
        print(f"{result.condition}: {result.round_csv}")


if __name__ == "__main__":
    main()
