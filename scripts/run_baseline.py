"""Run baseline REx experiment conditions."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.simulation.runner import run_conditions


BASELINE_CONDITIONS = [
    "fixed_low_temperature",
    "fixed_mid_temperature",
    "fixed_high_temperature",
    "reputation_linear",
    "reputation_piecewise",
]


def main() -> None:
    results = run_conditions(
        default_path=ROOT / "config/default.yaml",
        experiments_path=ROOT / "config/experiments.yaml",
        conditions=BASELINE_CONDITIONS,
        experiment_name="rex_baseline",
    )
    for result in results:
        print(f"wrote {result.agent_csv}")
        print(f"wrote {result.round_csv}")


if __name__ == "__main__":
    main()
