"""Run malicious-agent injection experiment with multi-seed support."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.simulation.runner import MultiSeedResult, run_conditions
from src.utils.io import load_yaml


def main() -> None:
    default_config = load_yaml(ROOT / "config/default.yaml")
    seeds = default_config.get("seeds", list(range(42, 72)))

    print(f"Running malicious_agents x {len(seeds)} seeds")

    results = run_conditions(
        default_path=ROOT / "config/default.yaml",
        experiments_path=ROOT / "config/experiments.yaml",
        conditions=["reputation_piecewise_with_malicious_agents"],
        experiment_name="rex_malicious_agents",
        seeds=seeds,
    )
    for result in results:
        if isinstance(result, MultiSeedResult):
            print(f"[{result.condition}] {len(result.seeds)} seeds -> {result.summary_csv}")
        else:
            print(f"[{result.condition}] single run -> {result.round_csv}")


if __name__ == "__main__":
    main()
