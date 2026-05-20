"""Plot all available REx round-level results."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.visualization.plots import create_all_plots


def main() -> None:
    paths = create_all_plots(
        results_dir=ROOT / "outputs/results",
        figures_dir=ROOT / "outputs/figures",
    )
    if not paths:
        print("no round-level result CSV files found")
        return
    for path in paths:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
