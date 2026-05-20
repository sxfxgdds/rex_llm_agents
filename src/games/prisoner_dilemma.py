"""Repeated Prisoner's Dilemma game."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PrisonerDilemmaGame:
    """Two-player Prisoner's Dilemma with canonical payoffs."""

    name: str = "prisoner_dilemma"
    actions: tuple[str, str] = ("cooperate", "defect")
    payoff_matrix: dict[tuple[str, str], tuple[float, float]] = field(
        default_factory=lambda: {
            ("cooperate", "cooperate"): (3.0, 3.0),
            ("cooperate", "defect"): (0.0, 5.0),
            ("defect", "cooperate"): (5.0, 0.0),
            ("defect", "defect"): (1.0, 1.0),
        }
    )

    def payoff(self, action_a: str, action_b: str) -> tuple[float, float]:
        if action_a not in self.actions:
            raise ValueError(f"Invalid action for player A: {action_a}")
        if action_b not in self.actions:
            raise ValueError(f"Invalid action for player B: {action_b}")
        return self.payoff_matrix[(action_a, action_b)]
