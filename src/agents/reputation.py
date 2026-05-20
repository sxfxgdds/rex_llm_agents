"""Reputation dynamics with asymmetric hysteresis."""

from __future__ import annotations

from dataclasses import dataclass


def clip_reputation(value: float) -> float:
    return min(1.0, max(0.0, value))


@dataclass(frozen=True)
class ReputationModel:
    """Update reputation from observed cooperative or defective actions."""

    alpha: float = 0.8
    defect_penalty_scale: float = 1.5
    cooperate_recovery_scale: float = 0.7

    def update(self, reputation: float, action: str) -> float:
        old_reputation = clip_reputation(reputation)
        observed_score = self._observed_score(action)
        base_next = clip_reputation(
            self.alpha * old_reputation + (1.0 - self.alpha) * observed_score
        )
        delta = base_next - old_reputation

        if action == "defect" and delta < 0.0:
            delta *= self.defect_penalty_scale
        elif action == "cooperate" and delta > 0.0:
            delta *= self.cooperate_recovery_scale

        return clip_reputation(old_reputation + delta)

    @staticmethod
    def _observed_score(action: str) -> float:
        if action == "cooperate":
            return 1.0
        if action == "defect":
            return 0.0
        raise ValueError(f"Unknown action for reputation update: {action}")
