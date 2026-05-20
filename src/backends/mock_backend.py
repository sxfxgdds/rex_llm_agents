"""Seedable mock backend for offline LLM-agent simulations."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from src.backends.base_backend import BaseLLMBackend


def _clip(value: float, lower: float, upper: float) -> float:
    return min(upper, max(lower, value))


@dataclass
class MockLLMBackend(BaseLLMBackend):
    """A probabilistic stand-in for an LLM decision backend.

    The backend uses structured metadata instead of external API calls. It is
    intentionally simple, seedable, and stable enough for tests and baselines.
    """

    seed: int = 42
    rng: random.Random = field(init=False)

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)

    def generate_decision(
        self, prompt: str, temperature: float, metadata: dict[str, Any]
    ) -> dict[str, Any]:
        del prompt
        own_reputation = float(metadata.get("own_reputation", 0.5))
        opponent_reputation = float(metadata.get("opponent_reputation", 0.5))
        risk_tendency = str(metadata.get("risk_tendency", "medium"))
        mode = str(metadata.get("mode", "normal"))
        is_malicious = bool(metadata.get("is_malicious", False))

        if is_malicious:
            cooperate_probability = self._malicious_cooperation_probability(metadata)
        else:
            cooperate_probability = self._cooperation_probability(
                own_reputation=own_reputation,
                opponent_reputation=opponent_reputation,
                risk_tendency=risk_tendency,
                mode=mode,
            )

        cooperate_probability = self._apply_temperature(
            probability=cooperate_probability,
            temperature=temperature,
        )
        action = "cooperate" if self.rng.random() < cooperate_probability else "defect"
        confidence = _clip(max(cooperate_probability, 1.0 - cooperate_probability), 0.0, 1.0)
        risk_level = self._risk_level(action=action, risk_tendency=risk_tendency)
        reason = self._reason(
            action=action,
            cooperate_probability=cooperate_probability,
            mode=mode,
            risk_tendency=risk_tendency,
            is_malicious=is_malicious,
        )

        return {
            "action": action,
            "confidence": confidence,
            "risk_level": risk_level,
            "reason": reason,
        }

    def _cooperation_probability(
        self,
        own_reputation: float,
        opponent_reputation: float,
        risk_tendency: str,
        mode: str,
    ) -> float:
        probability = 0.45 + 0.25 * own_reputation + 0.30 * opponent_reputation
        if mode == "recovery":
            probability += 0.20
        if risk_tendency == "low":
            probability += 0.10
        elif risk_tendency == "medium_high":
            probability -= 0.05
        return _clip(probability, 0.02, 0.98)

    def _malicious_cooperation_probability(self, metadata: dict[str, Any]) -> float:
        own_reputation = float(metadata.get("own_reputation", 0.5))
        round_number = int(metadata.get("round", 0))
        warmup_rounds = int(metadata.get("malicious_warmup_rounds", 20))
        exploit_threshold = float(metadata.get("malicious_exploit_threshold", 0.7))

        if own_reputation < 0.3:
            return 0.88
        if round_number <= warmup_rounds:
            return 0.82
        if own_reputation > exploit_threshold:
            return 0.18
        return 0.45

    def _apply_temperature(self, probability: float, temperature: float) -> float:
        bounded_temperature = _clip(temperature, 0.0, 1.0)
        adjusted = probability

        if bounded_temperature < 0.5:
            sharpen = (0.5 - bounded_temperature) / 0.5
            if adjusted >= 0.5:
                adjusted += (1.0 - adjusted) * 0.45 * sharpen
            else:
                adjusted *= 1.0 - 0.45 * sharpen
        else:
            soften = (bounded_temperature - 0.5) / 0.5
            adjusted = adjusted * (1.0 - 0.35 * soften) + 0.5 * (0.35 * soften)

        noise_span = 0.20 * bounded_temperature
        adjusted += self.rng.uniform(-noise_span, noise_span)
        return _clip(adjusted, 0.02, 0.98)

    @staticmethod
    def _risk_level(action: str, risk_tendency: str) -> float:
        baseline = {
            "low": 0.25,
            "medium": 0.50,
            "medium_high": 0.65,
            "high": 0.80,
        }.get(risk_tendency, 0.50)
        if action == "defect":
            baseline += 0.15
        return _clip(baseline, 0.0, 1.0)

    @staticmethod
    def _reason(
        action: str,
        cooperate_probability: float,
        mode: str,
        risk_tendency: str,
        is_malicious: bool,
    ) -> str:
        agent_type = "malicious" if is_malicious else "standard"
        return (
            f"{agent_type} mock policy chose {action}; "
            f"p_cooperate={cooperate_probability:.3f}, mode={mode}, risk={risk_tendency}"
        )
