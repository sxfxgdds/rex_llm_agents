"""LLM agent wrapper for repeated social dilemma decisions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.agents.exploration import ReputationExplorationController
from src.agents.memory import AgentMemory
from src.backends.base_backend import BaseLLMBackend


@dataclass
class LLMAgent:
    agent_id: int
    reputation: float
    backend: BaseLLMBackend
    payoff: float = 0.0
    memory: AgentMemory = field(default_factory=AgentMemory)
    current_temperature: float = 0.5
    reflection_depth: int = 1
    risk_tendency: str = "medium"
    strategy_profile: str = "reputation_regulated"
    is_malicious: bool = False
    mode: str = "normal"

    def update_exploration(self, controller: ReputationExplorationController) -> None:
        state = controller.get_state(self.reputation, self.memory)
        self.current_temperature = state.temperature
        self.reflection_depth = state.reflection_depth
        self.risk_tendency = state.risk_tendency
        self.mode = state.mode

    def decide(
        self,
        opponent: "LLMAgent",
        game_name: str,
        round_number: int,
        malicious_warmup_rounds: int = 20,
        malicious_exploit_threshold: float = 0.7,
    ) -> dict[str, Any]:
        prompt = self._build_prompt(opponent=opponent, game_name=game_name, round_number=round_number)
        metadata = {
            "agent_id": self.agent_id,
            "opponent_id": opponent.agent_id,
            "own_reputation": self.reputation,
            "opponent_reputation": opponent.reputation,
            "temperature": self.current_temperature,
            "reflection_depth": self.reflection_depth,
            "risk_tendency": self.risk_tendency,
            "mode": self.mode,
            "is_malicious": self.is_malicious,
            "round": round_number,
            "game": game_name,
            "malicious_warmup_rounds": malicious_warmup_rounds,
            "malicious_exploit_threshold": malicious_exploit_threshold,
        }
        decision = self.backend.generate_decision(
            prompt=prompt,
            temperature=self.current_temperature,
            metadata=metadata,
        )
        return self._normalize_decision(decision)

    def _build_prompt(self, opponent: "LLMAgent", game_name: str, round_number: int) -> str:
        recovery_text = "yes" if self.mode == "recovery" else "no"
        return (
            f"Game: {game_name}. Round: {round_number}. "
            f"Agent {self.agent_id} reputation: {self.reputation:.3f}. "
            f"Opponent {opponent.agent_id} reputation: {opponent.reputation:.3f}. "
            f"Temperature: {self.current_temperature:.3f}. "
            f"Reflection depth: {self.reflection_depth}. "
            f"Risk tendency: {self.risk_tendency}. Recovery mode: {recovery_text}. "
            f"Recent memory: {self.memory.summarize_for_prompt(k=5)}"
        )

    @staticmethod
    def _normalize_decision(decision: dict[str, Any]) -> dict[str, Any]:
        action = str(decision.get("action", "cooperate")).lower()
        if action not in {"cooperate", "defect"}:
            action = "cooperate"
        confidence = float(decision.get("confidence", 0.5))
        risk_level = float(decision.get("risk_level", 0.5))
        reason = str(decision.get("reason", "No reason provided."))
        return {
            "action": action,
            "confidence": min(1.0, max(0.0, confidence)),
            "risk_level": min(1.0, max(0.0, risk_level)),
            "reason": reason,
        }
