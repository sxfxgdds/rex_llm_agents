"""Reputation-regulated exploration controls."""

from __future__ import annotations

from dataclasses import dataclass

from src.agents.memory import AgentMemory


@dataclass(frozen=True)
class ExplorationState:
    temperature: float
    reflection_depth: int
    risk_tendency: str
    mode: str


@dataclass(frozen=True)
class ReputationExplorationController:
    """Map reputation and recent behavior into exploration parameters."""

    mode: str = "piecewise"
    fixed_temperature: float = 0.5
    t_min: float = 0.2
    t_max: float = 0.8
    shock_window: int = 3
    shock_temperature_multiplier: float = 0.6

    def get_state(self, reputation: float, memory: AgentMemory | None = None) -> ExplorationState:
        bounded_reputation = min(1.0, max(0.0, reputation))

        if self.mode == "fixed":
            state = ExplorationState(
                temperature=self.fixed_temperature,
                reflection_depth=1,
                risk_tendency="medium",
                mode="fixed",
            )
        elif self.mode == "linear":
            temperature = self.t_min + (self.t_max - self.t_min) * bounded_reputation
            state = self._state_from_reputation(
                bounded_reputation,
                temperature=temperature,
                mode_override="linear",
            )
        elif self.mode == "piecewise":
            state = self._piecewise_state(bounded_reputation)
        else:
            raise ValueError(f"Unknown exploration mode: {self.mode}")

        if (
            self.mode != "fixed"
            and memory is not None
            and memory.defected_recently(self.shock_window)
        ):
            return ExplorationState(
                temperature=max(0.0, state.temperature * self.shock_temperature_multiplier),
                reflection_depth=state.reflection_depth,
                risk_tendency=state.risk_tendency,
                mode=state.mode,
            )
        return state

    def _piecewise_state(self, reputation: float) -> ExplorationState:
        if reputation < 0.3:
            return ExplorationState(
                temperature=0.2,
                reflection_depth=3,
                risk_tendency="low",
                mode="recovery",
            )
        if reputation < 0.7:
            return ExplorationState(
                temperature=0.5,
                reflection_depth=2,
                risk_tendency="medium",
                mode="normal",
            )
        return ExplorationState(
            temperature=0.75,
            reflection_depth=1,
            risk_tendency="medium_high",
            mode="exploration",
        )

    def _state_from_reputation(
        self, reputation: float, temperature: float, mode_override: str
    ) -> ExplorationState:
        if reputation < 0.3:
            return ExplorationState(
                temperature=temperature,
                reflection_depth=3,
                risk_tendency="low",
                mode="recovery",
            )
        if reputation < 0.7:
            return ExplorationState(
                temperature=temperature,
                reflection_depth=2,
                risk_tendency="medium",
                mode=mode_override,
            )
        return ExplorationState(
            temperature=temperature,
            reflection_depth=1,
            risk_tendency="medium_high",
            mode=mode_override,
        )
