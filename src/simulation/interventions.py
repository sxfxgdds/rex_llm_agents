"""Simulation interventions for robustness experiments."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from src.agents.llm_agent import LLMAgent


@dataclass
class ReputationAttack:
    attack_round: int
    target_fraction: float
    reputation_drop: float
    seed: int
    applied: bool = False
    targeted_agent_ids: list[int] = field(default_factory=list)

    def apply(self, agents: list[LLMAgent], round_number: int) -> list[int]:
        if self.applied or round_number != self.attack_round:
            return []
        if self.target_fraction <= 0.0 or self.reputation_drop <= 0.0:
            self.applied = True
            return []

        rng = random.Random(self.seed)
        target_count = max(1, int(round(len(agents) * self.target_fraction)))
        target_count = min(target_count, len(agents))
        targets = rng.sample(agents, target_count)
        for agent in targets:
            agent.reputation = max(0.0, agent.reputation - self.reputation_drop)
        self.targeted_agent_ids = [agent.agent_id for agent in targets]
        self.applied = True
        return self.targeted_agent_ids


@dataclass(frozen=True)
class MaliciousAgentInjection:
    malicious_fraction: float
    seed: int

    def apply(self, agents: list[LLMAgent]) -> list[int]:
        if self.malicious_fraction <= 0.0:
            return []
        rng = random.Random(self.seed)
        malicious_count = max(1, int(round(len(agents) * self.malicious_fraction)))
        malicious_count = min(malicious_count, len(agents))
        selected = rng.sample(agents, malicious_count)
        for agent in selected:
            agent.is_malicious = True
            agent.strategy_profile = "malicious_reputation_gaming"
        return [agent.agent_id for agent in selected]
