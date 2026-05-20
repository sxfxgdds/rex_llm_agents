"""Agent memory for repeated social interactions."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class InteractionRecord:
    """One remembered interaction from an agent's perspective."""

    round: int
    opponent_id: int
    self_action: str
    opponent_action: str
    self_payoff: float
    opponent_payoff: float
    self_reputation: float
    opponent_reputation: float
    reason: str


@dataclass
class AgentMemory:
    """Chronological interaction memory for a single LLM agent."""

    records: list[InteractionRecord] = field(default_factory=list)

    def add_record(
        self,
        round: int,
        opponent_id: int,
        self_action: str,
        opponent_action: str,
        self_payoff: float,
        opponent_payoff: float,
        self_reputation: float,
        opponent_reputation: float,
        reason: str,
    ) -> None:
        self.records.append(
            InteractionRecord(
                round=round,
                opponent_id=opponent_id,
                self_action=self_action,
                opponent_action=opponent_action,
                self_payoff=self_payoff,
                opponent_payoff=opponent_payoff,
                self_reputation=self_reputation,
                opponent_reputation=opponent_reputation,
                reason=reason,
            )
        )

    def get_recent(self, k: int) -> list[InteractionRecord]:
        if k <= 0:
            return []
        return self.records[-k:]

    def get_history_with(self, opponent_id: int) -> list[InteractionRecord]:
        return [record for record in self.records if record.opponent_id == opponent_id]

    def summarize_for_prompt(self, k: int = 5) -> str:
        recent = self.get_recent(k)
        if not recent:
            return "No prior interactions."
        summaries = []
        for record in recent:
            summaries.append(
                "round={round}, opponent={opponent}, self={self_action}, "
                "opponent_action={opponent_action}, payoff={payoff:.2f}, "
                "opponent_payoff={opponent_payoff:.2f}, reputation={rep:.3f}".format(
                    round=record.round,
                    opponent=record.opponent_id,
                    self_action=record.self_action,
                    opponent_action=record.opponent_action,
                    payoff=record.self_payoff,
                    opponent_payoff=record.opponent_payoff,
                    rep=record.self_reputation,
                )
            )
        return " | ".join(summaries)

    def defected_recently(self, k: int = 3) -> bool:
        return any(record.self_action == "defect" for record in self.get_recent(k))
