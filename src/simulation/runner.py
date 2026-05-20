"""Simulation runner for REx experiments."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.agents.exploration import ReputationExplorationController
from src.agents.llm_agent import LLMAgent
from src.agents.reputation import ReputationModel
from src.backends.mock_backend import MockLLMBackend
from src.games.prisoner_dilemma import PrisonerDilemmaGame
from src.metrics.metrics import (
    action_entropy,
    average_payoff,
    average_reputation,
    average_temperature,
    cooperation_rate,
    defection_rate,
    reputation_variance,
    social_welfare,
)
from src.simulation.interventions import MaliciousAgentInjection, ReputationAttack
from src.utils.io import deep_update, load_yaml, write_csv_rows


AGENT_FIELDNAMES = [
    "experiment_name",
    "condition",
    "round",
    "agent_id",
    "opponent_id",
    "game",
    "action",
    "opponent_action",
    "payoff",
    "cumulative_payoff",
    "reputation",
    "temperature",
    "reflection_depth",
    "risk_tendency",
    "mode",
    "confidence",
    "risk_level",
    "reason",
    "is_malicious",
]

ROUND_FIELDNAMES = [
    "experiment_name",
    "condition",
    "round",
    "cooperation_rate",
    "defection_rate",
    "avg_reputation",
    "reputation_variance",
    "avg_temperature",
    "avg_payoff",
    "social_welfare",
    "action_entropy",
]


@dataclass
class SimulationResult:
    condition: str
    agent_csv: Path
    round_csv: Path
    agent_rows: list[dict[str, Any]]
    round_rows: list[dict[str, Any]]


@dataclass
class SimulationRunner:
    config: dict[str, Any]
    experiment_name: str = "rex_experiment"
    condition: str = "reputation_piecewise"
    rng: random.Random = field(init=False)
    agents: list[LLMAgent] = field(init=False, default_factory=list)
    game: PrisonerDilemmaGame = field(init=False)
    reputation_model: ReputationModel = field(init=False)
    exploration_controller: ReputationExplorationController = field(init=False)
    attack: ReputationAttack | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.seed = int(self.config.get("seed", 42))
        self.rng = random.Random(self.seed)
        self.game = self._build_game()
        self.reputation_model = self._build_reputation_model()
        self.exploration_controller = self._build_exploration_controller()
        self.agents = self._build_agents()
        self._apply_malicious_injection()
        self.attack = self._build_attack()

    @classmethod
    def from_config_files(
        cls,
        default_path: str | Path,
        experiments_path: str | Path,
        condition: str,
        experiment_name: str = "rex_experiment",
    ) -> "SimulationRunner":
        default_config = load_yaml(default_path)
        experiments = load_yaml(experiments_path)
        if condition not in experiments:
            raise KeyError(f"Unknown experiment condition: {condition}")
        config = deep_update(default_config, experiments[condition])
        return cls(config=config, experiment_name=experiment_name, condition=condition)

    def run(self) -> SimulationResult:
        agent_rows: list[dict[str, Any]] = []
        round_rows: list[dict[str, Any]] = []
        num_rounds = int(self.config.get("num_rounds", 100))

        for round_number in range(1, num_rounds + 1):
            if self.attack is not None:
                self.attack.apply(self.agents, round_number)
            for agent in self.agents:
                agent.update_exploration(self.exploration_controller)

            actions_this_round: list[str] = []
            payoffs_this_round: list[float] = []
            paired_agents = self._random_pairs()

            for agent_a, agent_b in paired_agents:
                decision_a = agent_a.decide(
                    opponent=agent_b,
                    game_name=self.game.name,
                    round_number=round_number,
                    malicious_warmup_rounds=self._malicious_warmup_rounds,
                    malicious_exploit_threshold=self._malicious_exploit_threshold,
                )
                decision_b = agent_b.decide(
                    opponent=agent_a,
                    game_name=self.game.name,
                    round_number=round_number,
                    malicious_warmup_rounds=self._malicious_warmup_rounds,
                    malicious_exploit_threshold=self._malicious_exploit_threshold,
                )
                action_a = decision_a["action"]
                action_b = decision_b["action"]
                payoff_a, payoff_b = self.game.payoff(action_a, action_b)

                agent_a.payoff += payoff_a
                agent_b.payoff += payoff_b
                agent_a.reputation = self.reputation_model.update(agent_a.reputation, action_a)
                agent_b.reputation = self.reputation_model.update(agent_b.reputation, action_b)

                self._record_memory(
                    round_number=round_number,
                    agent=agent_a,
                    opponent=agent_b,
                    self_action=action_a,
                    opponent_action=action_b,
                    self_payoff=payoff_a,
                    opponent_payoff=payoff_b,
                    reason=decision_a["reason"],
                )
                self._record_memory(
                    round_number=round_number,
                    agent=agent_b,
                    opponent=agent_a,
                    self_action=action_b,
                    opponent_action=action_a,
                    self_payoff=payoff_b,
                    opponent_payoff=payoff_a,
                    reason=decision_b["reason"],
                )

                actions_this_round.extend([action_a, action_b])
                payoffs_this_round.extend([payoff_a, payoff_b])
                agent_rows.extend(
                    [
                        self._agent_row(
                            round_number,
                            agent_a,
                            agent_b,
                            action_a,
                            action_b,
                            payoff_a,
                            decision_a,
                        ),
                        self._agent_row(
                            round_number,
                            agent_b,
                            agent_a,
                            action_b,
                            action_a,
                            payoff_b,
                            decision_b,
                        ),
                    ]
                )

            round_rows.append(
                self._round_row(
                    round_number=round_number,
                    actions=actions_this_round,
                    payoffs=payoffs_this_round,
                )
            )

        output_dir = Path(str(self.config.get("output_dir", "outputs/results")))
        agent_csv = output_dir / f"{self.condition}_agent_results.csv"
        round_csv = output_dir / f"{self.condition}_round_results.csv"
        write_csv_rows(agent_csv, AGENT_FIELDNAMES, agent_rows)
        write_csv_rows(round_csv, ROUND_FIELDNAMES, round_rows)
        return SimulationResult(
            condition=self.condition,
            agent_csv=agent_csv,
            round_csv=round_csv,
            agent_rows=agent_rows,
            round_rows=round_rows,
        )

    @property
    def _malicious_warmup_rounds(self) -> int:
        return int(self.config.get("malicious", {}).get("warmup_rounds", 20))

    @property
    def _malicious_exploit_threshold(self) -> float:
        return float(self.config.get("malicious", {}).get("exploit_threshold", 0.7))

    def _build_game(self) -> PrisonerDilemmaGame:
        game_name = str(self.config.get("game", "prisoner_dilemma"))
        if game_name != "prisoner_dilemma":
            raise ValueError(f"Unsupported game in minimal version: {game_name}")
        return PrisonerDilemmaGame()

    def _build_reputation_model(self) -> ReputationModel:
        reputation_config = self.config.get("reputation", {})
        return ReputationModel(
            alpha=float(reputation_config.get("alpha", 0.8)),
            defect_penalty_scale=float(reputation_config.get("defect_penalty_scale", 1.5)),
            cooperate_recovery_scale=float(
                reputation_config.get("cooperate_recovery_scale", 0.7)
            ),
        )

    def _build_exploration_controller(self) -> ReputationExplorationController:
        exploration_config = self.config.get("exploration", {})
        return ReputationExplorationController(
            mode=str(exploration_config.get("mode", "piecewise")),
            fixed_temperature=float(exploration_config.get("fixed_temperature", 0.5)),
            t_min=float(exploration_config.get("t_min", 0.2)),
            t_max=float(exploration_config.get("t_max", 0.8)),
        )

    def _build_agents(self) -> list[LLMAgent]:
        backend_name = str(self.config.get("backend", "mock"))
        if backend_name != "mock":
            raise ValueError(f"Unsupported backend in minimal version: {backend_name}")

        num_agents = int(self.config.get("num_agents", 20))
        initial_reputation = float(self.config.get("initial_reputation", 0.5))
        backend = MockLLMBackend(seed=self.seed + 1000)
        return [
            LLMAgent(
                agent_id=agent_id,
                reputation=initial_reputation,
                backend=backend,
            )
            for agent_id in range(num_agents)
        ]

    def _apply_malicious_injection(self) -> None:
        malicious_config = self.config.get("malicious", {})
        injector = MaliciousAgentInjection(
            malicious_fraction=float(malicious_config.get("fraction", 0.0)),
            seed=self.seed + 2000,
        )
        injector.apply(self.agents)

    def _build_attack(self) -> ReputationAttack | None:
        attack_config = self.config.get("attack", {})
        if not bool(attack_config.get("enabled", False)):
            return None
        return ReputationAttack(
            attack_round=int(attack_config.get("attack_round", 50)),
            target_fraction=float(attack_config.get("target_fraction", 0.2)),
            reputation_drop=float(attack_config.get("reputation_drop", 0.5)),
            seed=self.seed + 3000,
        )

    def _random_pairs(self) -> list[tuple[LLMAgent, LLMAgent]]:
        shuffled = list(self.agents)
        self.rng.shuffle(shuffled)
        return [
            (shuffled[index], shuffled[index + 1])
            for index in range(0, len(shuffled) - 1, 2)
        ]

    @staticmethod
    def _record_memory(
        round_number: int,
        agent: LLMAgent,
        opponent: LLMAgent,
        self_action: str,
        opponent_action: str,
        self_payoff: float,
        opponent_payoff: float,
        reason: str,
    ) -> None:
        agent.memory.add_record(
            round=round_number,
            opponent_id=opponent.agent_id,
            self_action=self_action,
            opponent_action=opponent_action,
            self_payoff=self_payoff,
            opponent_payoff=opponent_payoff,
            self_reputation=agent.reputation,
            opponent_reputation=opponent.reputation,
            reason=reason,
        )

    def _agent_row(
        self,
        round_number: int,
        agent: LLMAgent,
        opponent: LLMAgent,
        action: str,
        opponent_action: str,
        payoff: float,
        decision: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "experiment_name": self.experiment_name,
            "condition": self.condition,
            "round": round_number,
            "agent_id": agent.agent_id,
            "opponent_id": opponent.agent_id,
            "game": self.game.name,
            "action": action,
            "opponent_action": opponent_action,
            "payoff": payoff,
            "cumulative_payoff": agent.payoff,
            "reputation": agent.reputation,
            "temperature": agent.current_temperature,
            "reflection_depth": agent.reflection_depth,
            "risk_tendency": agent.risk_tendency,
            "mode": agent.mode,
            "confidence": decision["confidence"],
            "risk_level": decision["risk_level"],
            "reason": decision["reason"],
            "is_malicious": agent.is_malicious,
        }

    def _round_row(
        self,
        round_number: int,
        actions: list[str],
        payoffs: list[float],
    ) -> dict[str, Any]:
        reputations = [agent.reputation for agent in self.agents]
        temperatures = [agent.current_temperature for agent in self.agents]
        return {
            "experiment_name": self.experiment_name,
            "condition": self.condition,
            "round": round_number,
            "cooperation_rate": cooperation_rate(actions),
            "defection_rate": defection_rate(actions),
            "avg_reputation": average_reputation(reputations),
            "reputation_variance": reputation_variance(reputations),
            "avg_temperature": average_temperature(temperatures),
            "avg_payoff": average_payoff(payoffs),
            "social_welfare": social_welfare(payoffs),
            "action_entropy": action_entropy(actions),
        }


def run_conditions(
    default_path: str | Path,
    experiments_path: str | Path,
    conditions: list[str],
    experiment_name: str = "rex_experiment",
) -> list[SimulationResult]:
    results = []
    for condition in conditions:
        runner = SimulationRunner.from_config_files(
            default_path=default_path,
            experiments_path=experiments_path,
            condition=condition,
            experiment_name=experiment_name,
        )
        results.append(runner.run())
    return results
