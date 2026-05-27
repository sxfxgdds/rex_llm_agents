"""Simulation runner for REx experiments."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.agents.exploration import ReputationExplorationController
from src.agents.llm_agent import LLMAgent
from src.agents.reputation import ReputationModel
from src.backends.base_backend import BaseLLMBackend
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

SUMMARY_FIELDNAMES = [
    "experiment_name",
    "condition",
    "round",
    "num_seeds",
    "cooperation_rate_mean",
    "cooperation_rate_ci95",
    "defection_rate_mean",
    "defection_rate_ci95",
    "avg_reputation_mean",
    "avg_reputation_ci95",
    "reputation_variance_mean",
    "reputation_variance_ci95",
    "avg_temperature_mean",
    "avg_temperature_ci95",
    "avg_payoff_mean",
    "avg_payoff_ci95",
    "social_welfare_mean",
    "social_welfare_ci95",
    "action_entropy_mean",
    "action_entropy_ci95",
]


@dataclass
class SimulationResult:
    condition: str
    agent_csv: Path
    round_csv: Path
    agent_rows: list[dict[str, Any]]
    round_rows: list[dict[str, Any]]


@dataclass
class MultiSeedResult:
    condition: str
    seeds: list[int]
    summary_csv: Path
    summary_rows: list[dict[str, Any]]


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
        self._agents_built = False
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

    @classmethod
    def run_multi_seed(
        cls,
        default_path: str | Path,
        experiments_path: str | Path,
        condition: str,
        experiment_name: str = "rex_experiment",
        seeds: list[int] | None = None,
    ) -> "MultiSeedResult":
        if seeds is None:
            seeds = list(range(42, 72))

        all_round_rows: list[list[dict[str, Any]]] = []
        for seed in seeds:
            default_config = load_yaml(default_path)
            experiments = load_yaml(experiments_path)
            config = deep_update(default_config, experiments[condition])
            config["seed"] = seed
            runner = cls(config=config, experiment_name=experiment_name, condition=condition)
            result = runner.run()
            all_round_rows.append(result.round_rows)

        summary_rows = cls._aggregate_seeds(experiment_name, condition, seeds, all_round_rows)

        output_dir = Path(all_round_rows[0][0].get("experiment_name", experiment_name)).parent if all_round_rows else Path("outputs/results")
        output_dir = Path("outputs/results")
        summary_csv = output_dir / f"{condition}_summary.csv"
        write_csv_rows(summary_csv, SUMMARY_FIELDNAMES, summary_rows)

        return MultiSeedResult(
            condition=condition,
            seeds=seeds,
            summary_csv=summary_csv,
            summary_rows=summary_rows,
        )

    @staticmethod
    def _aggregate_seeds(
        experiment_name: str,
        condition: str,
        seeds: list[int],
        all_round_rows: list[list[dict[str, Any]]],
    ) -> list[dict[str, Any]]:
        num_rounds = len(all_round_rows[0])
        num_seeds = len(seeds)
        metric_keys = [
            "cooperation_rate",
            "defection_rate",
            "avg_reputation",
            "reputation_variance",
            "avg_temperature",
            "avg_payoff",
            "social_welfare",
            "action_entropy",
        ]

        summary_rows: list[dict[str, Any]] = []
        for r in range(num_rounds):
            row: dict[str, Any] = {
                "experiment_name": experiment_name,
                "condition": condition,
                "round": r + 1,
                "num_seeds": num_seeds,
            }
            for key in metric_keys:
                values = [float(seed_rows[r][key]) for seed_rows in all_round_rows]
                mean = sum(values) / num_seeds
                if num_seeds > 1:
                    variance = sum((v - mean) ** 2 for v in values) / (num_seeds - 1)
                    std_err = math.sqrt(variance / num_seeds)
                    ci95 = 1.96 * std_err
                else:
                    ci95 = 0.0
                row[f"{key}_mean"] = round(mean, 6)
                row[f"{key}_ci95"] = round(ci95, 6)
            summary_rows.append(row)
        return summary_rows

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
        backend_config = self.config.get("backend", {})
        if isinstance(backend_config, str):
            backend_config = {"type": backend_config}
        backend_name = str(backend_config.get("type", "mock"))
        num_agents = int(self.config.get("num_agents", 20))
        initial_reputation = float(self.config.get("initial_reputation", 0.5))
        backend = self._build_backend(backend_name, backend_config)
        return [
            LLMAgent(
                agent_id=agent_id,
                reputation=initial_reputation,
                backend=backend,
            )
            for agent_id in range(num_agents)
        ]

    def _build_backend(self, backend_name: str, backend_config: dict) -> BaseLLMBackend:
        if backend_name == "mock":
            return MockLLMBackend(seed=self.seed + 1000)
        elif backend_name == "openai":
            return self._build_openai_backend(backend_config)
        else:
            raise ValueError(
                f"Unsupported backend: {backend_name}. "
                f"Available: mock, openai"
            )

    def _build_openai_backend(self, backend_config: dict) -> BaseLLMBackend:
        try:
            from src.backends.openai_backend import OpenAIChatBackend
        except ImportError:
            raise ImportError(
                "openai package is required for the openai backend. "
                "Install it with: pip install openai"
            )

        import os

        model = str(backend_config.get("model", "gpt-4o-mini"))
        api_key_env = str(backend_config.get("api_key_env", "OPENAI_API_KEY"))
        base_url = str(backend_config.get("base_url", "https://api.openai.com/v1"))
        max_retries = int(backend_config.get("max_retries", 2))
        timeout = float(backend_config.get("timeout", 30.0))

        api_key = os.environ.get(api_key_env, "")
        if not api_key:
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(
                f"Environment variable '{api_key_env}' not set. "
                f"API calls will fail."
            )

        return OpenAIChatBackend(
            model=model,
            api_key=api_key,
            base_url=base_url,
            max_retries=max_retries,
            timeout=timeout,
        )

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
    seeds: list[int] | None = None,
) -> list[SimulationResult | MultiSeedResult]:
    results: list[SimulationResult | MultiSeedResult] = []
    for condition in conditions:
        if seeds is not None:
            result = SimulationRunner.run_multi_seed(
                default_path=default_path,
                experiments_path=experiments_path,
                condition=condition,
                experiment_name=experiment_name,
                seeds=seeds,
            )
        else:
            runner = SimulationRunner.from_config_files(
                default_path=default_path,
                experiments_path=experiments_path,
                condition=condition,
                experiment_name=experiment_name,
            )
            result = runner.run()
        results.append(result)
    return results
