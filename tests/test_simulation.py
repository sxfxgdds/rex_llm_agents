import csv

from src.simulation.runner import SimulationRunner


def test_small_simulation_runs_and_writes_csv(tmp_path) -> None:
    config = {
        "num_agents": 6,
        "num_rounds": 5,
        "game": "prisoner_dilemma",
        "backend": "mock",
        "seed": 123,
        "initial_reputation": 0.5,
        "exploration": {
            "mode": "piecewise",
            "fixed_temperature": 0.5,
            "t_min": 0.2,
            "t_max": 0.8,
        },
        "reputation": {
            "alpha": 0.8,
            "defect_penalty_scale": 1.5,
            "cooperate_recovery_scale": 0.7,
        },
        "malicious": {
            "fraction": 0.0,
            "warmup_rounds": 20,
            "exploit_threshold": 0.7,
        },
        "attack": {
            "enabled": False,
            "attack_round": 50,
            "target_fraction": 0.2,
            "reputation_drop": 0.5,
        },
        "output_dir": str(tmp_path),
    }
    result = SimulationRunner(
        config=config,
        experiment_name="test_experiment",
        condition="test_condition",
    ).run()

    assert result.agent_csv.exists()
    assert result.round_csv.exists()

    with result.round_csv.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 5
    assert "cooperation_rate" in rows[0]
    assert "avg_reputation" in rows[0]
