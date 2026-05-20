from src.agents.reputation import ReputationModel


def test_cooperation_increases_reputation_slowly() -> None:
    model = ReputationModel(alpha=0.8)
    updated = model.update(0.5, "cooperate")
    assert 0.5 < updated < 0.6


def test_defection_decreases_reputation_faster() -> None:
    model = ReputationModel(alpha=0.8)
    updated = model.update(0.5, "defect")
    assert updated < 0.4


def test_reputation_remains_bounded() -> None:
    model = ReputationModel(alpha=0.8)
    reputation = 0.5
    for _ in range(50):
        reputation = model.update(reputation, "cooperate")
    assert 0.0 <= reputation <= 1.0

    for _ in range(50):
        reputation = model.update(reputation, "defect")
    assert 0.0 <= reputation <= 1.0
