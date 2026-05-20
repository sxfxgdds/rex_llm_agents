from src.games.prisoner_dilemma import PrisonerDilemmaGame


def test_prisoner_dilemma_payoff_matrix() -> None:
    game = PrisonerDilemmaGame()
    assert game.payoff("cooperate", "cooperate") == (3.0, 3.0)
    assert game.payoff("cooperate", "defect") == (0.0, 5.0)
    assert game.payoff("defect", "cooperate") == (5.0, 0.0)
    assert game.payoff("defect", "defect") == (1.0, 1.0)
