"""Round-level metrics for REx simulations."""

from __future__ import annotations

import math
from collections import Counter
from statistics import fmean


def cooperation_rate(actions: list[str]) -> float:
    if not actions:
        return 0.0
    return actions.count("cooperate") / len(actions)


def defection_rate(actions: list[str]) -> float:
    if not actions:
        return 0.0
    return actions.count("defect") / len(actions)


def average_payoff(payoffs: list[float]) -> float:
    return fmean(payoffs) if payoffs else 0.0


def social_welfare(payoffs: list[float]) -> float:
    return float(sum(payoffs))


def average_reputation(reputations: list[float]) -> float:
    return fmean(reputations) if reputations else 0.0


def reputation_variance(reputations: list[float]) -> float:
    if not reputations:
        return 0.0
    mean = average_reputation(reputations)
    return sum((reputation - mean) ** 2 for reputation in reputations) / len(reputations)


def average_temperature(temperatures: list[float]) -> float:
    return fmean(temperatures) if temperatures else 0.0


def action_entropy(actions: list[str]) -> float:
    if not actions:
        return 0.0
    counts = Counter(actions)
    total = len(actions)
    entropy = 0.0
    for count in counts.values():
        probability = count / total
        entropy -= probability * math.log(probability)
    return entropy
