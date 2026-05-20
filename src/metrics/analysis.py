"""Simple post-hoc analysis helpers for intervention runs."""

from __future__ import annotations

from typing import Any, Iterable


def _as_records(round_df: Any) -> list[dict[str, Any]]:
    if hasattr(round_df, "to_dict"):
        return list(round_df.to_dict("records"))
    return list(round_df)


def _float(record: dict[str, Any], key: str) -> float:
    return float(record.get(key, 0.0))


def _int(record: dict[str, Any], key: str) -> int:
    return int(float(record.get(key, 0)))


def _pre_attack_value(records: list[dict[str, Any]], attack_round: int, key: str) -> float | None:
    before = [record for record in records if _int(record, "round") < attack_round]
    if not before:
        return None
    return _float(before[-1], key)


def _recovery_time(records: Iterable[dict[str, Any]], attack_round: int, key: str) -> int | None:
    ordered = sorted(_as_records(records), key=lambda record: _int(record, "round"))
    baseline = _pre_attack_value(ordered, attack_round, key)
    if baseline is None:
        return None
    target = 0.95 * baseline
    for record in ordered:
        round_number = _int(record, "round")
        if round_number >= attack_round and _float(record, key) >= target:
            return round_number - attack_round
    return None


def reputation_recovery_time(round_df: Any, attack_round: int) -> int | None:
    return _recovery_time(round_df, attack_round, "avg_reputation")


def cooperation_recovery_time(round_df: Any, attack_round: int) -> int | None:
    return _recovery_time(round_df, attack_round, "cooperation_rate")


def welfare_drop(round_df: Any, attack_round: int) -> float:
    records = sorted(_as_records(round_df), key=lambda record: _int(record, "round"))
    baseline = _pre_attack_value(records, attack_round, "social_welfare")
    if baseline is None:
        return 0.0
    post_attack = [
        _float(record, "social_welfare")
        for record in records
        if _int(record, "round") >= attack_round
    ]
    if not post_attack:
        return 0.0
    return max(0.0, baseline - min(post_attack))


def betrayal_cascade_size(round_df: Any, threshold: float = 0.6) -> int:
    records = _as_records(round_df)
    return sum(1 for record in records if _float(record, "defection_rate") >= threshold)
