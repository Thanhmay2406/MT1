from __future__ import annotations

from collections.abc import Sequence


def _match_value(match: object, key: str):
    if isinstance(match, dict):
        return match[key]
    return getattr(match, key)


def compute_object_damage(original_matches: Sequence[object], intervened_matches: Sequence[object]) -> list[dict]:
    original: dict[int, float] = {}
    for match in original_matches:
        gt_id = int(_match_value(match, "gt_id"))
        if gt_id in original:
            raise ValueError(f"duplicate original GT identity: {gt_id}")
        score = _match_value(match, "prediction_score")
        original[gt_id] = float(score)
    intervened: dict[int, float] = {}
    for match in intervened_matches:
        gt_id = int(_match_value(match, "gt_id"))
        if gt_id in intervened:
            raise ValueError(f"duplicate intervened GT identity: {gt_id}")
        intervened[gt_id] = float(_match_value(match, "prediction_score"))
    rows = []
    for gt_id, original_utility in original.items():
        intervened_utility = intervened.get(gt_id, 0.0)
        rows.append(
            {
                "gt_id": gt_id,
                "original_utility": original_utility,
                "intervened_utility": intervened_utility,
                "damage": original_utility - intervened_utility,
            }
        )
    return rows


def aggregate_image_damage(object_damages: Sequence[dict]) -> float:
    if not object_damages:
        return 0.0
    return sum(float(row["damage"]) for row in object_damages) / len(object_damages)


def aggregate_channel_damage(image_damages: Sequence[float]) -> float:
    if not image_damages:
        raise ValueError("Cannot aggregate an empty channel damage set")
    return sum(float(value) for value in image_damages) / len(image_damages)
