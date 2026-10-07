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


def correspondence_diagnostics(original_matches, intervened_matches):
    original = {int(_match_value(m, "gt_id")): m for m in original_matches}
    intervened = {int(_match_value(m, "gt_id")): m for m in intervened_matches}
    count = sum(gt not in intervened for gt in original)
    localization = [{"gt_id": gt, "original_iou": float(_match_value(m, "iou")),
                     "intervened_iou": float(_match_value(intervened[gt], "iou")) if gt in intervened else 0.,
                     "localization_damage": float(_match_value(m, "iou")) - (float(_match_value(intervened[gt], "iou")) if gt in intervened else 0.)}
                    for gt, m in original.items()]
    return {"class_flip": {"value": None, "status": "not_identifiable",
                           "reason": "cross_forward_detection_identity_not_defined", "scope": "originally_eligible_gt", "primary_damage_unchanged": True},
            "same_class_match_failure_count": count, "evaluated_originally_eligible_count": len(original),
            "same_class_match_failure_rate": count / len(original) if original else None,
            "localization": localization, "full_diagnostic_completion": False}
