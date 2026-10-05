from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable, Sequence


METHODS = ("gxa", "activation", "l1", "taylor")


def average_ranks(values: Sequence[float]) -> list[float]:
    ordered = sorted((float(value), index) for index, value in enumerate(values))
    ranks = [0.0] * len(values)
    position = 0
    while position < len(ordered):
        end = position + 1
        while end < len(ordered) and ordered[end][0] == ordered[position][0]:
            end += 1
        rank = (position + 1 + end) / 2.0
        for _, index in ordered[position:end]:
            ranks[index] = rank
        position = end
    return ranks


def _pearson(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) != len(y) or not x:
        raise ValueError("Correlation vectors must have equal non-zero length")
    x_mean = sum(x) / len(x)
    y_mean = sum(y) / len(y)
    numerator = sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y))
    x_norm = math.sqrt(sum((a - x_mean) ** 2 for a in x))
    y_norm = math.sqrt(sum((b - y_mean) ** 2 for b in y))
    if x_norm == 0.0 or y_norm == 0.0:
        return 0.0
    return numerator / (x_norm * y_norm)


def spearman_tie_aware(x: Sequence[float], y: Sequence[float]) -> dict:
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("Spearman requires equal vectors with at least two values")
    rx = average_ranks(x)
    ry = average_ranks(y)
    non_identifiable = len(set(rx)) == 1 or len(set(ry)) == 1
    return {"value": 0.0 if non_identifiable else _pearson(rx, ry), "non_identifiable": non_identifiable}


def kendall_tau_b(x: Sequence[float], y: Sequence[float]) -> dict:
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("Kendall tau-b requires equal vectors with at least two values")
    concordant = discordant = ties_x = ties_y = ties_both = 0
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            dx = (float(x[i]) > float(x[j])) - (float(x[i]) < float(x[j]))
            dy = (float(y[i]) > float(y[j])) - (float(y[i]) < float(y[j]))
            if dx == 0 and dy == 0:
                ties_both += 1
            elif dx == 0:
                ties_x += 1
            elif dy == 0:
                ties_y += 1
            elif dx == dy:
                concordant += 1
            else:
                discordant += 1
    denominator = math.sqrt((concordant + discordant + ties_x) * (concordant + discordant + ties_y))
    value = 0.0 if denominator == 0.0 else (concordant - discordant) / denominator
    return {"value": value, "concordant": concordant, "discordant": discordant, "ties_x": ties_x, "ties_y": ties_y, "ties_both": ties_both}


def _grouped(rows: Sequence[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["group_id"])].append(row)
    return dict(grouped)


def group_statistics(rows: Sequence[dict], methods: Sequence[str] = METHODS) -> dict[str, dict]:
    result = {}
    for group_id, group_rows in _grouped(rows).items():
        method_stats = {}
        damage = [float(row["damage"]) for row in group_rows]
        for method in methods:
            importance = [float(row["scores"][method]["raw"]) for row in group_rows]
            method_stats[method] = {
                "spearman": spearman_tie_aware(importance, damage),
                "kendall": kendall_tau_b(importance, damage),
            }
        result[group_id] = {"group_id": group_id, "channel_count": len(group_rows), "methods": method_stats}
    return result


def macro_average(group_stats: dict[str, dict], statistic: str, method: str) -> dict:
    values = [float(group["methods"][method][statistic]["value"]) for group in group_stats.values()]
    flags = [bool(group["methods"][method][statistic].get("non_identifiable", False)) for group in group_stats.values()]
    return {"value": sum(values) / len(values) if values else 0.0, "group_count": len(values), "non_identifiable_groups": sum(flags)}


def paired_macro_differences(group_stats: dict[str, dict], methods: Sequence[str] = METHODS) -> dict[str, dict]:
    primary = float(macro_average(group_stats, "spearman", "gxa")["value"])
    return {method: {"value": primary - float(macro_average(group_stats, "spearman", method)["value"])} for method in methods if method != "gxa"}


def _residuals_by_group(rows: Sequence[dict], predictor: str, outcome: str) -> list[tuple[list[float], list[float]]]:
    residual_groups = []
    for group_rows in _grouped(rows).values():
        baseline = average_ranks([float(row["scores"][predictor]["raw"]) for row in group_rows])
        values = average_ranks([float(row[outcome]) for row in group_rows])
        baseline_mean = sum(baseline) / len(baseline)
        value_mean = sum(values) / len(values)
        centered = [value - value_mean for value in values]
        baseline_centered = [value - baseline_mean for value in baseline]
        denominator = sum(value * value for value in baseline_centered)
        slope = 0.0 if denominator == 0.0 else sum(a * b for a, b in zip(baseline_centered, centered)) / denominator
        residual_groups.append((
            [value - slope * base for value, base in zip(centered, baseline_centered)],
            [value - slope * base for value, base in zip(centered, baseline_centered)],
        ))
    return residual_groups


def partial_rank_macro(rows: Sequence[dict], baseline: str) -> dict:
    grouped = _grouped(rows)
    values = []
    non_identifiable = 0
    for group_rows in grouped.values():
        gxa = average_ranks([float(row["scores"]["gxa"]["raw"]) for row in group_rows])
        damage = average_ranks([float(row["damage"]) for row in group_rows])
        control = average_ranks([float(row["scores"][baseline]["raw"]) for row in group_rows])
        control_mean = sum(control) / len(control)
        gxa_mean = sum(gxa) / len(gxa)
        damage_mean = sum(damage) / len(damage)
        control_centered = [value - control_mean for value in control]
        gxa_centered = [value - gxa_mean for value in gxa]
        damage_centered = [value - damage_mean for value in damage]
        denominator = sum(value * value for value in control_centered)
        beta_gxa = 0.0 if denominator == 0.0 else sum(a * b for a, b in zip(control_centered, gxa_centered)) / denominator
        beta_damage = 0.0 if denominator == 0.0 else sum(a * b for a, b in zip(control_centered, damage_centered)) / denominator
        result = _pearson(
            [value - beta_gxa * control for value, control in zip(gxa_centered, control_centered)],
            [value - beta_damage * control for value, control in zip(damage_centered, control_centered)],
        )
        if denominator == 0.0:
            non_identifiable += 1
        values.append(result)
    return {"value": sum(values) / len(values) if values else 0.0, "group_count": len(values), "non_identifiable_groups": non_identifiable}


def percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("Cannot calculate percentile of empty values")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight
