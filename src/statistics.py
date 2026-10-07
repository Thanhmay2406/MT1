from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable, Sequence


METHODS = ("gxa", "activation", "l1", "taylor")


def average_ranks(values: Sequence[float]) -> list[float]:
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("Rank inputs must be finite")
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
    if not all(math.isfinite(float(v)) for v in (*x, *y)):
        raise ValueError("Ordering inputs must be finite")
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
    value = None if denominator == 0.0 else (concordant - discordant) / denominator
    return {"value": value, "status": "not_identifiable" if value is None else "ok", "concordant": concordant, "discordant": discordant, "ties_x": ties_x, "ties_y": ties_y, "ties_both": ties_both}


def pairwise_ordering_agreement(x: Sequence[float], y: Sequence[float]) -> dict:
    counts = kendall_tau_b(x, y)
    comparable = counts["concordant"] + counts["discordant"]
    return {**{k: v for k, v in counts.items() if k not in ("value", "status")},
            "total_pairs": len(x) * (len(x) - 1) // 2, "comparable_pairs": comparable,
            "value": counts["concordant"] / comparable if comparable else None,
            "status": "ok" if comparable else "no_comparable_pairs"}


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
                "ordering_agreement": pairwise_ordering_agreement(importance, damage),
            }
        result[group_id] = {"group_id": group_id, "channel_count": len(group_rows), "methods": method_stats,
                            "stage": group_rows[0].get("stage"), "hidden_conv": group_rows[0].get("hidden_conv")}
    return result


def macro_average(group_stats: dict[str, dict], statistic: str, method: str) -> dict:
    values = [group["methods"][method][statistic]["value"] for group in group_stats.values()]
    flags = [bool(group["methods"][method][statistic].get("non_identifiable", False)) for group in group_stats.values()]
    valid = bool(values) and all(value is not None for value in values)
    return {"value": sum(values) / len(values) if valid else None, "status": "ok" if valid else "not_identifiable", "group_count": len(values), "non_identifiable_groups": sum(flags), "undefined_groups": sum(value is None for value in values)}


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
    import numpy as np

    grouped = _grouped(rows)
    if not grouped or any(len(group) < 2 for group in grouped.values()):
        raise ValueError("Partial rank requires at least two observations per group")
    q = len(grouped)
    vectors, weights, indicators = [], [], []
    for slot, group_rows in enumerate(grouped.values()):
        n = len(group_rows)
        vectors.extend(zip(*[(np.asarray(average_ranks([r["damage"] if m == "damage" else r["scores"][m]["raw"] for r in group_rows]), dtype=np.float64) - 1) / (n - 1)
                             for m in ("gxa", "damage", baseline)]))
        weights.extend([1 / (q * n)] * n)
        indicators.extend([slot] * n)
    z = np.asarray(vectors, dtype=np.float64)
    w = np.asarray(weights)
    root = np.sqrt(w)
    X = np.column_stack((np.eye(q)[indicators], z[:, 2]))
    A = root[:, None] * X
    U, s, _ = np.linalg.svd(A, full_matrices=False)
    n, p = A.shape
    eps = np.finfo(np.float64).eps
    rcond = max(n, p) * eps
    cutoff = rcond * s[0]
    rank = int(np.count_nonzero(s > cutoff))
    v = root[:, None] * z[:, :2]
    residual = (v - U[:, :rank] @ (U[:, :rank].T @ v)) / root[:, None]
    residual -= np.sum(w[:, None] * residual, axis=0)
    norms = np.linalg.norm(root[:, None] * residual, axis=0)
    thresholds = 64 * eps * max(n, p) * np.maximum(1, np.linalg.norm(v, axis=0))
    if not np.isfinite(residual).all():
        raise ValueError("Non-finite weighted projection")
    metadata = {"group_count": q, "numeric_convention": "weighted_svd_float64_v1", "design_rank": rank,
                "design_n_columns": p, "singular_values": s.tolist(), "svd_rcond": rcond,
                "svd_cutoff": float(cutoff), "rank_deficient": rank < p,
                "baseline_redundant_after_group_effects": rank == q,
                "residual_norms": norms.tolist(), "residual_thresholds": thresholds.tolist(),
                "convention_applied": False}
    for index, name in enumerate(("gxa", "damage")):
        if norms[index] <= thresholds[index]:
            return {**metadata, "value": None, "status": "non_identifiable", "reason": f"{name}_residual_zero_or_numerically_zero"}
    value = float(np.sum(w * residual[:, 0] * residual[:, 1]) / np.prod(norms))
    return {**metadata, "value": value, "status": "ok"}


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
