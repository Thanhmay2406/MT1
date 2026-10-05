from __future__ import annotations

import random
from typing import Sequence

from statistics import METHODS, partial_rank_macro


PERMUTATION_SEED = 20260905
PERMUTATION_REPLICATES = 100_000
BASELINES = ("l1", "taylor", "activation")


def holm_correction(p_values: dict[str, float], alpha: float = 0.05) -> dict[str, dict]:
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    adjusted = {}
    running = 0.0
    for index, (name, value) in enumerate(ordered):
        corrected = min(1.0, (len(ordered) - index) * value)
        running = max(running, corrected)
        adjusted[name] = {"raw_p": value, "holm_p": running, "reject": running <= alpha}
    return adjusted


def permutation_tests(
    rows: Sequence[dict],
    *,
    baselines: Sequence[str] = BASELINES,
    replicates: int = PERMUTATION_REPLICATES,
    seed: int = PERMUTATION_SEED,
) -> dict:
    if replicates < 1:
        raise ValueError("Permutation replicates must be positive")
    grouped = {}
    for row in rows:
        grouped.setdefault(row["group_id"], []).append(row)
    observed = {baseline: partial_rank_macro(rows, baseline)["value"] for baseline in baselines}
    exceedances = {baseline: 0 for baseline in baselines}
    rng = random.Random(seed)
    for _ in range(replicates):
        permuted_rows = []
        for group_rows in grouped.values():
            damages = [float(row["damage"]) for row in group_rows]
            rng.shuffle(damages)
            permuted_rows.extend({**row, "damage": damage} for row, damage in zip(group_rows, damages))
        for baseline in baselines:
            if partial_rank_macro(permuted_rows, baseline)["value"] >= observed[baseline]:
                exceedances[baseline] += 1
    raw_p = {baseline: (1 + exceedances[baseline]) / (replicates + 1) for baseline in baselines}
    result = holm_correction(raw_p)
    for baseline in baselines:
        result[baseline].update({"observed": observed[baseline], "exceedances": exceedances[baseline]})
    return {"seed": seed, "replicates": replicates, "plus_one": True, "alpha": 0.05, "statistics": result}
