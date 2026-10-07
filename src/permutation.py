from __future__ import annotations

import random
from typing import Sequence

from audit_statistics import METHODS, partial_rank_macro


PERMUTATION_SEED = 20260905
PERMUTATION_REPLICATES = 100_000
BASELINES = ("l1", "taylor", "activation")


def holm_correction(p_values: dict[str, float | None], alpha: float = 0.05) -> dict[str, dict]:
    if any(value is None for value in p_values.values()):
        return {name: {"raw_p": value, "holm_p": None, "reject": False, "status": "family_incomplete"} for name, value in p_values.items()}
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
    if set(baselines) != set(BASELINES) or len(baselines) != 3:
        raise ValueError("Permutation inference must retain the complete H2-H4 family")
    grouped = {}
    for row in rows:
        grouped.setdefault(row["group_id"], []).append(row)
    observed = {baseline: partial_rank_macro(rows, baseline)["value"] for baseline in baselines}
    exceedances = {baseline: 0 for baseline in baselines}
    undefined = {baseline: 0 for baseline in baselines}
    ledger = []
    rng = random.Random(seed)
    for _ in range(replicates):
        permuted_rows = []
        for group_rows in grouped.values():
            damages = [float(row["damage"]) for row in group_rows]
            rng.shuffle(damages)
            permuted_rows.extend({**row, "damage": damage} for row, damage in zip(group_rows, damages))
        results = {b: partial_rank_macro(permuted_rows, b) for b in baselines}
        ledger.append(results)
        for baseline in baselines:
            value = results[baseline]["value"]
            if value is None:
                undefined[baseline] += 1
            elif observed[baseline] is not None and value >= observed[baseline]:
                exceedances[baseline] += 1
    raw_p = {baseline: (1 + exceedances[baseline]) / (replicates + 1) if observed[baseline] is not None and not undefined[baseline] else None for baseline in baselines}
    result = holm_correction(raw_p)
    for baseline in baselines:
        result[baseline].update({"observed": observed[baseline], "exceedances": exceedances[baseline], "undefined_permutations": undefined[baseline],
                                 "inference_status": "ok" if raw_p[baseline] is not None else "inference_unavailable"})
    return {"seed": seed, "replicates": replicates, "plus_one": True, "alpha": 0.05, "statistics": result,
            "family_status": "complete" if all(p is not None for p in raw_p.values()) else "incomplete", "ledger": ledger}
