from __future__ import annotations

import random
from typing import Sequence

from statistics import METHODS, group_statistics, macro_average, percentile


BOOTSTRAP_SEED = 20260905
BOOTSTRAP_REPLICATES = 10_000


def hierarchical_bootstrap(
    rows: Sequence[dict],
    eligible_image_ids: Sequence[int],
    *,
    methods: Sequence[str] = METHODS,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    if replicates < 1 or not eligible_image_ids:
        raise ValueError("Bootstrap requires positive replicates and eligible images")
    grouped = {}
    for row in rows:
        grouped.setdefault(row["group_id"], []).append(row)
    if any(len(group_rows) != 12 for group_rows in grouped.values()):
        raise ValueError("Bootstrap requires 12 sampled channels per group")
    rng = random.Random(seed)
    group_ids = list(grouped)
    values = {method: [] for method in methods}
    for _ in range(replicates):
        sampled_images = [rng.choice(list(eligible_image_ids)) for _ in eligible_image_ids]
        sampled_groups = [rng.choice(group_ids) for _ in group_ids]
        replicate_rows = []
        for draw_index, group_id in enumerate(sampled_groups):
            for source_row in (rng.choice(grouped[group_id]) for _ in range(12)):
                image_damage = source_row["damage_by_image"]
                damage = sum(float(image_damage[str(image_id)]) for image_id in sampled_images) / len(sampled_images)
                replicate_rows.append({**source_row, "group_id": f"{group_id}#bootstrap_draw={draw_index}", "damage": damage})
        stats = group_statistics(replicate_rows, methods)
        for method in methods:
            values[method].append(macro_average(stats, "spearman", method)["value"])
    return {
        "seed": seed,
        "replicates": replicates,
        "confidence_level": 0.95,
        "statistics": {
            method: {"lower": percentile(method_values, 0.025), "upper": percentile(method_values, 0.975), "mean": sum(method_values) / len(method_values)}
            for method, method_values in values.items()
        },
    }
