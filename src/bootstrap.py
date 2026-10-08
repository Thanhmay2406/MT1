from __future__ import annotations

import random
from typing import Sequence

from audit_statistics import METHODS, group_statistics, macro_average, percentile, paired_macro_differences, partial_rank_macro


BOOTSTRAP_SEED = 20260905
BOOTSTRAP_REPLICATES = 10_000


def hierarchical_bootstrap(
    rows: Sequence[dict],
    eligible_image_ids: Sequence[int],
    *,
    methods: Sequence[str] = METHODS,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
    ledger_sink=None,
    progress_callback=None,
) -> dict:
    if replicates < 1 or not eligible_image_ids:
        raise ValueError("Bootstrap requires positive replicates and eligible images")
    if set(methods) != set(METHODS) or len(methods) != 4:
        raise ValueError("Bootstrap requires all four paired methods")
    if len(set(eligible_image_ids)) != len(eligible_image_ids):
        raise ValueError("Original eligible image IDs must be unique")
    grouped = {}
    for row in rows:
        grouped.setdefault(row["group_id"], []).append(row)
    if any(len(group_rows) != 12 for group_rows in grouped.values()):
        raise ValueError("Bootstrap requires 12 sampled channels per group")
    rng = random.Random(seed)
    group_ids = list(grouped)
    values = {method: [] for method in methods}
    kendall = {method: [] for method in methods}
    paired = {method: [] for method in methods if method != "gxa"}
    partial = {method: [] for method in methods if method != "gxa"}
    ledger = [] if ledger_sink is None else ledger_sink
    for row in rows:
        contributions = row.get("importance_by_image")
        if not contributions or any(str(i) not in contributions for i in eligible_image_ids):
            raise ValueError("Paired bootstrap requires per-image importance for every eligible image")
        for i in eligible_image_ids:
            if any(m not in contributions[str(i)] for m in methods if m != "l1"):
                raise ValueError("Incomplete per-image importance")
    for _ in range(replicates):
        sampled_groups = [rng.choice(group_ids) for _ in group_ids]
        channel_draws = [[rng.choice(grouped[g]) for _ in range(12)] for g in sampled_groups]
        sampled_images = [rng.choice(list(eligible_image_ids)) for _ in eligible_image_ids]
        replicate_rows = []
        for draw_index, group_id in enumerate(sampled_groups):
            for channel_slot, source_row in enumerate(channel_draws[draw_index]):
                image_damage = source_row["damage_by_image"]
                damage = sum(float(image_damage[str(image_id)]) for image_id in sampled_images) / len(sampled_images)
                scores = {m: dict(source_row["scores"][m]) for m in methods}
                for m in methods:
                    if m != "l1":
                        scores[m]["raw"] = sum(float(source_row["importance_by_image"][str(i)][m]) for i in sampled_images) / len(sampled_images)
                replicate_rows.append({**source_row, "group_id": f"bootstrap_slot={draw_index}",
                                       "original_group_id": group_id, "bootstrap_group_slot": draw_index,
                                       "bootstrap_channel_slot": channel_slot, "damage": damage, "scores": scores})
        stats = group_statistics(replicate_rows, methods)
        replicate_stats = {"spearman": {}, "kendall": {}, "paired_differences": paired_macro_differences(stats, methods),
                           "partial_rank": {m: partial_rank_macro(replicate_rows, m) for m in partial}}
        for method in methods:
            for statistic, destination in (("spearman", values), ("kendall", kendall)):
                result = macro_average(stats, statistic, method)
                destination[method].append(result["value"])
                replicate_stats[statistic][method] = result
        for method in partial:
            paired[method].append(replicate_stats["paired_differences"][method]["value"])
            partial[method].append(replicate_stats["partial_rank"][method]["value"])
        ledger.append({"replicate": len(ledger), "image_ids": sampled_images, "original_group_ids": sampled_groups,
                       "channel_ids": [[r["canonical_id"] for r in draw] for draw in channel_draws], "statistics": replicate_stats})
        if progress_callback is not None:
            progress_callback(len(ledger), replicates)
    observed_groups = group_statistics(rows, methods)
    def intervals(collection, observed):
        result = {}
        for method, draws in collection.items():
            finite = sum(v is not None for v in draws)
            valid = finite == replicates and observed[method] is not None
            result[method] = {"lower": percentile(draws, .025) if valid else None,
                              "upper": percentile(draws, .975) if valid else None,
                              "mean": sum(draws) / replicates if valid else None,
                              "status": "ok" if valid else ("undefined_replicates_present" if finite < replicates else "inference_unavailable"),
                              "reason": None if valid else ("undefined_replicates_present" if finite < replicates else "observed_statistic_undefined"),
                              "observed": observed[method], "finite_replicates": finite, "undefined_replicates": replicates - finite}
        return result
    return {
        "seed": seed,
        "replicates": replicates,
        "confidence_level": 0.95,
        "statistics": intervals(values, {m: macro_average(observed_groups, "spearman", m)["value"] for m in methods}),
        "kendall": intervals(kendall, {m: macro_average(observed_groups, "kendall", m)["value"] for m in methods}),
        "paired_differences": intervals(paired, {m: r["value"] for m, r in paired_macro_differences(observed_groups, methods).items()}),
        "partial_rank": intervals(partial, {m: partial_rank_macro(rows, m)["value"] for m in partial}),
        "ledger": ledger,
    }
