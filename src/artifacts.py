from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any


E2_SCHEMA_VERSION = "causal_audit_e2_importance/v2"
E3_SCHEMA_VERSION = "causal_audit_e3_damage/v2"
E4_SCHEMA_VERSION = "causal_audit_e4_equivalence/v2"
E5_SCHEMA_VERSION = "causal_audit_e5_statistics/v2"
E6_SCHEMA_VERSION = "causal_audit_e6_reproducibility/v2"
E7_SCHEMA_VERSION = "causal_audit_e7_final_report/v1"
_LEAF_ENCODER = json.JSONEncoder(allow_nan=False)


class DiskLedger:
    """Temporary, sequential JSON records; never a resumable artifact."""

    def __init__(self, directory: str | Path):
        Path(directory).mkdir(parents=True, exist_ok=True)
        self._file = tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=directory, prefix=".mt1-ledger-", delete=False
        )
        self.path = Path(self._file.name)
        self._count = 0

    def append(self, record: Any) -> None:
        self._file.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        self._count += 1

    def __len__(self) -> int:
        return self._count

    def __iter__(self):
        self._file.flush()
        with self.path.open(encoding="utf-8") as stream:
            for line in stream:
                yield json.loads(line)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        try:
            self._file.close()
        finally:
            self.path.unlink(missing_ok=True)


def _json_chunks(value: Any, level: int = 0):
    """Pretty-print without building a whole-payload string or disk ledger list."""
    pad = "  " * level
    child_pad = "  " * (level + 1)
    if isinstance(value, dict):
        if not value:
            yield "{}"
            return
        yield "{\n"
        for index, key in enumerate(sorted(value)):
            if index:
                yield ",\n"
            if isinstance(key, str):
                encoded_key = _LEAF_ENCODER.encode(key)
            elif key is None or isinstance(key, (bool, int, float)):
                encoded_key = _LEAF_ENCODER.encode(_LEAF_ENCODER.encode(key))
            else:
                raise TypeError(f"JSON keys must be str, int, float, bool or None: {type(key).__name__}")
            yield child_pad + encoded_key + ": "
            yield from _json_chunks(value[key], level + 1)
        yield "\n" + pad + "}"
    elif isinstance(value, (list, tuple, DiskLedger)):
        if not len(value):
            yield "[]"
            return
        yield "[\n"
        for index, item in enumerate(value):
            if index:
                yield ",\n"
            yield child_pad
            yield from _json_chunks(item, level + 1)
        yield "\n" + pad + "]"
    else:
        yield _LEAF_ENCODER.encode(value)


def write_json_artifact(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(payload, "to_dict"):
        payload = payload.to_dict()
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=target.parent, delete=False) as handle:
            temporary = Path(handle.name)
            for chunk in _json_chunks(payload):
                handle.write(chunk)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def read_json_artifact(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def percentile_ranks_by_group(rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(str(row["group_id"]), []).append(row)
    output = [dict(row) for row in rows]
    positions = {id(row): index for index, row in enumerate(rows)}
    for group_rows in grouped.values():
        ordered = sorted(group_rows, key=lambda row: float(row["raw"]))
        index = 0
        while index < len(ordered):
            end = index + 1
            while end < len(ordered) and float(ordered[end]["raw"]) == float(ordered[index]["raw"]):
                end += 1
            average_rank = (index + 1 + end) / 2.0
            percentile = 0.0 if len(ordered) == 1 else (average_rank - 1.0) / (len(ordered) - 1.0)
            for tied in ordered[index:end]:
                output[positions[id(tied)]] = {**tied, "percentile": percentile}
            index = end
    return output


def validate_e2_contributions(channels, eligible_image_count):
    image_ids = None
    for row in channels:
        contributions = row.get("importance_by_image")
        if not isinstance(contributions, dict) or len(contributions) != eligible_image_count:
            raise ValueError("E2 requires complete per-image importance contributions")
        if image_ids is None:
            image_ids = set(contributions)
        if set(contributions) != image_ids:
            raise ValueError("E2 channels must share the same eligible image domain")
        for values in contributions.values():
            if set(values) != {"gxa", "activation", "taylor"} or not all(math.isfinite(float(v)) for v in values.values()):
                raise ValueError("E2 per-image contributions must contain three finite methods")


def build_e2_artifact(
    *,
    checkpoint_sha256: str,
    probe_sha256: str,
    eligibility_sha256: str,
    matching_iou_threshold: float,
    groups: list[dict],
    channels: list[dict],
    image_count: int,
    eligible_image_count: int,
    eligible_instance_count: int,
    determinism: dict | None = None,
) -> dict:
    if len(channels) != sum(int(group["channels"]) for group in groups):
        raise ValueError("Channel rows do not match structural group widths")
    canonical_ids = [row["canonical_id"] for row in channels]
    if len(set(canonical_ids)) != len(canonical_ids):
        raise ValueError("E2 canonical channel IDs must be unique")
    methods = ("gxa", "activation", "l1", "taylor")
    for row in channels:
        if set(row.get("scores", {})) != set(methods):
            raise ValueError("Each channel must contain all E2 score families")
        for method in methods:
            for key in ("raw", "percentile"):
                value = float(row["scores"][method][key])
                if value != value or value in (float("inf"), float("-inf")):
                    raise ValueError("E2 scores must be finite")
    validate_e2_contributions(channels, eligible_image_count)
    artifact = {
        "schema_version": E2_SCHEMA_VERSION,
        "checkpoint_sha256": checkpoint_sha256,
        "probe_sha256": probe_sha256,
        "eligibility_sha256": eligibility_sha256,
        "matching_iou_threshold": float(matching_iou_threshold),
        "group_count": len(groups),
        "channel_count": len(channels),
        "image_count": int(image_count),
        "eligible_image_count": int(eligible_image_count),
        "eligible_instance_count": int(eligible_instance_count),
        "groups": groups,
        "channels": channels,
    }
    if determinism is not None:
        artifact["determinism"] = determinism
    artifact["implementation_contract"] = "mt1_clarifications_implementation_guide:Q01-Q05"
    return artifact


def read_e2_artifact(path: str | Path) -> dict:
    payload = read_json_artifact(path)
    if payload.get("schema_version") not in (E2_SCHEMA_VERSION, E2_SCHEMA_VERSION.replace("/v2", "/v1")):
        raise ValueError("Unsupported E2 artifact schema")
    if payload["schema_version"] == E2_SCHEMA_VERSION:
        validate_e2_contributions(payload["channels"], payload["eligible_image_count"])
    return payload


def build_e3_artifact(
    *,
    checkpoint_sha256: str,
    probe_sha256: str,
    eligibility_sha256: str,
    importance_sha256: str,
    dataset_manifest_sha256: str,
    channel_manifest_sha256: str,
    sample_id: str,
    sample_identity_sha256: str,
    matching_iou_threshold: float,
    channel_ids: list[str],
    image_count: int,
    pairs: list[dict],
) -> dict:
    expected_pair_count = len(channel_ids) * int(image_count)
    pair_keys = [(row.get("canonical_id"), int(row.get("image_id"))) for row in pairs]
    if len(set(pair_keys)) != len(pair_keys):
        raise ValueError("E3 artifact contains duplicate channel-image pairs")
    if len(set(channel_ids)) != len(channel_ids):
        raise ValueError("E3 channel IDs must be unique")
    allowed = set(channel_ids)
    if any(key[0] not in allowed for key in pair_keys):
        raise ValueError("E3 pair references an undeclared channel")
    return {
        "schema_version": E3_SCHEMA_VERSION,
        "checkpoint_sha256": checkpoint_sha256,
        "probe_sha256": probe_sha256,
        "eligibility_sha256": eligibility_sha256,
        "importance_sha256": importance_sha256,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "channel_manifest_sha256": channel_manifest_sha256,
        "sample_id": sample_id,
        "sample_identity_sha256": sample_identity_sha256,
        "matching_iou_threshold": float(matching_iou_threshold),
        "channel_count": len(channel_ids),
        "image_count": int(image_count),
        "expected_pair_count": expected_pair_count,
        "completed_pair_count": len(pairs),
        "ok_pair_count": sum(row.get("status") == "ok" for row in pairs),
        "error_pair_count": sum(row.get("status") == "error" for row in pairs),
        "complete": len(pairs) == expected_pair_count and all(row.get("status") == "ok" for row in pairs),
        "channel_ids": channel_ids,
        "pairs": pairs,
    }


def read_e3_artifact(path: str | Path) -> dict:
    payload = read_json_artifact(path)
    if payload.get("schema_version") not in (E3_SCHEMA_VERSION, E3_SCHEMA_VERSION.replace("/v2", "/v1")):
        raise ValueError("Unsupported E3 artifact schema")
    return payload


def build_e4_artifact(
    *,
    checkpoint_sha256: str,
    probe_sha256: str,
    eligibility_sha256: str,
    importance_sha256: str,
    intervention_sha256: str,
    dataset_manifest_sha256: str,
    channel_manifest_sha256: str,
    e4_sample_sha256: str,
    sample_id: str,
    sample_identity_sha256: str,
    tolerance: float,
    channel_ids: list[str],
    image_ids: list[int],
    pairs: list[dict],
    dataset_verification: dict | None = None,
) -> dict:
    expected_pair_count = len(channel_ids) * len(image_ids)
    keys = [(row.get("canonical_id"), int(row.get("image_id"))) for row in pairs]
    if len(keys) != len(set(keys)):
        raise ValueError("E4 artifact contains duplicate channel-image pairs")
    if len(set(channel_ids)) != len(channel_ids) or len(set(image_ids)) != len(image_ids):
        raise ValueError("E4 sample identities must be unique")
    if any(channel not in set(channel_ids) for channel, _ in keys) or any(image not in set(image_ids) for _, image in keys):
        raise ValueError("E4 pair references an undeclared sample identity")
    for row in pairs:
        if row.get("status") != "ok":
            continue
        if row.get("full_endpoint_verification") is not True:
            raise ValueError("E4 v2 requires full endpoint verification, not utility-only comparisons")
        for field in ("max_utility_abs_difference", "max_damage_abs_difference"):
            value = float(row.get(field, float("nan")))
            if not math.isfinite(value):
                raise ValueError("E4 comparison values must be finite")
    complete = len(pairs) == expected_pair_count and all(row.get("status") == "ok" for row in pairs)
    equivalent = complete and all(row.get("equivalent") is True for row in pairs)
    return {
        "schema_version": E4_SCHEMA_VERSION,
        "dataset_verification": dataset_verification,
        "checkpoint_sha256": checkpoint_sha256,
        "probe_sha256": probe_sha256,
        "eligibility_sha256": eligibility_sha256,
        "importance_sha256": importance_sha256,
        "intervention_sha256": intervention_sha256,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "channel_manifest_sha256": channel_manifest_sha256,
        "e4_sample_sha256": e4_sample_sha256,
        "sample_id": sample_id,
        "sample_identity_sha256": sample_identity_sha256,
        "tolerance": float(tolerance),
        "channel_count": len(channel_ids),
        "image_count": len(image_ids),
        "expected_pair_count": expected_pair_count,
        "completed_pair_count": len(pairs),
        "equivalent_pair_count": sum(row.get("equivalent") is True for row in pairs),
        "complete": complete,
        "equivalence_status": "equivalent" if equivalent else ("non_equivalent" if complete else "incomplete"),
        "channel_ids": channel_ids,
        "image_ids": image_ids,
        "pairs": pairs,
    }


def read_e4_artifact(path: str | Path) -> dict:
    payload = read_json_artifact(path)
    if payload.get("schema_version") not in (E4_SCHEMA_VERSION, E4_SCHEMA_VERSION.replace("/v2", "/v1")):
        raise ValueError("Unsupported E4 artifact schema")
    return payload


def build_e5_artifact(
    *,
    checkpoint_sha256: str,
    probe_sha256: str,
    eligibility_sha256: str,
    importance_sha256: str,
    intervention_sha256: str,
    equivalence_sha256: str,
    dataset_manifest_sha256: str,
    channel_manifest_sha256: str,
    equivalence_status: str,
    matching_iou_threshold: float,
    eligible_image_count: int,
    eligible_instance_count: int,
    channel_rows: list[dict],
    group_statistics: dict,
    macro_statistics: dict,
    paired_differences: dict,
    hypotheses: dict,
    bootstrap: dict,
    permutation: dict,
) -> dict:
    canonical_ids = [row.get("canonical_id") for row in channel_rows]
    if len(channel_rows) != 384 or len(set(canonical_ids)) != 384:
        raise ValueError("E5 requires 384 unique channel rows")
    if len(group_statistics) != 32:
        raise ValueError("E5 requires 32 structural groups")
    for row in channel_rows:
        for field in ("damage",):
            if not math.isfinite(float(row[field])):
                raise ValueError("E5 damage values must be finite")
        for method in ("gxa", "activation", "l1", "taylor"):
            if not math.isfinite(float(row["scores"][method]["raw"])):
                raise ValueError("E5 importance values must be finite")
    return {
        "schema_version": E5_SCHEMA_VERSION,
        "checkpoint_sha256": checkpoint_sha256,
        "probe_sha256": probe_sha256,
        "eligibility_sha256": eligibility_sha256,
        "importance_sha256": importance_sha256,
        "intervention_sha256": intervention_sha256,
        "equivalence_sha256": equivalence_sha256,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "channel_manifest_sha256": channel_manifest_sha256,
        "equivalence_status": equivalence_status,
        "matching_iou_threshold": float(matching_iou_threshold),
        "group_count": 32,
        "channel_count": 384,
        "eligible_image_count": int(eligible_image_count),
        "eligible_instance_count": int(eligible_instance_count),
        "bootstrap": bootstrap,
        "permutation": permutation,
        "channel_rows": channel_rows,
        "group_statistics": group_statistics,
        "macro_statistics": macro_statistics,
        "paired_differences": paired_differences,
        "hypotheses": hypotheses,
        "complete": True,
        "scientific_completion": False,
    }


def read_e5_artifact(path: str | Path) -> dict:
    payload = read_json_artifact(path)
    if payload.get("schema_version") not in (E5_SCHEMA_VERSION, E5_SCHEMA_VERSION.replace("/v2", "/v1")):
        raise ValueError("Unsupported E5 artifact schema")
    return payload


def read_e6_artifact(path: str | Path) -> dict:
    payload = read_json_artifact(path)
    if payload.get("schema_version") not in (E6_SCHEMA_VERSION, E6_SCHEMA_VERSION.replace("/v2", "/v1")):
        raise ValueError("Unsupported E6 artifact schema")
    return payload


def read_e7_artifact(path: str | Path) -> dict:
    payload = read_json_artifact(path)
    if payload.get("schema_version") != E7_SCHEMA_VERSION:
        raise ValueError("Unsupported E7 artifact schema")
    return payload
