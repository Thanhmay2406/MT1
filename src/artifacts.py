from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


E2_SCHEMA_VERSION = "causal_audit_e2_importance/v1"


def write_json_artifact(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(payload, "to_dict"):
        payload = payload.to_dict()
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=target.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)


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
    return {
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


def read_e2_artifact(path: str | Path) -> dict:
    payload = read_json_artifact(path)
    if payload.get("schema_version") != E2_SCHEMA_VERSION:
        raise ValueError("Unsupported E2 artifact schema")
    return payload
