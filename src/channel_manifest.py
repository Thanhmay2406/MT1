from __future__ import annotations

from pathlib import Path
from typing import Any

from integrity import assert_file_sha256


CHANNEL_MANIFEST_SCHEMA = "study_b_e1_channel_sample/v2"
FROZEN_CHANNEL_MANIFEST_SHA256 = "c39bfcec29744c6c3c01024fba89515114f49f7fbbe07d29bd7d8ead3fdeea99"
EXPECTED_SAMPLE_ID = "sb_e1_sample_554d7f387c8baeea"
EXPECTED_GROUP_COUNT = 32
EXPECTED_CHANNELS_PER_GROUP = 12
EXPECTED_CHANNEL_COUNT = 384
EXPECTED_E2_SCHEMA = "causal_audit_e2_importance/v2"
EXPECTED_E2_CHANNEL_COUNT = 7552

_GROUP_KIND_TO_HIDDEN_CONV = {
    "bottleneck_conv1_hidden": "conv1",
    "bottleneck_conv2_hidden": "conv2",
}


def load_channel_manifest(path: str | Path) -> dict[str, Any]:
    import json

    path = Path(path)
    assert_file_sha256(path, FROZEN_CHANNEL_MANIFEST_SHA256)
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise ValueError("Channel manifest must be a JSON object")
    return manifest


def normalize_manifest_channel_id(entry: dict[str, Any]) -> str:
    required = (
        "canonical_channel_id",
        "producer_name",
        "local_channel_index",
    )
    missing = [key for key in required if key not in entry]
    if missing:
        raise ValueError(f"Manifest entry missing fields: {missing}")
    parts = str(entry["canonical_channel_id"]).split("|")
    if len(parts) != 8:
        raise ValueError("Invalid extended canonical channel ID")
    if parts[0] != entry["group_id"] or parts[4] != entry["producer_name"]:
        raise ValueError("Extended canonical channel ID does not match entry metadata")
    if parts[5] != entry["norm_name"] or parts[6] != entry["consumer_name"]:
        raise ValueError("Extended canonical channel dependency metadata mismatch")
    if int(parts[2]) != int(entry["block"]) or int(parts[7]) != int(entry["local_channel_index"]):
        raise ValueError("Extended canonical channel index metadata mismatch")
    return f"{entry['producer_name']}|channel={int(entry['local_channel_index'])}"


def _validate_criterion_identities(manifest: dict[str, Any], checkpoint_sha256: str, probe_sha256: str) -> None:
    identities = manifest.get("criterion_identities")
    if not isinstance(identities, dict):
        raise ValueError("Missing criterion identities")
    for criterion in ("xai", "taylor", "l1"):
        metadata = identities.get(criterion)
        if not isinstance(metadata, dict):
            raise ValueError(f"Missing {criterion} criterion identity")
        if metadata.get("baseline_checkpoint_sha256") != checkpoint_sha256:
            raise ValueError(f"{criterion} checkpoint hash mismatch")
        if int(metadata.get("channels", -1)) != EXPECTED_E2_CHANNEL_COUNT:
            raise ValueError(f"{criterion} channel universe mismatch")
        if int(metadata.get("groups", -1)) != EXPECTED_GROUP_COUNT:
            raise ValueError(f"{criterion} group count mismatch")
    if identities["xai"].get("probe_sha256") != probe_sha256:
        raise ValueError("XAI probe hash mismatch")
    if identities["taylor"].get("probe_sha256") != probe_sha256:
        raise ValueError("Taylor probe hash mismatch")


def validate_channel_manifest(
    manifest: dict[str, Any],
    e2_artifact: dict[str, Any],
    checkpoint_sha256: str,
    probe_sha256: str,
) -> int:
    if manifest.get("schema_version") != CHANNEL_MANIFEST_SCHEMA:
        raise ValueError("Unsupported channel manifest schema")
    if manifest.get("sample_id") != EXPECTED_SAMPLE_ID:
        raise ValueError("Unexpected frozen channel sample ID")
    if manifest.get("total_channels") != EXPECTED_CHANNEL_COUNT:
        raise ValueError("Manifest must contain 384 entries")
    if manifest.get("group_count") != EXPECTED_GROUP_COUNT:
        raise ValueError("Manifest must contain 32 groups")
    if manifest.get("channels_per_group") != EXPECTED_CHANNELS_PER_GROUP:
        raise ValueError("Manifest must contain 12 channels per group")
    if e2_artifact.get("schema_version") not in (EXPECTED_E2_SCHEMA, EXPECTED_E2_SCHEMA.replace("/v2", "/v1")):
        raise ValueError("Unsupported E2 artifact schema")
    if e2_artifact["schema_version"] == EXPECTED_E2_SCHEMA:
        from artifacts import validate_e2_contributions
        validate_e2_contributions(e2_artifact["channels"], e2_artifact["eligible_image_count"])
    if e2_artifact.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError("E2 checkpoint hash mismatch")
    if e2_artifact.get("probe_sha256") != probe_sha256:
        raise ValueError("E2 probe hash mismatch")
    if e2_artifact.get("channel_count") != EXPECTED_E2_CHANNEL_COUNT:
        raise ValueError("E2 channel universe must contain 7552 channels")

    _validate_criterion_identities(manifest, checkpoint_sha256, probe_sha256)
    entries = manifest.get("entries")
    if not isinstance(entries, list) or len(entries) != EXPECTED_CHANNEL_COUNT:
        raise ValueError("Manifest must contain 384 entries")
    e2_rows = {row["canonical_id"]: row for row in e2_artifact.get("channels", [])}
    if len(e2_rows) != EXPECTED_E2_CHANNEL_COUNT:
        raise ValueError("E2 canonical channel IDs must be unique")
    e2_groups = {row["group_id"]: row for row in e2_artifact.get("groups", [])}
    if len(e2_groups) != EXPECTED_GROUP_COUNT:
        raise ValueError("E2 structural groups must be unique")

    seen: set[str] = set()
    group_counts: dict[str, int] = {}
    for entry in entries:
        normalized = normalize_manifest_channel_id(entry)
        if normalized in seen:
            raise ValueError("Manifest canonical channel IDs must be unique")
        seen.add(normalized)
        group_id = str(entry["group_id"])
        group_counts[group_id] = group_counts.get(group_id, 0) + 1
        e2_row = e2_rows.get(normalized)
        if e2_row is None:
            raise ValueError(f"Manifest channel is absent from E2 canonical universe: {normalized}")
        expected_hidden_conv = _GROUP_KIND_TO_HIDDEN_CONV.get(entry.get("group_kind"))
        if expected_hidden_conv is None:
            raise ValueError(f"Unsupported manifest group kind: {entry.get('group_kind')}")
        for field in ("group_id", "stage", "block", "local_channel_index"):
            if e2_row.get(field) != entry.get(field):
                raise ValueError(f"Manifest/E2 {field} mismatch for {normalized}")
        if e2_row.get("hidden_conv") != expected_hidden_conv:
            raise ValueError(f"Manifest/E2 hidden_conv mismatch for {normalized}")
        e2_group = e2_groups.get(group_id)
        if e2_group is None:
            raise ValueError(f"Manifest group is absent from E2: {group_id}")
        for field in ("producer_name", "norm_name", "consumer_name"):
            if e2_group.get(field) != entry.get(field):
                raise ValueError(f"Manifest/E2 dependency mismatch for {normalized}")

    if len(group_counts) != EXPECTED_GROUP_COUNT:
        raise ValueError("Manifest must contain exactly 32 group IDs")
    if any(count != EXPECTED_CHANNELS_PER_GROUP for count in group_counts.values()):
        raise ValueError("Each manifest group must contain exactly 12 channels")
    return len(seen)
