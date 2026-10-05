from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from artifacts import read_json_artifact
from channel_manifest import FROZEN_CHANNEL_MANIFEST_SHA256, load_channel_manifest
from equivalence import E4_SAMPLE_SCHEMA, canonical_json_hash
from integrity import assert_file_sha256, sha256_file

CHECKPOINT_SHA256 = "953a2b8d8e412227a89b9dd42c0899b33de28f281110af54829ec531db16dda0"
PROBE_SHA256 = "1016ed7eacda87c6b368c880a5564799e22e7fa757e11cd9cf93845b396d811d"
DATASET_MANIFEST_SHA256 = "fce7c5bc78d606c641873220f124d4656f1c32d96c7ca1e2dbc7b0b4d4f6536f"
E1_SCHEMA = "causal_audit_e1_eligibility/v1"
E2_SCHEMA = "causal_audit_e2_importance/v1"
E3_SCHEMA = "causal_audit_e3_damage/v1"
EXPECTED_SAMPLE_ID = "sb_e4_equivalence_8x16"
SEED_ID = "sha256:causal_audit_e4_equivalence/v1|sb_e1_sample_554d7f387c8baeea"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze the deterministic E4 8x16 sample")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--probe", required=True)
    parser.add_argument("--eligibility", required=True)
    parser.add_argument("--importance", required=True)
    parser.add_argument("--intervention", required=True)
    parser.add_argument("--channel-manifest", required=True)
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _rank(seed: str, kind: str, value: str) -> str:
    return hashlib.sha256(f"{seed}|{kind}|{value}".encode("utf-8")).hexdigest()


def _canonical_payload(manifest: dict) -> dict:
    return {key: value for key, value in manifest.items() if key not in {"sample_sha256", "sample_identity_sha256"}}


def main() -> int:
    args = parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"E4 sample already exists; refusing to regenerate: {output}")
    checkpoint_sha256 = assert_file_sha256(args.checkpoint, CHECKPOINT_SHA256)
    probe_sha256 = assert_file_sha256(args.probe, PROBE_SHA256)
    channel_manifest_sha256 = assert_file_sha256(args.channel_manifest, FROZEN_CHANNEL_MANIFEST_SHA256)
    dataset_manifest = read_json_artifact(args.dataset_manifest)
    if dataset_manifest.get("manifest_sha256") != DATASET_MANIFEST_SHA256:
        raise ValueError("Dataset manifest self-hash mismatch")
    probe = json.loads(Path(args.probe).read_text(encoding="utf-8"))
    eligibility = read_json_artifact(args.eligibility)
    importance = read_json_artifact(args.importance)
    intervention = read_json_artifact(args.intervention)
    if eligibility.get("schema_version") != E1_SCHEMA or importance.get("schema_version") != E2_SCHEMA:
        raise ValueError("E1/E2 schema mismatch")
    if intervention.get("schema_version") != E3_SCHEMA or not intervention.get("complete"):
        raise ValueError("E3 artifact must be complete before freezing E4 sample")
    if any(payload.get("checkpoint_sha256") != checkpoint_sha256 for payload in (eligibility, importance, intervention)):
        raise ValueError("Checkpoint provenance mismatch")
    if any(payload.get("probe_sha256") != probe_sha256 for payload in (eligibility, importance, intervention)):
        raise ValueError("Probe provenance mismatch")
    if importance.get("eligibility_sha256") != sha256_file(args.eligibility):
        raise ValueError("E2 does not reference the supplied E1 artifact")
    if intervention.get("eligibility_sha256") != sha256_file(args.eligibility) or intervention.get("importance_sha256") != sha256_file(args.importance):
        raise ValueError("E3 does not reference the supplied E1/E2 artifacts")
    if intervention.get("dataset_manifest_sha256") != DATASET_MANIFEST_SHA256:
        raise ValueError("E3 dataset manifest provenance mismatch")
    channel_manifest = load_channel_manifest(args.channel_manifest)
    entries = channel_manifest["entries"]
    groups: dict[str, list[dict]] = {}
    for entry in entries:
        groups.setdefault(entry["group_id"], []).append(entry)
    group_order = list(groups)
    selected_group_ids = [group_order[(index * len(group_order)) // 8] for index in range(8)]
    selected_channels = []
    for group_id in selected_group_ids:
        selected_channels.append(min(groups[group_id], key=lambda entry: _rank(SEED_ID, "channel", entry["canonical_channel_id"])))
    selected_channel_ids = [f"{entry['producer_name']}|channel={int(entry['local_channel_index'])}" for entry in selected_channels]
    eligible_images = [row for row in eligibility["images"] if row.get("matches")]
    if len(eligible_images) < 16:
        raise ValueError("E1 does not contain 16 eligible images")
    chosen_image_ids = {
        int(row["image_id"])
        for row in sorted(eligible_images, key=lambda row: _rank(SEED_ID, "image", str(row["image_id"])))[:16]
    }
    probe_images = {int(image["id"]): image for image in probe["images"]}
    selected_images = [probe_images[image_id] for image_id in [int(image["id"]) for image in probe["images"]] if image_id in chosen_image_ids]
    e3_sha256 = sha256_file(args.intervention)
    manifest = {
        "schema_version": E4_SAMPLE_SCHEMA,
        "sample_id": EXPECTED_SAMPLE_ID,
        "sampling_rule": "8 evenly spaced structural groups; one SHA-256-ranked channel per group; 16 SHA-256-ranked eligible TRAIN images; restore probe order",
        "sampling_seed_id": SEED_ID,
        "checkpoint_sha256": checkpoint_sha256,
        "probe_sha256": probe_sha256,
        "eligibility_sha256": sha256_file(args.eligibility),
        "importance_sha256": sha256_file(args.importance),
        "intervention_sha256": e3_sha256,
        "dataset_manifest_sha256": DATASET_MANIFEST_SHA256,
        "channel_manifest_sha256": channel_manifest_sha256,
        "channels": [
            {
                "canonical_id": canonical_id,
                "group_id": entry["group_id"],
                "producer_name": entry["producer_name"],
                "norm_name": entry["norm_name"],
                "consumer_name": entry["consumer_name"],
                "local_channel_index": int(entry["local_channel_index"]),
            }
            for canonical_id, entry in zip(selected_channel_ids, selected_channels)
        ],
        "images": [{"image_id": int(image["id"]), "file_name": image["file_name"]} for image in selected_images],
        "channel_count": 8,
        "image_count": 16,
    }
    identity = {
        "channels": manifest["channels"],
        "images": manifest["images"],
        "sample_id": EXPECTED_SAMPLE_ID,
    }
    manifest["sample_identity_sha256"] = canonical_json_hash(identity)
    manifest["sample_sha256"] = canonical_json_hash(_canonical_payload(manifest))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("E4_SAMPLE_FROZEN")
    print(f"output={output}")
    print(f"channel_count={manifest['channel_count']}")
    print(f"image_count={manifest['image_count']}")
    print(f"sample_identity_sha256={manifest['sample_identity_sha256']}")
    print(f"sample_sha256={manifest['sample_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
