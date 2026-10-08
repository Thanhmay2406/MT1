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

from artifacts import read_json_artifact, write_json_artifact
from channel_manifest import FROZEN_CHANNEL_MANIFEST_SHA256, load_channel_manifest
from equivalence import E4_SAMPLE_SCHEMA, canonical_json_hash, select_e4_sample
from integrity import assert_file_sha256

CHECKPOINT_SHA256 = "953a2b8d8e412227a89b9dd42c0899b33de28f281110af54829ec531db16dda0"
PROBE_SHA256 = "1016ed7eacda87c6b368c880a5564799e22e7fa757e11cd9cf93845b396d811d"
DATASET_MANIFEST_SHA256 = "fce7c5bc78d606c641873220f124d4656f1c32d96c7ca1e2dbc7b0b4d4f6536f"
EXPECTED_SAMPLE_ID = "sb_e4_equivalence_8x16"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze the deterministic E4 8x16 sample")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--probe", required=True)
    parser.add_argument("--channel-manifest", required=True)
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


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
    channel_manifest = load_channel_manifest(args.channel_manifest)
    entries = channel_manifest["entries"]
    selected_channels, selected_images = select_e4_sample(entries, probe)
    selected_channel_ids = [f"{entry['producer_name']}|channel={int(entry['local_channel_index'])}" for entry in selected_channels]
    manifest = {
        "schema_version": E4_SAMPLE_SCHEMA,
        "sample_id": EXPECTED_SAMPLE_ID,
        "sampling_rule": "raw SHA256-min canonical ID per stage x hidden-conv; first 16 probe entries irrespective of eligibility",
        "selection_inputs": "frozen_channel_manifest_and_probe_only",
        "checkpoint_sha256": checkpoint_sha256,
        "probe_sha256": probe_sha256,
        "dataset_manifest_sha256": DATASET_MANIFEST_SHA256,
        "channel_manifest_sha256": channel_manifest_sha256,
        "channels": [
            {
                "canonical_id": canonical_id,
                "group_id": entry["group_id"],
                "canonical_channel_id": entry["canonical_channel_id"],
                "stage": entry["stage"],
                "hidden_conv": entry.get("hidden_conv") or {"bottleneck_conv1_hidden": "conv1", "bottleneck_conv2_hidden": "conv2"}[entry["group_kind"]],
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
    write_json_artifact(output, manifest)
    print("E4_SAMPLE_FROZEN")
    print(f"output={output}")
    print(f"channel_count={manifest['channel_count']}")
    print(f"image_count={manifest['image_count']}")
    print(f"sample_identity_sha256={manifest['sample_identity_sha256']}")
    print(f"sample_sha256={manifest['sample_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
