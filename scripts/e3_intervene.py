from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from artifacts import build_e3_artifact, read_e3_artifact, read_json_artifact, write_json_artifact
from channel_manifest import FROZEN_CHANNEL_MANIFEST_SHA256, load_channel_manifest, validate_channel_manifest
from damage import aggregate_image_damage, compute_object_damage
from dataset_manifest import (
    DATASET_MANIFEST_SCHEMA,
    assert_dataset_manifest_sha256,
    load_dataset_manifest,
    validate_dataset_manifest,
)
from detector import load_detector_checkpoint
from identities import discover_structural_groups
from inference import detection_from_prediction, image_to_tensor
from integrity import assert_file_sha256, sha256_file
from intervention import InterventionSpec, ModelStateGuard, run_intervened_forward
from matching import GroundTruthObject, match_detections_to_gt

EXPECTED_CHECKPOINT_SHA256 = "953a2b8d8e412227a89b9dd42c0899b33de28f281110af54829ec531db16dda0"
EXPECTED_PROBE_SHA256 = "1016ed7eacda87c6b368c880a5564799e22e7fa757e11cd9cf93845b396d811d"
EXPECTED_DATASET_MANIFEST_SHA256 = "fce7c5bc78d606c641873220f124d4656f1c32d96c7ca1e2dbc7b0b4d4f6536f"
EXPECTED_E1_SCHEMA = "causal_audit_e1_eligibility/v1"
EXPECTED_E2_SCHEMA = "causal_audit_e2_importance/v1"
EXPECTED_PROBE_IMAGES = 300
EXPECTED_PROBE_ANNOTATIONS = 372
MATCHING_IOU_THRESHOLD = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="E3 same-channel causal intervention")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--probe", required=True)
    parser.add_argument("--images-root", required=True)
    parser.add_argument("--eligibility", required=True)
    parser.add_argument("--importance", required=True)
    parser.add_argument("--channel-manifest", required=True)
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=500,
        help="Persist resumable artifact after this many new pairs (default: 500)",
    )
    return parser.parse_args()


def load_probe(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        probe = json.load(handle)
    if not all(isinstance(probe.get(key), list) for key in ("images", "annotations", "categories")):
        raise ValueError("Probe must contain images, annotations, and categories lists")
    if len(probe["images"]) != EXPECTED_PROBE_IMAGES:
        raise ValueError(f"Expected {EXPECTED_PROBE_IMAGES} probe images")
    if len(probe["annotations"]) != EXPECTED_PROBE_ANNOTATIONS:
        raise ValueError(f"Expected {EXPECTED_PROBE_ANNOTATIONS} probe annotations")
    return probe


def validate_e1(eligibility: dict, probe: dict, checkpoint_sha256: str, probe_sha256: str) -> None:
    if eligibility.get("schema_version") != EXPECTED_E1_SCHEMA:
        raise ValueError("Unsupported E1 eligibility schema")
    if eligibility.get("checkpoint_sha256") != checkpoint_sha256 or eligibility.get("probe_sha256") != probe_sha256:
        raise ValueError("E1 provenance does not match frozen checkpoint/probe")
    if float(eligibility.get("matching_iou_threshold", -1)) != MATCHING_IOU_THRESHOLD:
        raise ValueError("E1 matching IoU threshold must remain 0.5")
    if eligibility.get("image_count") != EXPECTED_PROBE_IMAGES:
        raise ValueError("E1 image count does not match frozen probe")
    rows = eligibility.get("images", [])
    if len(rows) != len(probe["images"]):
        raise ValueError("E1 image ledger is incomplete")
    eligible_images = eligible_instances = 0
    for row, image in zip(rows, probe["images"]):
        if row.get("image_id") != image.get("id") or row.get("file_name") != image.get("file_name"):
            raise ValueError(f"E1 image order mismatch at image {image.get('id')}")
        matches = row.get("matches", [])
        if row.get("eligible_gt_ids") != [match.get("gt_id") for match in matches]:
            raise ValueError(f"E1 eligible IDs mismatch at image {image.get('id')}")
        if matches:
            eligible_images += 1
            eligible_instances += len(matches)
    if eligibility.get("eligible_image_count") != eligible_images:
        raise ValueError("E1 eligible image count is inconsistent")
    if eligibility.get("eligible_instance_count") != eligible_instances:
        raise ValueError("E1 eligible instance count is inconsistent")


def resolve_device(requested: str) -> str:
    import torch

    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _validate_inputs(args: argparse.Namespace, require_dataset: bool) -> dict:
    paths = {name: Path(getattr(args, name)) for name in (
        "checkpoint", "probe", "images_root", "eligibility", "importance", "channel_manifest", "dataset_manifest", "output"
    )}
    output = paths["output"]
    if output.exists() and not args.overwrite and not args.resume:
        raise FileExistsError(f"Output already exists; pass --overwrite or --resume: {output}")
    required_files = ("checkpoint", "probe", "eligibility", "importance", "channel_manifest")
    if any(not paths[name].is_file() for name in required_files) or not paths["images_root"].is_dir():
        raise FileNotFoundError("Checkpoint, probe, E1, E2, channel manifest, and images root are required")
    if args.checkpoint_every < 1:
        raise ValueError("--checkpoint-every must be positive")
    checkpoint_sha256 = assert_file_sha256(paths["checkpoint"], EXPECTED_CHECKPOINT_SHA256)
    probe_sha256 = assert_file_sha256(paths["probe"], EXPECTED_PROBE_SHA256)
    channel_manifest_sha256 = assert_file_sha256(paths["channel_manifest"], FROZEN_CHANNEL_MANIFEST_SHA256)
    dataset_manifest_sha256 = None
    dataset_manifest = None
    if paths["dataset_manifest"].is_file():
        dataset_manifest_sha256 = assert_dataset_manifest_sha256(paths["dataset_manifest"], EXPECTED_DATASET_MANIFEST_SHA256)
        dataset_manifest = load_dataset_manifest(paths["dataset_manifest"])
        if dataset_manifest.get("schema_version") != DATASET_MANIFEST_SCHEMA:
            raise ValueError("Unsupported dataset manifest schema")
        validate_dataset_manifest(
            dataset_manifest,
            paths["images_root"].parent,
            json.loads(paths["probe"].read_text(encoding="utf-8")),
            probe_path=paths["probe"],
        )
    elif require_dataset:
        raise FileNotFoundError("Frozen dataset manifest is required for scientific E3")
    probe = load_probe(paths["probe"])
    missing_images = [image["file_name"] for image in probe["images"] if not (paths["images_root"] / image["file_name"]).is_file()]
    if missing_images:
        raise FileNotFoundError(f"Missing {len(missing_images)} probe images")
    eligibility = read_json_artifact(paths["eligibility"])
    validate_e1(eligibility, probe, checkpoint_sha256, probe_sha256)
    importance = read_json_artifact(paths["importance"])
    if importance.get("schema_version") != EXPECTED_E2_SCHEMA:
        raise ValueError("Unsupported E2 importance schema")
    if importance.get("checkpoint_sha256") != checkpoint_sha256 or importance.get("probe_sha256") != probe_sha256:
        raise ValueError("E2 provenance does not match frozen checkpoint/probe")
    if importance.get("eligibility_sha256") != sha256_file(paths["eligibility"]):
        raise ValueError("E2 eligibility hash does not match supplied E1 artifact")
    eligibility_sha256 = sha256_file(paths["eligibility"])
    importance_sha256 = sha256_file(paths["importance"])
    manifest = load_channel_manifest(paths["channel_manifest"])
    validate_channel_manifest(manifest, importance, checkpoint_sha256, probe_sha256)
    return {
        "paths": paths,
        "probe": probe,
        "eligibility": eligibility,
        "importance": importance,
        "manifest": manifest,
        "checkpoint_sha256": checkpoint_sha256,
        "probe_sha256": probe_sha256,
        "channel_manifest_sha256": channel_manifest_sha256,
        "eligibility_sha256": eligibility_sha256,
        "importance_sha256": importance_sha256,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "dataset_manifest": dataset_manifest,
    }


def dry_run(args: argparse.Namespace) -> int:
    data = _validate_inputs(args, require_dataset=False)
    if data["dataset_manifest_sha256"] is None:
        print("E3_DRY_RUN_BLOCKED")
        print("reason=missing_frozen_dataset_manifest")
        print("scientific_run_allowed=false")
        return 2
    manifest = data["manifest"]
    print("E3_DRY_RUN_OK")
    print(f"channel_count={manifest['total_channels']}")
    print(f"group_count={manifest['group_count']}")
    print(f"channels_per_group={manifest['channels_per_group']}")
    print(f"expected_pair_count={manifest['total_channels'] * EXPECTED_PROBE_IMAGES}")
    print(f"channel_manifest_sha256={data['channel_manifest_sha256']}")
    print(f"dataset_manifest_sha256={data['dataset_manifest_sha256']}")
    return 0


def _build_specs(manifest: dict) -> list[InterventionSpec]:
    return [
        InterventionSpec(
            canonical_id=f"{entry['producer_name']}|channel={int(entry['local_channel_index'])}",
            group_id=entry["group_id"],
            norm_name=entry["norm_name"],
            channel_index=int(entry["local_channel_index"]),
        )
        for entry in manifest["entries"]
    ]


def _detections_from_output(output: dict, label_to_category_id: dict[int, int]):
    return [detection_from_prediction(output, index, label_to_category_id) for index in range(len(output["boxes"]))]


def _ordered_pairs(data: dict, pairs: list[dict]) -> list[dict]:
    channel_order = {
        f"{entry['producer_name']}|channel={int(entry['local_channel_index'])}": index
        for index, entry in enumerate(data["manifest"]["entries"])
    }
    image_order = {int(image["id"]): index for index, image in enumerate(data["probe"]["images"])}
    return sorted(
        pairs,
        key=lambda row: (channel_order[row["canonical_id"]], image_order[int(row["image_id"])]),
    )


def _write_progress(data: dict, pairs: list[dict]) -> None:
    manifest = data["manifest"]
    artifact = build_e3_artifact(
        checkpoint_sha256=data["checkpoint_sha256"],
        probe_sha256=data["probe_sha256"],
        eligibility_sha256=data["eligibility_sha256"],
        importance_sha256=data["importance_sha256"],
        dataset_manifest_sha256=data["dataset_manifest_sha256"],
        channel_manifest_sha256=data["channel_manifest_sha256"],
        sample_id=manifest["sample_id"],
        sample_identity_sha256=manifest["sample_identity_sha256"],
        matching_iou_threshold=MATCHING_IOU_THRESHOLD,
        channel_ids=[f"{entry['producer_name']}|channel={int(entry['local_channel_index'])}" for entry in manifest["entries"]],
        image_count=EXPECTED_PROBE_IMAGES,
        pairs=_ordered_pairs(data, pairs),
    )
    write_json_artifact(data["paths"]["output"], artifact)


def run_real(args: argparse.Namespace) -> int:
    import torch
    from PIL import Image

    data = _validate_inputs(args, require_dataset=True)
    device = resolve_device(args.device)
    model, metadata = load_detector_checkpoint(data["paths"]["checkpoint"], device=device)
    state_guard = ModelStateGuard(model)
    groups = discover_structural_groups(model)
    group_by_id = {group.group_id: group for group in groups}
    specs = _build_specs(data["manifest"])
    for spec in specs:
        if spec.group_id not in group_by_id or group_by_id[spec.group_id].norm_name != spec.norm_name:
            raise RuntimeError(f"Manifest module mapping changed for {spec.canonical_id}")

    existing: dict[tuple[str, int], dict] = {}
    pair_rows: dict[tuple[str, int], dict] = {}
    if args.resume and data["paths"]["output"].exists():
        previous = read_e3_artifact(data["paths"]["output"])
        expected_provenance = {
            "checkpoint_sha256": data["checkpoint_sha256"],
            "probe_sha256": data["probe_sha256"],
            "eligibility_sha256": data["eligibility_sha256"],
            "importance_sha256": data["importance_sha256"],
            "dataset_manifest_sha256": data["dataset_manifest_sha256"],
            "channel_manifest_sha256": data["channel_manifest_sha256"],
            "sample_id": data["manifest"]["sample_id"],
            "sample_identity_sha256": data["manifest"]["sample_identity_sha256"],
        }
        if any(previous.get(key) != value for key, value in expected_provenance.items()):
            raise ValueError("Resume artifact provenance does not match frozen inputs")
        if previous.get("channel_ids") != [spec.canonical_id for spec in specs]:
            raise ValueError("Resume artifact channel order does not match frozen manifest")
        pair_rows = {(row["canonical_id"], int(row["image_id"])): row for row in previous.get("pairs", [])}
        existing = {key: row for key, row in pair_rows.items() if row.get("status") == "ok"}
    annotations_by_image: dict[int, list[dict]] = {}
    for annotation in data["probe"]["annotations"]:
        annotations_by_image.setdefault(int(annotation["image_id"]), []).append(annotation)
    eligibility_by_image = {int(row["image_id"]): row for row in data["eligibility"]["images"]}
    newly_completed = 0
    for image in data["probe"]["images"]:
        image_id = int(image["id"])
        with Image.open(data["paths"]["images_root"] / image["file_name"]) as opened:
            tensor = image_to_tensor(opened).to(torch.device(device))
        gt_objects = [GroundTruthObject(int(a["id"]), int(a["category_id"]), a["bbox"]) for a in annotations_by_image.get(image_id, [])]
        original_row = eligibility_by_image[image_id]
        for spec in specs:
            key = (spec.canonical_id, int(image["id"]))
            if key in existing:
                continue
            pair = {"canonical_id": spec.canonical_id, "image_id": int(image["id"]), "file_name": image["file_name"]}
            try:
                before_digest = state_guard.digest
                prediction = run_intervened_forward(model, tensor, spec, state_guard=state_guard)
                after_digest = state_guard.digest
                detections = _detections_from_output(prediction, metadata["label_to_category_id"])
                rematches = match_detections_to_gt(gt_objects, detections, MATCHING_IOU_THRESHOLD)
                object_damage = compute_object_damage(original_row.get("matches", []), rematches)
                pair.update({
                    "status": "ok",
                    "eligible_gt_ids": original_row.get("eligible_gt_ids", []),
                    "original_matches": original_row.get("matches", []),
                    "intervened_matches": [match.to_dict() for match in rematches],
                    "object_damage": object_damage,
                    "image_damage": aggregate_image_damage(object_damage),
                    "state_digest_before": before_digest,
                    "state_digest_after": after_digest,
                    "state_restored": before_digest == after_digest,
                    "rng_restored": True,
                })
            except Exception as error:
                pair.update({"status": "error", "error_type": type(error).__name__, "error": str(error), "state_restored": False, "rng_restored": False})
            pair_rows[key] = pair
            newly_completed += 1
            if newly_completed % args.checkpoint_every == 0:
                _write_progress(data, list(pair_rows.values()))
    _write_progress(data, list(pair_rows.values()))
    final = read_e3_artifact(data["paths"]["output"])
    if not final["complete"]:
        raise RuntimeError(f"E3 incomplete: {final['completed_pair_count']}/{final['expected_pair_count']} pairs")
    print("E3_OK")
    print(f"output={data['paths']['output']}")
    print(f"completed_pair_count={final['completed_pair_count']}")
    return 0


def main() -> int:
    args = parse_args()
    return dry_run(args) if args.dry_run else run_real(args)


if __name__ == "__main__":
    raise SystemExit(main())
