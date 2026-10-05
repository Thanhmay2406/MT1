from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from artifacts import build_e4_artifact, read_e3_artifact, read_e4_artifact, read_json_artifact, write_json_artifact
from channel_manifest import FROZEN_CHANNEL_MANIFEST_SHA256, load_channel_manifest, validate_channel_manifest
from damage import compute_object_damage
from dataset_manifest import DATASET_MANIFEST_SCHEMA, assert_dataset_manifest_sha256, load_dataset_manifest, validate_dataset_manifest
from detector import load_detector_checkpoint
from equivalence import E4_SAMPLE_SCHEMA, E4_TOLERANCE, PhysicalRemovalSpec, build_physical_removal_model, canonical_json_hash, compare_mask_and_physical
from identities import discover_structural_groups
from inference import detection_from_prediction, image_to_tensor
from integrity import assert_file_sha256, sha256_file
from matching import GroundTruthObject, match_detections_to_gt

CHECKPOINT_SHA256 = "953a2b8d8e412227a89b9dd42c0899b33de28f281110af54829ec531db16dda0"
PROBE_SHA256 = "1016ed7eacda87c6b368c880a5564799e22e7fa757e11cd9cf93845b396d811d"
DATASET_MANIFEST_SHA256 = "fce7c5bc78d606c641873220f124d4656f1c32d96c7ca1e2dbc7b0b4d4f6536f"
E1_SCHEMA = "causal_audit_e1_eligibility/v1"
E2_SCHEMA = "causal_audit_e2_importance/v1"
E3_SCHEMA = "causal_audit_e3_damage/v1"
EXPECTED_IMAGES = 300
EXPECTED_ANNOTATIONS = 372
MATCHING_IOU_THRESHOLD = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="E4 post-BN mask versus physical removal equivalence")
    for name in ("checkpoint", "probe", "images-root", "eligibility", "importance", "intervention", "channel-manifest", "e4-sample", "dataset-manifest", "output"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _load_probe(path: Path) -> dict:
    probe = json.loads(path.read_text(encoding="utf-8"))
    if len(probe.get("images", [])) != EXPECTED_IMAGES or len(probe.get("annotations", [])) != EXPECTED_ANNOTATIONS:
        raise ValueError("Probe size does not match frozen E1/E3 probe")
    return probe


def _validate_e1(e1: dict, probe: dict, checkpoint_sha256: str, probe_sha256: str) -> None:
    if e1.get("schema_version") != E1_SCHEMA:
        raise ValueError("Unsupported E1 schema")
    if e1.get("checkpoint_sha256") != checkpoint_sha256 or e1.get("probe_sha256") != probe_sha256:
        raise ValueError("E1 provenance mismatch")
    rows = e1.get("images", [])
    if len(rows) != len(probe["images"]):
        raise ValueError("E1 image ledger is incomplete")
    for row, image in zip(rows, probe["images"]):
        if row.get("image_id") != image.get("id") or row.get("file_name") != image.get("file_name"):
            raise ValueError("E1/probe image order mismatch")


def _validate_e3(e3: dict, probe: dict, channel_ids: list[str], checkpoint_sha256: str, probe_sha256: str) -> dict[tuple[str, int], dict]:
    if e3.get("schema_version") != E3_SCHEMA or not e3.get("complete"):
        raise ValueError("E3 must be complete before E4")
    if e3.get("checkpoint_sha256") != checkpoint_sha256 or e3.get("probe_sha256") != probe_sha256:
        raise ValueError("E3 checkpoint/probe provenance mismatch")
    expected_keys = {(channel, int(image["id"])) for channel in channel_ids for image in probe["images"]}
    rows = e3.get("pairs", [])
    indexed = {(row.get("canonical_id"), int(row.get("image_id"))): row for row in rows}
    if len(indexed) != len(rows) or set(indexed) != expected_keys:
        raise ValueError("E3 channel-image Cartesian coverage is incomplete or duplicated")
    if any(row.get("status") != "ok" or row.get("state_restored") is not True or row.get("rng_restored") is not True for row in rows):
        raise ValueError("E3 contains an unverified pair")
    return indexed


def _validate_sample(sample: dict, e3: dict, probe: dict, e1: dict, channel_ids: list[str], hashes: dict[str, str]) -> tuple[list[dict], list[dict]]:
    if sample.get("schema_version") != E4_SAMPLE_SCHEMA:
        raise ValueError("Unsupported E4 sample schema")
    if sample.get("sample_sha256") != canonical_json_hash({key: value for key, value in sample.items() if key not in {"sample_sha256", "sample_identity_sha256"}}):
        raise ValueError("E4 sample self-hash mismatch")
    identity = {"channels": sample.get("channels"), "images": sample.get("images"), "sample_id": sample.get("sample_id")}
    if sample.get("sample_identity_sha256") != canonical_json_hash(identity):
        raise ValueError("E4 sample identity hash mismatch")
    for field in ("checkpoint_sha256", "probe_sha256", "eligibility_sha256", "importance_sha256", "intervention_sha256", "dataset_manifest_sha256", "channel_manifest_sha256"):
        if sample.get(field) != hashes[field]:
            raise ValueError(f"E4 sample {field} mismatch")
    channels = sample.get("channels", [])
    images = sample.get("images", [])
    if len(channels) != 8 or len(images) != 16 or sample.get("channel_count") != 8 or sample.get("image_count") != 16:
        raise ValueError("E4 sample must contain exactly 8 channels and 16 images")
    if len({row["canonical_id"] for row in channels}) != 8 or len({row["group_id"] for row in channels}) != 8:
        raise ValueError("E4 channels must be unique and span 8 groups")
    if any(row["canonical_id"] not in set(channel_ids) for row in channels):
        raise ValueError("E4 sample channel is absent from frozen channel manifest")
    probe_by_id = {int(image["id"]): image for image in probe["images"]}
    e1_by_id = {int(row["image_id"]): row for row in e1["images"]}
    if len({int(row["image_id"]) for row in images}) != 16:
        raise ValueError("E4 sample image IDs must be unique")
    for row in images:
        image_id = int(row["image_id"])
        if image_id not in probe_by_id or row["file_name"] != probe_by_id[image_id]["file_name"]:
            raise ValueError("E4 sample image does not match probe")
        if not e1_by_id[image_id].get("matches"):
            raise ValueError("E4 sample image is not eligible in E1")
    return channels, images


def _resolve_device(requested: str) -> str:
    import torch

    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return "cuda" if requested == "auto" and torch.cuda.is_available() else ("cpu" if requested == "auto" else requested)


def _preflight(args: argparse.Namespace) -> dict:
    paths = {key.replace("-", "_"): Path(getattr(args, key.replace("-", "_"))) for key in ("checkpoint", "probe", "images-root", "eligibility", "importance", "intervention", "channel-manifest", "e4-sample", "dataset-manifest", "output")}
    if paths["output"].exists() and not args.overwrite and not args.resume:
        raise FileExistsError(f"Output already exists; pass --overwrite or --resume: {paths['output']}")
    if not paths["images_root"].is_dir():
        raise FileNotFoundError("Images root is missing")
    checkpoint_sha256 = assert_file_sha256(paths["checkpoint"], CHECKPOINT_SHA256)
    probe_sha256 = assert_file_sha256(paths["probe"], PROBE_SHA256)
    channel_manifest_sha256 = assert_file_sha256(paths["channel_manifest"], FROZEN_CHANNEL_MANIFEST_SHA256)
    dataset_manifest_sha256 = assert_dataset_manifest_sha256(paths["dataset_manifest"], DATASET_MANIFEST_SHA256)
    dataset_manifest = load_dataset_manifest(paths["dataset_manifest"])
    if dataset_manifest.get("schema_version") != DATASET_MANIFEST_SCHEMA:
        raise ValueError("Unsupported dataset manifest schema")
    probe = _load_probe(paths["probe"])
    validate_dataset_manifest(dataset_manifest, paths["images_root"].parent, probe, probe_path=paths["probe"])
    e1 = read_json_artifact(paths["eligibility"])
    _validate_e1(e1, probe, checkpoint_sha256, probe_sha256)
    e2 = read_json_artifact(paths["importance"])
    if e2.get("schema_version") != E2_SCHEMA or e2.get("checkpoint_sha256") != checkpoint_sha256 or e2.get("probe_sha256") != probe_sha256:
        raise ValueError("E2 provenance mismatch")
    e1_sha256 = sha256_file(paths["eligibility"])
    e2_sha256 = sha256_file(paths["importance"])
    if e2.get("eligibility_sha256") != e1_sha256:
        raise ValueError("E2 does not reference supplied E1")
    channel_manifest = load_channel_manifest(paths["channel_manifest"])
    validate_channel_manifest(channel_manifest, e2, checkpoint_sha256, probe_sha256)
    channel_ids = [f"{entry['producer_name']}|channel={int(entry['local_channel_index'])}" for entry in channel_manifest["entries"]]
    e3 = read_e3_artifact(paths["intervention"])
    e3_sha256 = sha256_file(paths["intervention"])
    if e3.get("eligibility_sha256") != e1_sha256 or e3.get("importance_sha256") != e2_sha256 or e3.get("dataset_manifest_sha256") != dataset_manifest_sha256 or e3.get("channel_manifest_sha256") != channel_manifest_sha256:
        raise ValueError("E3 provenance mismatch")
    e3_pairs = _validate_e3(e3, probe, channel_ids, checkpoint_sha256, probe_sha256)
    sample = json.loads(paths["e4_sample"].read_text(encoding="utf-8"))
    hashes = {
        "checkpoint_sha256": checkpoint_sha256,
        "probe_sha256": probe_sha256,
        "eligibility_sha256": e1_sha256,
        "importance_sha256": e2_sha256,
        "intervention_sha256": e3_sha256,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "channel_manifest_sha256": channel_manifest_sha256,
    }
    sample_channels, sample_images = _validate_sample(sample, e3, probe, e1, channel_ids, hashes)
    missing_images = [row["file_name"] for row in sample_images if not (paths["images_root"] / row["file_name"]).is_file()]
    if missing_images:
        raise FileNotFoundError(f"Missing {len(missing_images)} selected E4 images")
    return {"paths": paths, "probe": probe, "e1": e1, "e2": e2, "e3": e3, "e3_pairs": e3_pairs, "channel_manifest": channel_manifest, "sample": sample, "sample_channels": sample_channels, "sample_images": sample_images, "hashes": hashes, "channel_ids": channel_ids, "dataset_manifest": dataset_manifest}


def dry_run(args: argparse.Namespace) -> int:
    data = _preflight(args)
    print("E4_DRY_RUN_OK")
    print("channel_count=8")
    print("image_count=16")
    print("expected_pair_count=128")
    print(f"e4_sample_sha256={sha256_file(data['paths']['e4_sample'])}")
    print(f"intervention_sha256={data['hashes']['intervention_sha256']}")
    return 0


def _build_spec(entry: dict) -> PhysicalRemovalSpec:
    return PhysicalRemovalSpec(
        canonical_id=entry["canonical_id"],
        group_id=entry["group_id"],
        producer_name=entry["producer_name"],
        norm_name=entry["norm_name"],
        consumer_name=entry["consumer_name"],
        channel_index=int(entry["local_channel_index"]),
    )


def run_real(args: argparse.Namespace) -> int:
    import torch
    from PIL import Image

    data = _preflight(args)
    device = _resolve_device(args.device)
    model, metadata = load_detector_checkpoint(data["paths"]["checkpoint"], device=device)
    groups = {group.group_id: group for group in discover_structural_groups(model)}
    for entry in data["sample_channels"]:
        group = groups.get(entry["group_id"])
        if group is None or group.producer_name != entry["producer_name"] or group.norm_name != entry["norm_name"] or group.consumer_name != entry["consumer_name"]:
            raise RuntimeError(f"Physical mapping changed for {entry['canonical_id']}")
    annotations_by_image: dict[int, list[dict]] = {}
    for annotation in data["probe"]["annotations"]:
        annotations_by_image.setdefault(int(annotation["image_id"]), []).append(annotation)
    e1_by_image = {int(row["image_id"]): row for row in data["e1"]["images"]}
    tensors = {}
    for image in data["sample_images"]:
        with Image.open(data["paths"]["images_root"] / image["file_name"]) as opened:
            tensors[int(image["image_id"])] = image_to_tensor(opened).to(torch.device(device))
    pair_rows = {}
    if args.resume and data["paths"]["output"].exists():
        previous = read_e4_artifact(data["paths"]["output"])
        for key, value in data["hashes"].items():
            artifact_key = "intervention_sha256" if key == "intervention_sha256" else key
            if previous.get(artifact_key) != value:
                raise ValueError("Resume E4 artifact provenance mismatch")
        pair_rows = {(row["canonical_id"], int(row["image_id"])): row for row in previous.get("pairs", []) if row.get("status") == "ok"}
    channel_ids = [entry["canonical_id"] for entry in data["sample_channels"]]
    image_ids = [int(image["image_id"]) for image in data["sample_images"]]
    for channel_index, entry in enumerate(data["sample_channels"], start=1):
        spec = _build_spec(entry)
        physical_model = build_physical_removal_model(model, spec).to(torch.device(device)).eval()
        for image_index, image in enumerate(data["sample_images"], start=1):
            image_id = int(image["image_id"])
            key = (entry["canonical_id"], image_id)
            if key in pair_rows:
                continue
            row = {"canonical_id": entry["canonical_id"], "image_id": image_id, "file_name": image["file_name"]}
            try:
                gt_objects = [GroundTruthObject(int(a["id"]), int(a["category_id"]), a["bbox"]) for a in annotations_by_image.get(image_id, [])]
                with torch.inference_mode():
                    prediction = physical_model([tensors[image_id]])[0]
                detections = [detection_from_prediction(prediction, index, metadata["label_to_category_id"]) for index in range(len(prediction["boxes"]))]
                rematches = match_detections_to_gt(gt_objects, detections, MATCHING_IOU_THRESHOLD)
                physical_damage = compute_object_damage(e1_by_image[image_id].get("matches", []), rematches)
                mask_row = data["e3_pairs"][key]
                comparison = compare_mask_and_physical(mask_row.get("object_damage", []), physical_damage, E4_TOLERANCE)
                row.update({
                    "status": "ok",
                    "original_matches": e1_by_image[image_id].get("matches", []),
                    "mask_object_damage": mask_row.get("object_damage", []),
                    "mask_image_damage": mask_row.get("image_damage", 0.0),
                    "physical_matches": [match.to_dict() for match in rematches],
                    "physical_object_damage": physical_damage,
                    "physical_image_damage": sum(float(item["damage"]) for item in physical_damage) / len(physical_damage) if physical_damage else 0.0,
                    **comparison,
                })
            except Exception as error:
                row.update({"status": "error", "error_type": type(error).__name__, "error": str(error), "equivalent": False})
            pair_rows[key] = row
            write_json_artifact(data["paths"]["output"], build_e4_artifact(
                checkpoint_sha256=data["hashes"]["checkpoint_sha256"], probe_sha256=data["hashes"]["probe_sha256"], eligibility_sha256=data["hashes"]["eligibility_sha256"], importance_sha256=data["hashes"]["importance_sha256"], intervention_sha256=data["hashes"]["intervention_sha256"], dataset_manifest_sha256=data["hashes"]["dataset_manifest_sha256"], channel_manifest_sha256=data["hashes"]["channel_manifest_sha256"], e4_sample_sha256=sha256_file(data["paths"]["e4_sample"]), sample_id=data["sample"]["sample_id"], sample_identity_sha256=data["sample"]["sample_identity_sha256"], tolerance=E4_TOLERANCE, channel_ids=channel_ids, image_ids=image_ids, pairs=list(pair_rows.values()),
            ))
        del physical_model
    final = read_e4_artifact(data["paths"]["output"])
    if not final["complete"]:
        raise RuntimeError(f"E4 incomplete: {final['completed_pair_count']}/{final['expected_pair_count']} pairs")
    print("E4_OK")
    print(f"output={data['paths']['output']}")
    print(f"completed_pair_count={final['completed_pair_count']}")
    print(f"equivalence_status={final['equivalence_status']}")
    return 0


def main() -> int:
    args = parse_args()
    return dry_run(args) if args.dry_run else run_real(args)


if __name__ == "__main__":
    raise SystemExit(main())
