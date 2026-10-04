from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from integrity import HashMismatchError, sha256_file


DATASET_MANIFEST_SCHEMA = "causal_audit_dataset_manifest/amended-v1"
MANIFEST_ID = "mt1_dataset_manifest_amendment_v1"
SUPERSEDED_MANIFEST_SHA256 = "4f46d0133bf834cb03e3c8fb0c9984d0ebaad8b5006f84198ecbb455240c5f9e"


def canonical_json(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("utf-8")


def manifest_sha256(manifest: dict[str, Any]) -> str:
    payload = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    return hashlib.sha256(canonical_json(payload)).hexdigest()


def _file_record(path: Path, root: Path, split: str, kind: str) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "split": split,
        "kind": kind,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _load_coco(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not all(isinstance(value.get(key), list) for key in ("images", "annotations", "categories")):
        raise ValueError(f"Invalid COCO annotation file: {path}")
    return value


def build_dataset_manifest(dataset_root: str | Path, probe_path: str | Path) -> dict[str, Any]:
    root = Path(dataset_root)
    probe_path = Path(probe_path)
    if not root.is_dir() or not probe_path.is_file():
        raise FileNotFoundError("Dataset root and probe file are required")
    records: list[dict[str, Any]] = []
    splits: dict[str, dict[str, Any]] = {}
    for split in ("train", "valid"):
        split_root = root / split
        annotation_path = split_root / "_annotations.coco.json"
        if not split_root.is_dir() or not annotation_path.is_file():
            raise FileNotFoundError(f"Missing {split} split or COCO annotation file")
        images = sorted(split_root.glob("*.jpg"), key=lambda path: path.name)
        unexpected = [path for path in split_root.iterdir() if path.name != annotation_path.name and path not in images]
        if unexpected:
            raise ValueError(f"Unexpected {split} scientific files: {', '.join(path.name for path in unexpected)}")
        coco = _load_coco(annotation_path)
        records.extend(_file_record(path, root, split, "image") for path in images)
        records.append(_file_record(annotation_path, root, split, "coco_annotations"))
        splits[split] = {
            "image_root": f"{split}/",
            "image_count": len(images),
            "annotation_file": annotation_path.relative_to(root).as_posix(),
            "annotation_count": len(coco["annotations"]),
            "category_ids": sorted(int(category["id"]) for category in coco["categories"]),
        }

    metadata_records = []
    for relative in ("dataset-metadata.json", "README.dataset.txt", "README.roboflow.txt"):
        path = root / relative
        if path.is_file():
            metadata_records.append(_file_record(path, root, "metadata", "source_metadata"))

    with probe_path.open("r", encoding="utf-8") as handle:
        probe = json.load(handle)
    train_names = {record["path"].split("/", 1)[1] for record in records if record["split"] == "train" and record["kind"] == "image"}
    probe_names = [str(image["file_name"]) for image in probe.get("images", [])]
    if any(name not in train_names for name in probe_names):
        raise ValueError("Probe contains an image outside the frozen TRAIN split")
    probe_sha = sha256_file(probe_path)
    manifest = {
        "schema_version": DATASET_MANIFEST_SCHEMA,
        "manifest_id": MANIFEST_ID,
        "amendment_id": "dataset_manifest_amendment_v1",
        "superseded_manifest_sha256": SUPERSEDED_MANIFEST_SHA256,
        "dataset_source": {
            "dataset_root": root.name,
            "source_id": "thanhmay2406/drill-bit-coco",
            "source_metadata": metadata_records,
        },
        "splits": splits,
        "probe_reference": {
            "path": probe_path.as_posix(),
            "sha256": probe_sha,
            "image_count": len(probe.get("images", [])),
            "annotation_count": len(probe.get("annotations", [])),
            "ordered_image_ids": [int(image["id"]) for image in probe.get("images", [])],
            "ordered_file_names": probe_names,
        },
        "excluded_splits": ["test"],
        "file_records": sorted(records, key=lambda record: record["path"]),
    }
    manifest["manifest_sha256"] = manifest_sha256(manifest)
    return manifest


def write_dataset_manifest(manifest: dict[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(canonical_json(manifest))


def load_dataset_manifest(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise ValueError("Dataset manifest must be a JSON object")
    if manifest.get("manifest_sha256") != manifest_sha256(manifest):
        raise ValueError("Dataset manifest self-hash mismatch")
    return manifest


def validate_dataset_manifest(
    manifest: dict[str, Any],
    dataset_root: str | Path,
    probe: dict[str, Any],
    probe_path: str | Path | None = None,
) -> None:
    if manifest.get("schema_version") != DATASET_MANIFEST_SCHEMA:
        raise ValueError("Unsupported dataset manifest schema")
    if manifest.get("manifest_id") != MANIFEST_ID:
        raise ValueError("Unexpected dataset manifest ID")
    if manifest.get("superseded_manifest_sha256") != SUPERSEDED_MANIFEST_SHA256:
        raise ValueError("Dataset manifest amendment does not supersede the frozen predecessor")
    root = Path(dataset_root)
    records = manifest.get("file_records")
    if not isinstance(records, list):
        raise ValueError("Dataset manifest file_records must be a list")
    paths = [record.get("path") for record in records]
    if len(paths) != len(set(paths)):
        raise ValueError("Dataset manifest contains duplicate paths")
    if any(path.startswith("test/") for path in paths):
        raise ValueError("Test split must not appear in scientific file records")
    actual_paths = set()
    for split in ("train", "valid"):
        split_root = root / split
        annotation = split_root / "_annotations.coco.json"
        if not annotation.is_file():
            raise FileNotFoundError(f"Missing {split} annotations")
        actual_paths.update((path.relative_to(root).as_posix() for path in split_root.glob("*.jpg")))
        actual_paths.add(annotation.relative_to(root).as_posix())
    if actual_paths != set(paths):
        raise ValueError("Dataset manifest file records do not match dataset tree")
    for record in records:
        path = root / record["path"]
        if path.stat().st_size != record["bytes"] or sha256_file(path) != record["sha256"]:
            raise ValueError(f"Dataset file changed: {record['path']}")
    train_names = {record["path"].split("/", 1)[1] for record in records if record["split"] == "train" and record["kind"] == "image"}
    probe_names = [str(image["file_name"]) for image in probe.get("images", [])]
    if probe_names != manifest.get("probe_reference", {}).get("ordered_file_names"):
        raise ValueError("Probe order or filenames do not match dataset manifest")
    if any(name not in train_names for name in probe_names):
        raise ValueError("Probe contains an image outside TRAIN")
    actual_probe_path = Path(probe_path) if probe_path is not None else Path(manifest["probe_reference"]["path"])
    if not actual_probe_path.is_file():
        raise FileNotFoundError(f"Probe file not found: {actual_probe_path}")
    if Path(manifest["probe_reference"]["path"]).name != actual_probe_path.name:
        raise ValueError("Probe path basename does not match manifest reference")
    if manifest["probe_reference"].get("sha256") != sha256_file(actual_probe_path):
        raise ValueError("Probe file changed from manifest reference")


def assert_dataset_manifest_sha256(path: str | Path, expected_sha256: str) -> str:
    manifest = load_dataset_manifest(path)
    actual_sha256 = manifest["manifest_sha256"]
    if actual_sha256 != expected_sha256:
        raise HashMismatchError(
            f"Dataset manifest self-hash mismatch for {path}: expected {expected_sha256}, got {actual_sha256}"
        )
    return actual_sha256
