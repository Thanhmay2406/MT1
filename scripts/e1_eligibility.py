from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from integrity import assert_file_sha256, sha256_file


EXPECTED_CHECKPOINT_SHA256 = "953a2b8d8e412227a89b9dd42c0899b33de28f281110af54829ec531db16dda0"
EXPECTED_PROBE_SHA256 = "1016ed7eacda87c6b368c880a5564799e22e7fa757e11cd9cf93845b396d811d"
EXPECTED_PROBE_IMAGES = 300
EXPECTED_PROBE_ANNOTATIONS = 372
EXPECTED_CATEGORY_IDS = [0, 1, 2, 3, 4, 5]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="E1 eligibility code-only entrypoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--probe", required=True)
    parser.add_argument("--images-root", required=True)
    parser.add_argument("--output")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--allow-any-hash",
        action="store_true",
        help="Allow synthetic fixtures to skip frozen checkpoint/probe hash checks.",
    )
    return parser.parse_args()


def load_probe(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        probe = json.load(handle)
    for key in ("images", "annotations", "categories"):
        if key not in probe or not isinstance(probe[key], list):
            raise ValueError(f"Probe COCO JSON must contain list field {key!r}")
    return probe


def validate_probe_schema(probe: dict, strict_frozen_probe: bool) -> None:
    image_ids = [image.get("id") for image in probe["images"]]
    if len(image_ids) != len(set(image_ids)):
        raise ValueError("Probe image IDs must be unique")

    known_image_ids = set(image_ids)
    for annotation in probe["annotations"]:
        if annotation.get("image_id") not in known_image_ids:
            raise ValueError(f"Annotation {annotation.get('id')} references an unknown image")
        bbox = annotation.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError(f"Annotation {annotation.get('id')} has invalid bbox")

    if strict_frozen_probe:
        category_ids = sorted(category["id"] for category in probe["categories"])
        if len(probe["images"]) != EXPECTED_PROBE_IMAGES:
            raise ValueError(f"Expected {EXPECTED_PROBE_IMAGES} probe images, got {len(probe['images'])}")
        if len(probe["annotations"]) != EXPECTED_PROBE_ANNOTATIONS:
            raise ValueError(
                f"Expected {EXPECTED_PROBE_ANNOTATIONS} probe annotations, got {len(probe['annotations'])}"
            )
        if category_ids != EXPECTED_CATEGORY_IDS:
            raise ValueError(f"Expected category IDs {EXPECTED_CATEGORY_IDS}, got {category_ids}")


def dry_run(args: argparse.Namespace) -> int:
    checkpoint = Path(args.checkpoint)
    probe_path = Path(args.probe)
    images_root = Path(args.images_root)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if not probe_path.is_file():
        raise FileNotFoundError(f"Probe not found: {probe_path}")
    if not images_root.is_dir():
        raise FileNotFoundError(f"Images root not found: {images_root}")

    if args.allow_any_hash:
        checkpoint_sha256 = sha256_file(checkpoint)
        probe_sha256 = sha256_file(probe_path)
    else:
        checkpoint_sha256 = assert_file_sha256(checkpoint, EXPECTED_CHECKPOINT_SHA256)
        probe_sha256 = assert_file_sha256(probe_path, EXPECTED_PROBE_SHA256)

    probe = load_probe(probe_path)
    validate_probe_schema(probe, strict_frozen_probe=not args.allow_any_hash)

    print("DRY_RUN_OK")
    print(f"checkpoint={checkpoint}")
    print(f"checkpoint_sha256={checkpoint_sha256}")
    print(f"probe={probe_path}")
    print(f"probe_sha256={probe_sha256}")
    print(f"probe_images={len(probe['images'])}")
    print(f"probe_annotations={len(probe['annotations'])}")
    print(f"category_ids={sorted(category['id'] for category in probe['categories'])}")
    return 0


def main() -> int:
    args = parse_args()
    if args.dry_run:
        return dry_run(args)
    raise SystemExit("Full E1 detector inference is outside this code-only implementation.")


if __name__ == "__main__":
    raise SystemExit(main())
