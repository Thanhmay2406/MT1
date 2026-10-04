from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from artifacts import build_e2_artifact, percentile_ranks_by_group, read_json_artifact, write_json_artifact
from integrity import assert_file_sha256, sha256_file


EXPECTED_CHECKPOINT_SHA256 = "953a2b8d8e412227a89b9dd42c0899b33de28f281110af54829ec531db16dda0"
EXPECTED_PROBE_SHA256 = "1016ed7eacda87c6b368c880a5564799e22e7fa757e11cd9cf93845b396d811d"
EXPECTED_E1_SCHEMA = "causal_audit_e1_eligibility/v1"
EXPECTED_PROBE_IMAGES = 300
EXPECTED_PROBE_ANNOTATIONS = 372
EXPECTED_CATEGORIES = [0, 1, 2, 3, 4, 5]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="E2 detector-conditioned importance scoring")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--probe", required=True)
    parser.add_argument("--images-root", required=True)
    parser.add_argument("--eligibility", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_probe(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        probe = json.load(handle)
    for key in ("images", "annotations", "categories"):
        if not isinstance(probe.get(key), list):
            raise ValueError(f"Probe must contain list field {key!r}")
    return probe


def validate_probe(probe: dict) -> None:
    if len(probe["images"]) != EXPECTED_PROBE_IMAGES:
        raise ValueError(f"Expected {EXPECTED_PROBE_IMAGES} probe images")
    if len(probe["annotations"]) != EXPECTED_PROBE_ANNOTATIONS:
        raise ValueError(f"Expected {EXPECTED_PROBE_ANNOTATIONS} probe annotations")
    category_ids = sorted(int(category["id"]) for category in probe["categories"])
    if category_ids != EXPECTED_CATEGORIES:
        raise ValueError(f"Expected category IDs {EXPECTED_CATEGORIES}, got {category_ids}")
    image_ids = [int(image["id"]) for image in probe["images"]]
    if len(image_ids) != len(set(image_ids)):
        raise ValueError("Probe image IDs must be unique")
    known_ids = set(image_ids)
    for annotation in probe["annotations"]:
        if int(annotation["image_id"]) not in known_ids:
            raise ValueError(f"Unknown annotation image_id: {annotation['image_id']}")
        if len(annotation.get("bbox", [])) != 4:
            raise ValueError(f"Invalid annotation bbox: {annotation.get('id')}")


def validate_image_files(probe: dict, images_root: Path) -> None:
    missing = [
        str(images_root / image["file_name"])
        for image in probe["images"]
        if not (images_root / image["file_name"]).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} probe image(s): {', '.join(missing[:3])}")


def validate_eligibility(eligibility: dict, probe: dict, checkpoint_sha256: str, probe_sha256: str) -> None:
    if eligibility.get("schema_version") != EXPECTED_E1_SCHEMA:
        raise ValueError("Unsupported E1 eligibility schema")
    if eligibility.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError("E1 checkpoint hash does not match requested checkpoint")
    if eligibility.get("probe_sha256") != probe_sha256:
        raise ValueError("E1 probe hash does not match requested probe")
    if float(eligibility.get("matching_iou_threshold", -1.0)) != 0.5:
        raise ValueError("E1 matching IoU threshold must remain 0.5")
    if eligibility.get("image_count") != len(probe["images"]):
        raise ValueError("E1 image count does not match probe")
    if len(eligibility.get("images", [])) != len(probe["images"]):
        raise ValueError("E1 image rows are incomplete")
    eligible_images = 0
    eligible_instances = 0
    for row, image in zip(eligibility["images"], probe["images"]):
        if row.get("image_id") != image["id"] or row.get("file_name") != image["file_name"]:
            raise ValueError(f"E1 image order mismatch at image {image['id']}")
        matches = row.get("matches", [])
        if row.get("eligible_gt_ids") != [match["gt_id"] for match in matches]:
            raise ValueError(f"E1 eligible IDs mismatch at image {image['id']}")
        if len({match["gt_id"] for match in matches}) != len(matches):
            raise ValueError(f"E1 reuses a GT identity at image {image['id']}")
        if len({match["prediction_index"] for match in matches}) != len(matches):
            raise ValueError(f"E1 reuses a prediction within image {image['id']}")
        for match in matches:
            if match["gt_category_id"] != match["prediction_category_id"]:
                raise ValueError(f"E1 class mismatch at image {image['id']}")
            if float(match["iou"]) < 0.5:
                raise ValueError(f"E1 below-threshold match at image {image['id']}")
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
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return "cuda" if requested == "cuda" or torch.cuda.is_available() else "cpu"


def _target_for_image(probe: dict, image_id: int, device: str, category_id_to_label):
    import torch

    annotations = [annotation for annotation in probe["annotations"] if annotation["image_id"] == image_id]
    boxes = []
    labels = []
    areas = []
    for annotation in annotations:
        x, y, width, height = [float(value) for value in annotation["bbox"]]
        if width <= 0 or height <= 0:
            continue
        boxes.append([x, y, x + width, y + height])
        category_id = int(annotation["category_id"])
        if category_id not in category_id_to_label:
            raise ValueError(f"Unknown COCO category ID in probe: {category_id}")
        labels.append(int(category_id_to_label[category_id]))
        areas.append(width * height)
    return {
        "boxes": torch.tensor(boxes, dtype=torch.float32, device=device).reshape(-1, 4),
        "labels": torch.tensor(labels, dtype=torch.int64, device=device),
        "image_id": torch.tensor([image_id], dtype=torch.int64, device=device),
        "area": torch.tensor(areas, dtype=torch.float32, device=device),
        "iscrowd": torch.zeros(len(boxes), dtype=torch.int64, device=device),
    }


def _snapshot_rng():
    import torch

    state = {"python": random.getstate(), "torch": torch.get_rng_state()}
    try:
        import numpy as np

        state["numpy"] = np.random.get_state()
    except ImportError:
        pass
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state):
    import torch

    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if "numpy" in state:
        import numpy as np

        np.random.set_state(state["numpy"])
    if "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def _set_batchnorm_eval(model):
    import torch.nn as nn

    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def _detections_from_output(output, label_to_category_id):
    from inference import detection_from_prediction, _as_list

    return [
        detection_from_prediction(output, index, label_to_category_id)
        for index in range(len(_as_list(output["boxes"])))
    ]


def _check_match_identity(current_matches, frozen_row):
    expected = [match["gt_id"] for match in frozen_row["matches"]]
    actual = [match.gt_id for match in current_matches]
    if actual != expected:
        raise RuntimeError(f"E1 match identity changed for image {frozen_row['image_id']}")


def compute_scores(model, probe, eligibility, groups, images_root, device, label_to_category_id, category_id_to_label):
    import torch
    from PIL import Image

    from attribution import gxa_channel_scores
    from baselines import activation_channel_scores, l1_channel_scores, taylor_channel_scores
    from hooks import PostBNHookBank
    from matching import GroundTruthObject, match_detections_to_gt

    annotations_by_image = {}
    for annotation in probe["annotations"]:
        annotations_by_image.setdefault(int(annotation["image_id"]), []).append(annotation)
    gxa_images = {group.group_id: [] for group in groups}
    activation_images = {group.group_id: [] for group in groups}
    taylor_images = {group.group_id: [] for group in groups}
    l1_scores = {
        group.group_id: l1_channel_scores(group.producer_module.weight.detach()).cpu()
        for group in groups
    }
    norm_names = [group.norm_name for group in groups]
    eligible_rows = [row for row in eligibility["images"] if row["matches"]]

    for row in eligible_rows:
        image_id = int(row["image_id"])
        image_path = Path(images_root) / row["file_name"]
        with Image.open(image_path) as opened:
            from inference import image_to_tensor

            image = image_to_tensor(opened).to(device)
        gt_objects = [
            GroundTruthObject(int(annotation["id"]), int(annotation["category_id"]), annotation["bbox"])
            for annotation in annotations_by_image.get(image_id, [])
        ]
        model.eval()
        with PostBNHookBank(model, norm_names) as bank:
            model.zero_grad(set_to_none=True)
            output = model([image])[0]
            detections = _detections_from_output(output, label_to_category_id)
            current_matches = match_detections_to_gt(
                gt_objects,
                detections,
                iou_threshold=float(eligibility["matching_iou_threshold"]),
            )
            _check_match_identity(current_matches, row)
            for group in groups:
                activation_images[group.group_id].append(
                    activation_channel_scores(bank.activation(group.norm_name).detach()).cpu()
                )
            per_group_objects = {group.group_id: [] for group in groups}
            for match_index, match in enumerate(current_matches):
                model.zero_grad(set_to_none=True)
                bank.clear()
                output["scores"][match.prediction_index].backward(retain_graph=match_index < len(current_matches) - 1)
                for group in groups:
                    gradient = bank.activation(group.norm_name).grad
                    if gradient is None:
                        raise RuntimeError(f"Missing post-BN gradient for {group.norm_name}")
                    per_group_objects[group.group_id].append(
                        gxa_channel_scores(bank.activation(group.norm_name), gradient).detach().cpu()
                    )
            for group in groups:
                group_scores = per_group_objects[group.group_id]
                gxa_images[group.group_id].append(torch.stack(group_scores).mean(dim=0))

        model.train()
        _set_batchnorm_eval(model)
        rng_state = _snapshot_rng()
        try:
            target = _target_for_image(probe, image_id, device, category_id_to_label)
            model.zero_grad(set_to_none=True)
            losses = model([image], [target])
            loss = sum(losses.values())
            loss.backward()
            for group in groups:
                if group.producer_module.weight.grad is None:
                    raise RuntimeError(f"Missing Taylor gradient for {group.producer_name}")
                taylor_images[group.group_id].append(
                    taylor_channel_scores(group.producer_module.weight.detach(), group.producer_module.weight.grad.detach()).cpu()
                )
        finally:
            _restore_rng(rng_state)
            model.eval()

    rows = []
    for group in groups:
        gxa = torch.stack(gxa_images[group.group_id]).mean(dim=0)
        activation = torch.stack(activation_images[group.group_id]).mean(dim=0)
        taylor = torch.stack(taylor_images[group.group_id]).mean(dim=0)
        method_vectors = {"gxa": gxa, "activation": activation, "l1": l1_scores[group.group_id], "taylor": taylor}
        for identity in group.channels_identities:
            index = identity.local_channel_index
            rows.append({
                "canonical_id": identity.canonical_id,
                "group_id": identity.group_id,
                "stage": identity.stage,
                "block": identity.block,
                "hidden_conv": identity.hidden_conv,
                "local_channel_index": index,
                "scores": {method: {"raw": float(vector[index].item())} for method, vector in method_vectors.items()},
            })
    for method in ("gxa", "activation", "l1", "taylor"):
        ranked = percentile_ranks_by_group([
            {"group_id": row["group_id"], "raw": row["scores"][method]["raw"], "row_index": index}
            for index, row in enumerate(rows)
        ])
        for ranked_row in ranked:
            rows[ranked_row["row_index"]]["scores"][method]["percentile"] = ranked_row["percentile"]
    return rows


def dry_run(args: argparse.Namespace) -> int:
    checkpoint = Path(args.checkpoint)
    probe_path = Path(args.probe)
    images_root = Path(args.images_root)
    eligibility_path = Path(args.eligibility)
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists; pass --overwrite to replace it: {output}")
    if not checkpoint.is_file() or not probe_path.is_file() or not images_root.is_dir() or not eligibility_path.is_file():
        raise FileNotFoundError("Checkpoint, probe, images root, and E1 artifact are required")
    checkpoint_sha256 = assert_file_sha256(checkpoint, EXPECTED_CHECKPOINT_SHA256)
    probe_sha256 = assert_file_sha256(probe_path, EXPECTED_PROBE_SHA256)
    probe = load_probe(probe_path)
    validate_probe(probe)
    validate_image_files(probe, images_root)
    eligibility = read_json_artifact(eligibility_path)
    validate_eligibility(eligibility, probe, checkpoint_sha256, probe_sha256)
    print("DRY_RUN_OK")
    print(f"probe_images={len(probe['images'])}")
    print(f"eligible_images={eligibility['eligible_image_count']}")
    print(f"eligible_instances={eligibility['eligible_instance_count']}")
    print(f"eligibility_sha256={sha256_file(eligibility_path)}")
    return 0


def run_real(args: argparse.Namespace) -> int:
    from detector import load_detector_checkpoint
    from identities import discover_structural_groups

    checkpoint = Path(args.checkpoint)
    probe_path = Path(args.probe)
    images_root = Path(args.images_root)
    eligibility_path = Path(args.eligibility)
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists; pass --overwrite to replace it: {output}")
    checkpoint_sha256 = assert_file_sha256(checkpoint, EXPECTED_CHECKPOINT_SHA256)
    probe_sha256 = assert_file_sha256(probe_path, EXPECTED_PROBE_SHA256)
    probe = load_probe(probe_path)
    validate_probe(probe)
    validate_image_files(probe, images_root)
    eligibility = read_json_artifact(eligibility_path)
    validate_eligibility(eligibility, probe, checkpoint_sha256, probe_sha256)
    device = resolve_device(args.device)
    model, metadata = load_detector_checkpoint(checkpoint, device=device)
    groups = discover_structural_groups(model)
    channels = compute_scores(
        model,
        probe,
        eligibility,
        groups,
        images_root,
        device,
        metadata["label_to_category_id"],
        metadata["category_id_to_label"],
    )
    group_rows = [
        {
            "group_id": group.group_id,
            "stage": group.stage,
            "block": group.block,
            "hidden_conv": group.hidden_conv,
            "producer_name": group.producer_name,
            "norm_name": group.norm_name,
            "consumer_name": group.consumer_name,
            "channels": group.channels,
        }
        for group in groups
    ]
    artifact = build_e2_artifact(
        checkpoint_sha256=checkpoint_sha256,
        probe_sha256=probe_sha256,
        eligibility_sha256=sha256_file(eligibility_path),
        matching_iou_threshold=float(eligibility["matching_iou_threshold"]),
        groups=group_rows,
        channels=channels,
        image_count=eligibility["image_count"],
        eligible_image_count=eligibility["eligible_image_count"],
        eligible_instance_count=eligibility["eligible_instance_count"],
    )
    write_json_artifact(output, artifact)
    print("E2_OK")
    print(f"device={device}")
    print(f"group_count={len(groups)}")
    print(f"channel_count={len(channels)}")
    print(f"output={output}")
    return 0


def main() -> int:
    args = parse_args()
    if args.dry_run:
        return dry_run(args)
    return run_real(args)


if __name__ == "__main__":
    raise SystemExit(main())
