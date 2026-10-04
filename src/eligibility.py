from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from matching import Detection, GroundTruthObject, match_detections_to_gt


SCHEMA_VERSION = "causal_audit_e1_eligibility/v1"


@dataclass(frozen=True)
class EligibilityArtifact:
    schema_version: str
    checkpoint_sha256: str
    probe_sha256: str
    matching_iou_threshold: float
    image_count: int
    eligible_image_count: int
    eligible_instance_count: int
    images: list[dict]

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "checkpoint_sha256": self.checkpoint_sha256,
            "probe_sha256": self.probe_sha256,
            "matching_iou_threshold": self.matching_iou_threshold,
            "image_count": self.image_count,
            "eligible_image_count": self.eligible_image_count,
            "eligible_instance_count": self.eligible_instance_count,
            "images": self.images,
        }


def _gt_objects_for_image(probe: dict) -> dict[int, list[GroundTruthObject]]:
    by_image: dict[int, list[GroundTruthObject]] = {}
    for annotation in probe.get("annotations", []):
        image_id = int(annotation["image_id"])
        by_image.setdefault(image_id, []).append(
            GroundTruthObject(
                id=int(annotation["id"]),
                category_id=int(annotation["category_id"]),
                bbox=annotation["bbox"],
            )
        )
    return by_image


def build_eligibility_artifact(
    probe: dict,
    predictions_by_image_id: Mapping[int, Sequence[Detection]],
    checkpoint_sha256: str,
    probe_sha256: str,
    iou_threshold: float = 0.5,
) -> EligibilityArtifact:
    annotations_by_image = _gt_objects_for_image(probe)
    artifact_images: list[dict] = []
    eligible_image_count = 0
    eligible_instance_count = 0

    for image in probe.get("images", []):
        image_id = int(image["id"])
        gt_objects = annotations_by_image.get(image_id, [])
        detections = predictions_by_image_id.get(image_id, [])
        matches = match_detections_to_gt(gt_objects, detections, iou_threshold=iou_threshold)
        eligible_gt_ids = [match.gt_id for match in matches]
        if eligible_gt_ids:
            eligible_image_count += 1
            eligible_instance_count += len(eligible_gt_ids)
        artifact_images.append(
            {
                "image_id": image_id,
                "file_name": image.get("file_name"),
                "width": image.get("width"),
                "height": image.get("height"),
                "eligible_gt_ids": eligible_gt_ids,
                "matches": [match.to_dict() for match in matches],
            }
        )

    return EligibilityArtifact(
        schema_version=SCHEMA_VERSION,
        checkpoint_sha256=checkpoint_sha256,
        probe_sha256=probe_sha256,
        matching_iou_threshold=iou_threshold,
        image_count=len(probe.get("images", [])),
        eligible_image_count=eligible_image_count,
        eligible_instance_count=eligible_instance_count,
        images=artifact_images,
    )
