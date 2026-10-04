from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, permutations
from typing import Sequence


@dataclass(frozen=True)
class GroundTruthObject:
    id: int
    category_id: int
    bbox: Sequence[float]


@dataclass(frozen=True)
class Detection:
    index: int
    category_id: int
    bbox: Sequence[float]
    score: float


@dataclass(frozen=True)
class MatchRecord:
    gt_id: int
    gt_category_id: int
    gt_bbox: list[float]
    prediction_index: int
    prediction_category_id: int
    prediction_bbox: list[float]
    prediction_score: float
    iou: float

    def to_dict(self) -> dict:
        return {
            "gt_id": self.gt_id,
            "gt_category_id": self.gt_category_id,
            "gt_bbox": self.gt_bbox,
            "prediction_index": self.prediction_index,
            "prediction_category_id": self.prediction_category_id,
            "prediction_bbox": self.prediction_bbox,
            "prediction_score": self.prediction_score,
            "iou": self.iou,
            "eligible": True,
        }


def bbox_iou_xywh(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, aw, ah = [float(x) for x in a]
    bx1, by1, bw, bh = [float(x) for x in b]
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = inter_w * inter_h
    union = aw * ah + bw * bh - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def match_detections_to_gt(
    gt_objects: Sequence[GroundTruthObject],
    detections: Sequence[Detection],
    iou_threshold: float = 0.5,
) -> list[MatchRecord]:
    valid_edges: dict[tuple[int, int], float] = {}
    gt_by_id = {item.id: item for item in gt_objects}
    det_by_index = {item.index: item for item in detections}
    gt_ids = sorted(gt_by_id)
    det_indices = sorted(det_by_index)

    for gt_id in gt_ids:
        gt = gt_by_id[gt_id]
        for det_index in det_indices:
            detection = det_by_index[det_index]
            if gt.category_id != detection.category_id:
                continue
            iou = bbox_iou_xywh(gt.bbox, detection.bbox)
            if iou >= iou_threshold:
                valid_edges[(gt_id, det_index)] = iou

    best_pairs: tuple[tuple[int, int], ...] = ()
    best_score: tuple[int, float, tuple[tuple[int, int], ...]] | None = None
    max_size = min(len(gt_ids), len(det_indices))

    for size in range(max_size + 1):
        for chosen_gt_ids in combinations(gt_ids, size):
            for chosen_det_indices in combinations(det_indices, size):
                for det_order in permutations(chosen_det_indices):
                    pairs = tuple(sorted(zip(chosen_gt_ids, det_order)))
                    if any(pair not in valid_edges for pair in pairs):
                        continue
                    total_iou = sum(valid_edges[pair] for pair in pairs)
                    score = (size, total_iou, tuple((-gt, -pred) for gt, pred in pairs))
                    if best_score is None or score > best_score:
                        best_score = score
                        best_pairs = pairs

    records: list[MatchRecord] = []
    for gt_id, det_index in best_pairs:
        gt = gt_by_id[gt_id]
        detection = det_by_index[det_index]
        records.append(
            MatchRecord(
                gt_id=gt.id,
                gt_category_id=gt.category_id,
                gt_bbox=[float(x) for x in gt.bbox],
                prediction_index=detection.index,
                prediction_category_id=detection.category_id,
                prediction_bbox=[float(x) for x in detection.bbox],
                prediction_score=float(detection.score),
                iou=valid_edges[(gt_id, det_index)],
            )
        )
    return records
