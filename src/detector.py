from __future__ import annotations

from pathlib import Path
from typing import Any


EXPECTED_CATEGORY_ID_TO_LABEL = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6}
EXPECTED_LABEL_TO_CATEGORY_ID = {value: key for key, value in EXPECTED_CATEGORY_ID_TO_LABEL.items()}


def build_detector(num_classes: int = 7):
    import torch
    from torchvision.models.detection import fasterrcnn_resnet50_fpn

    if num_classes != 7:
        raise ValueError(f"Frozen E1 detector requires num_classes=7, got {num_classes}")
    return fasterrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=None,
        num_classes=num_classes,
    )


def _validate_category_mapping(checkpoint: dict[str, Any]) -> None:
    if checkpoint.get("category_id_to_label") != EXPECTED_CATEGORY_ID_TO_LABEL:
        raise ValueError("checkpoint category_id_to_label does not match frozen mapping")
    if checkpoint.get("label_to_category_id") != EXPECTED_LABEL_TO_CATEGORY_ID:
        raise ValueError("checkpoint label_to_category_id does not match frozen mapping")


def load_detector_checkpoint(checkpoint_path: str | Path, device: str = "cpu"):
    import torch

    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("checkpoint must contain model_state_dict")
    _validate_category_mapping(checkpoint)

    model = build_detector(num_classes=7)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(torch.device(device))
    model.eval()
    return model, {
        "category_id_to_label": dict(EXPECTED_CATEGORY_ID_TO_LABEL),
        "label_to_category_id": dict(EXPECTED_LABEL_TO_CATEGORY_ID),
        "transform_min_size": tuple(model.transform.min_size),
        "transform_max_size": int(model.transform.max_size),
    }
