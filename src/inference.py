from __future__ import annotations

from pathlib import Path
from typing import Mapping

from matching import Detection


def image_to_tensor(image):
    import torchvision.transforms.functional as F

    image = image.convert("RGB")
    return F.pil_to_tensor(image).float().div(255.0)


def _as_list(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def detection_from_prediction(
    prediction: Mapping[str, object],
    prediction_index: int,
    label_to_category_id: Mapping[int, int],
) -> Detection:
    boxes = _as_list(prediction["boxes"])
    labels = _as_list(prediction["labels"])
    scores = _as_list(prediction["scores"])
    x1, y1, x2, y2 = [float(value) for value in boxes[prediction_index]]
    label = int(labels[prediction_index])
    if label not in label_to_category_id:
        raise ValueError(f"detector emitted unknown label {label}")
    return Detection(
        index=prediction_index,
        category_id=int(label_to_category_id[label]),
        bbox=[x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
        score=float(scores[prediction_index]),
    )


def run_detector(model, image, device: str, label_to_category_id: Mapping[int, int]) -> list[Detection]:
    import torch

    tensor = image_to_tensor(image).to(torch.device(device))
    with torch.no_grad():
        prediction = model([tensor])[0]
    boxes = _as_list(prediction["boxes"])
    return [
        detection_from_prediction(prediction, prediction_index, label_to_category_id)
        for prediction_index in range(len(boxes))
    ]


def predict_probe(model, probe: dict, images_root: str | Path, device: str, label_to_category_id):
    from PIL import Image

    images_root = Path(images_root)
    predictions_by_image_id: dict[int, list[Detection]] = {}
    for image in probe["images"]:
        image_path = images_root / image["file_name"]
        with Image.open(image_path) as opened:
            predictions_by_image_id[int(image["id"])] = run_detector(
                model,
                opened,
                device=device,
                label_to_category_id=label_to_category_id,
            )
    return predictions_by_image_id
