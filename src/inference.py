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


def serialize_prediction(prediction) -> dict:
    import torch

    required = {"boxes": torch.float32, "scores": torch.float32, "labels": torch.int64}
    count = len(prediction["scores"])
    result = {}
    for key, dtype in required.items():
        tensor = prediction[key].detach().cpu()
        expected_shape = (count, 4) if key == "boxes" else (count,)
        if tensor.dtype != dtype or tuple(tensor.shape) != expected_shape or not torch.isfinite(tensor).all():
            raise ValueError(f"Noncanonical detector output: {key}")
        result[key] = {"dtype": str(tensor.dtype), "shape": list(tensor.shape), "values": tensor.tolist()}
    return result


def assert_original_replay(prediction, frozen: dict) -> None:
    if serialize_prediction(prediction) != frozen:
        raise RuntimeError("Original detector output changed before backward")


def detections_from_serialized_prediction(serialized, label_to_category_id):
    prediction = {key: value["values"] for key, value in serialized.items()}
    return [detection_from_prediction(prediction, i, label_to_category_id) for i in range(len(prediction["scores"]))]


def target_for_image(probe, image_id, device, category_id_to_label):
    import torch
    annotations = [a for a in probe["annotations"] if int(a["image_id"]) == image_id]
    boxes, labels = [], []
    for a in annotations:
        x, y, w, h = map(float, a["bbox"])
        if w <= 0 or h <= 0:
            raise ValueError("Degenerate ground-truth box")
        boxes.append([x, y, x+w, y+h])
        labels.append(category_id_to_label[int(a["category_id"])])
    return {"boxes": torch.tensor(boxes, dtype=torch.float32, device=device).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.int64, device=device)}


def replay_metadata(model, *, repo_root, device):
    from reproducibility import runtime_fingerprint
    from intervention import model_state_digest
    from intervention import snapshot_rng, rng_identity
    return {"environment": runtime_fingerprint(repo_root=repo_root, device=device),
            "model_state": model_state_digest(model), "mode": "all_modules_eval",
            "dtype": "float32", "amp": False, "batch_size": 1,
            "rng_identity_sha256": rng_identity(snapshot_rng()),
            "transform": repr(getattr(model, "transform", None))}


def assert_replay_metadata(model, frozen, *, repo_root, device):
    if frozen is None or frozen != replay_metadata(model, repo_root=repo_root, device=device):
        raise RuntimeError("Original replay environment/model/transform/RNG mismatch")


def run_detector(model, image, device: str, label_to_category_id: Mapping[int, int], *, raw_outputs=None, image_id=None) -> list[Detection]:
    import torch

    tensor = image_to_tensor(image).to(torch.device(device))
    with torch.no_grad():
        prediction = model([tensor])[0]
    if raw_outputs is not None:
        raw_outputs[image_id] = serialize_prediction(prediction)
    boxes = _as_list(prediction["boxes"])
    return [
        detection_from_prediction(prediction, prediction_index, label_to_category_id)
        for prediction_index in range(len(boxes))
    ]


def predict_probe(model, probe: dict, images_root: str | Path, device: str, label_to_category_id, *, raw_outputs=None):
    from PIL import Image

    images_root = Path(images_root)
    predictions_by_image_id: dict[int, list[Detection]] = {}
    for image_index, image in enumerate(probe["images"], start=1):
        image_path = images_root / image["file_name"]
        with Image.open(image_path) as opened:
            predictions_by_image_id[int(image["id"])] = run_detector(
                model,
                opened,
                device=device,
                label_to_category_id=label_to_category_id,
                raw_outputs=raw_outputs,
                image_id=int(image["id"]),
            )
        if image_index % 10 == 0 or image_index == len(probe["images"]):
            print(f"E1_PROGRESS images={image_index}/{len(probe['images'])}", flush=True)
    return predictions_by_image_id
