from __future__ import annotations

from collections.abc import Sequence


def _channel_spatial_mean(value):
    if value.ndim == 4:
        return value.mean(dim=(0, 2, 3))
    if value.ndim == 3:
        return value.mean(dim=(1, 2))
    raise ValueError(f"Expected [N,C,H,W] or [C,H,W], got {tuple(value.shape)}")


def gxa_channel_scores(activation, gradient):
    if activation.shape != gradient.shape:
        raise ValueError("Activation and gradient shapes must match")
    return _channel_spatial_mean((activation * gradient).abs())


def activation_channel_scores(activation):
    return _channel_spatial_mean(activation.abs())


def aggregate_image_channel_scores(image_object_scores: Sequence[Sequence[float]]):
    if not image_object_scores:
        raise ValueError("Cannot aggregate an empty eligible image set")
    image_means: list[list[float]] = []
    for object_scores in image_object_scores:
        if not object_scores:
            raise ValueError("Eligible image must contain at least one object")
        width = len(object_scores[0])
        image_means.append(
            [
                sum(float(scores[channel]) for scores in object_scores) / len(object_scores)
                for channel in range(width)
            ]
        )
    width = len(image_means[0])
    return [
        sum(image[channel] for image in image_means) / len(image_means)
        for channel in range(width)
    ]
