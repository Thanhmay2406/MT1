from __future__ import annotations


def activation_channel_scores(activation):
    if activation.ndim == 4:
        return activation.abs().mean(dim=(0, 2, 3))
    if activation.ndim == 3:
        return activation.abs().mean(dim=(1, 2))
    raise ValueError(f"Expected [N,C,H,W] or [C,H,W], got {tuple(activation.shape)}")


def l1_channel_scores(weight):
    if weight.ndim != 4:
        raise ValueError(f"Expected Conv2d weight [C,Cin,Kh,Kw], got {tuple(weight.shape)}")
    return weight.abs().sum(dim=(1, 2, 3))


def taylor_channel_scores(weight, gradient):
    if weight.shape != gradient.shape:
        raise ValueError("Weight and gradient shapes must match")
    return (weight * gradient).reshape(weight.shape[0], -1).sum(dim=1).abs()
