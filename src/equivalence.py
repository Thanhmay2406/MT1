from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Sequence


E4_SAMPLE_SCHEMA = "causal_audit_e4_equivalence_sample/v2"
E4_TOLERANCE = 1e-5


@dataclass(frozen=True)
class PhysicalRemovalSpec:
    canonical_id: str
    group_id: str
    producer_name: str
    norm_name: str
    consumer_name: str
    channel_index: int


def _parent_and_attribute(model: Any, module_name: str):
    if "." in module_name:
        parent_name, attribute = module_name.rsplit(".", 1)
        parent = dict(model.named_modules()).get(parent_name)
    else:
        parent, attribute = model, module_name
    if parent is None or not hasattr(parent, attribute):
        raise KeyError(f"Module parent not found: {module_name}")
    return parent, attribute


def _kept_indices(width: int, removed: int, device):
    import torch

    if removed < 0 or removed >= width:
        raise ValueError(f"Channel index {removed} outside width {width}")
    return torch.tensor(
        [index for index in range(width) if index != removed],
        dtype=torch.long,
        device=device,
    )


def _replace_producer(producer: Any, channel_index: int):
    import torch
    import torch.nn as nn

    if not isinstance(producer, nn.Conv2d):
        raise TypeError("Physical removal producer must be Conv2d")
    if producer.groups != 1:
        raise ValueError("Physical removal currently requires an ungrouped producer Conv2d")
    kept = _kept_indices(producer.out_channels, channel_index, producer.weight.device)
    replacement = nn.Conv2d(
        producer.in_channels,
        producer.out_channels - 1,
        producer.kernel_size,
        stride=producer.stride,
        padding=producer.padding,
        dilation=producer.dilation,
        groups=producer.groups,
        bias=producer.bias is not None,
        padding_mode=producer.padding_mode,
    ).to(device=producer.weight.device, dtype=producer.weight.dtype)
    with torch.no_grad():
        replacement.weight.copy_(producer.weight.index_select(0, kept))
        if producer.bias is not None:
            replacement.bias.copy_(producer.bias.index_select(0, kept))
    replacement.weight.requires_grad_(producer.weight.requires_grad)
    if replacement.bias is not None:
        replacement.bias.requires_grad_(producer.bias.requires_grad)
    return replacement


def _replace_norm(norm: Any, channel_index: int):
    import torch
    import torch.nn as nn

    if not isinstance(norm, nn.BatchNorm2d):
        raise TypeError("Physical removal normalization must be BatchNorm2d")
    norm_device = norm.weight.device if norm.affine else norm.running_mean.device
    kept = _kept_indices(norm.num_features, channel_index, norm_device)
    replacement = nn.BatchNorm2d(
        norm.num_features - 1,
        eps=norm.eps,
        momentum=norm.momentum,
        affine=norm.affine,
        track_running_stats=norm.track_running_stats,
    ).to(device=norm.weight.device if norm.affine else norm.running_mean.device, dtype=norm.weight.dtype if norm.affine else norm.running_mean.dtype)
    with torch.no_grad():
        if norm.affine:
            replacement.weight.copy_(norm.weight.index_select(0, kept))
            replacement.bias.copy_(norm.bias.index_select(0, kept))
        if norm.track_running_stats:
            replacement.running_mean.copy_(norm.running_mean.index_select(0, kept))
            replacement.running_var.copy_(norm.running_var.index_select(0, kept))
            replacement.num_batches_tracked.copy_(norm.num_batches_tracked)
    if norm.affine:
        replacement.weight.requires_grad_(norm.weight.requires_grad)
        replacement.bias.requires_grad_(norm.bias.requires_grad)
    return replacement


def _replace_consumer(consumer: Any, channel_index: int):
    import torch
    import torch.nn as nn

    if not isinstance(consumer, nn.Conv2d):
        raise TypeError("Physical removal consumer must be Conv2d")
    if consumer.groups != 1:
        raise ValueError("Physical removal currently requires an ungrouped consumer Conv2d")
    kept = _kept_indices(consumer.in_channels, channel_index, consumer.weight.device)
    replacement = nn.Conv2d(
        consumer.in_channels - 1,
        consumer.out_channels,
        consumer.kernel_size,
        stride=consumer.stride,
        padding=consumer.padding,
        dilation=consumer.dilation,
        groups=consumer.groups,
        bias=consumer.bias is not None,
        padding_mode=consumer.padding_mode,
    ).to(device=consumer.weight.device, dtype=consumer.weight.dtype)
    with torch.no_grad():
        replacement.weight.copy_(consumer.weight.index_select(1, kept))
        if consumer.bias is not None:
            replacement.bias.copy_(consumer.bias)
    replacement.weight.requires_grad_(consumer.weight.requires_grad)
    if replacement.bias is not None:
        replacement.bias.requires_grad_(consumer.bias.requires_grad)
    return replacement


def remove_structural_channel(model: Any, spec: PhysicalRemovalSpec) -> Any:
    """Remove one dependency-consistent producer/BN/consumer channel in-place."""
    modules = dict(model.named_modules())
    for name in (spec.producer_name, spec.norm_name, spec.consumer_name):
        if name not in modules:
            raise KeyError(f"Physical-removal module not found: {name}")
    producer = modules[spec.producer_name]
    norm = modules[spec.norm_name]
    consumer = modules[spec.consumer_name]
    if producer.out_channels != norm.num_features or producer.out_channels != consumer.in_channels:
        raise ValueError("Producer/BatchNorm/consumer channel widths do not match")
    if spec.channel_index < 0 or spec.channel_index >= producer.out_channels:
        raise ValueError(f"Channel index {spec.channel_index} outside structural width {producer.out_channels}")
    producer_replacement = _replace_producer(producer, spec.channel_index)
    norm_replacement = _replace_norm(norm, spec.channel_index)
    consumer_replacement = _replace_consumer(consumer, spec.channel_index)
    for name, replacement in (
        (spec.producer_name, producer_replacement),
        (spec.norm_name, norm_replacement),
        (spec.consumer_name, consumer_replacement),
    ):
        parent, attribute = _parent_and_attribute(model, name)
        setattr(parent, attribute, replacement)
    return model


def build_physical_removal_model(model: Any, spec: PhysicalRemovalSpec) -> Any:
    cloned = copy.deepcopy(model)
    remove_structural_channel(cloned, spec)
    cloned.eval()
    return cloned


def _damage_by_gt(rows: Sequence[dict]) -> dict[int, dict]:
    result = {}
    for row in rows:
        gt_id = int(row["gt_id"])
        if gt_id in result:
            raise ValueError(f"Duplicate GT identity in damage rows: {gt_id}")
        result[gt_id] = row
    return result


def compare_mask_and_physical(mask_damage: Sequence[dict], physical_damage: Sequence[dict], tolerance: float = E4_TOLERANCE) -> dict:
    if tolerance < 0 or not math.isfinite(tolerance):
        raise ValueError("Equivalence tolerance must be finite and non-negative")
    mask = _damage_by_gt(mask_damage)
    physical = _damage_by_gt(physical_damage)
    if set(mask) != set(physical):
        raise ValueError("Mask and physical damage GT identities do not match")
    comparisons = []
    max_utility_difference = 0.0
    max_damage_difference = 0.0
    for gt_id in sorted(mask):
        mask_row = mask[gt_id]
        physical_row = physical[gt_id]
        utility_difference = abs(float(mask_row["intervened_utility"]) - float(physical_row["intervened_utility"]))
        damage_difference = abs(float(mask_row["damage"]) - float(physical_row["damage"]))
        max_utility_difference = max(max_utility_difference, utility_difference)
        max_damage_difference = max(max_damage_difference, damage_difference)
        comparisons.append({
            "gt_id": gt_id,
            "mask_utility": float(mask_row["intervened_utility"]),
            "physical_utility": float(physical_row["intervened_utility"]),
            "mask_damage": float(mask_row["damage"]),
            "physical_damage": float(physical_row["damage"]),
            "utility_abs_difference": utility_difference,
            "damage_abs_difference": damage_difference,
        })
    return {
        "comparison_scope": "utility_damage_diagnostic_only",
        "full_endpoint_verification": False,
        "object_comparisons": comparisons,
        "max_utility_abs_difference": max_utility_difference,
        "max_damage_abs_difference": max_damage_difference,
        "equivalent": max_utility_difference <= tolerance and max_damage_difference <= tolerance,
        "utility_damage_equivalent": max_utility_difference <= tolerance and max_damage_difference <= tolerance,
        "tolerance": tolerance,
    }


def canonical_json_hash(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def select_e4_sample(entries: Sequence[dict], probe: dict):
    strata = {}
    for entry in entries:
        kind = entry.get("hidden_conv") or {"bottleneck_conv1_hidden": "conv1", "bottleneck_conv2_hidden": "conv2"}.get(entry.get("group_kind"))
        strata.setdefault((entry["stage"], kind), []).append(entry)
    if len(strata) != 8 or len({s for s, _ in strata}) != 4 or {c for _, c in strata} != {"conv1", "conv2"}:
        raise ValueError("E4 requires four stages times two hidden-convolution strata")
    if len(probe["images"]) < 16:
        raise ValueError("E4 requires at least 16 probe entries")
    chosen = [min(strata[key], key=lambda entry: hashlib.sha256(entry["canonical_channel_id"].encode("utf-8")).hexdigest()) for key in sorted(strata)]
    return chosen, probe["images"][:16]


def compare_equivalence_endpoints(mask_consumer, physical_consumer, mask_output, physical_output, mask_matches, physical_matches, keep_indices):
    import torch
    from inference import serialize_prediction

    # PyTorch 2.10 default float32 rule, explicit even on another test runtime.
    rtol, atol = 1.3e-6, 1e-5
    errors = {}
    def close(name, a, b):
        try:
            if a.dtype != torch.float32 or b.dtype != torch.float32:
                raise AssertionError("Continuous endpoints must be float32")
            if not torch.isfinite(a).all() or not torch.isfinite(b).all():
                raise AssertionError("Non-finite endpoint")
            torch.testing.assert_close(a, b, rtol=rtol, atol=atol)
            return True
        except AssertionError as error:
            errors[name] = str(error)
            return False
    consumer = close("consumer", mask_consumer, physical_consumer)
    boxes = close("boxes", mask_output["boxes"], physical_output["boxes"])
    scores = close("scores", mask_output["scores"], physical_output["scores"])
    def identities(matches):
        return [(int(m["gt_id"]), int(m["gt_category_id"]), int(m["prediction_index"]), int(m["prediction_category_id"])) for m in matches]
    discrete = (mask_output["labels"].dtype == physical_output["labels"].dtype == torch.int64
                and torch.equal(mask_output["labels"], physical_output["labels"])
                and identities(mask_matches) == identities(physical_matches))
    def tensor_identity(value):
        t = value.detach().cpu().contiguous()
        return {"shape": list(t.shape), "dtype": str(t.dtype), "sha256": hashlib.sha256(t.numpy().tobytes()).hexdigest()}
    return {"equivalent": consumer and boxes and scores and discrete,
            "full_endpoint_verification": True,
            "comparison_scope": "consumer_final_output_and_matching",
            "consumer_equivalent": consumer, "final_continuous_equivalent": boxes and scores,
            "discrete_equivalent": discrete, "endpoint_errors": errors,
            "closeness_rule": {"reference": "torch_2.10_default_float32", "rtol": rtol, "atol": atol},
            "keep_indices": list(keep_indices), "mask_consumer": tensor_identity(mask_consumer),
            "physical_consumer": tensor_identity(physical_consumer),
            "mask_output": serialize_prediction(mask_output), "physical_output": serialize_prediction(physical_output)}


def capture_consumer_forward(model, image, consumer_name, spec=None):
    import torch
    from intervention import ModelStateGuard, snapshot_rng, restore_rng, rng_states_equal, post_bn_channel_mask, _assert_eval_mode
    from contextlib import nullcontext
    _assert_eval_mode(model)
    guard = ModelStateGuard(model)
    rng = snapshot_rng()
    captured = []
    def capture(_module, _inputs, output):
        captured.append(output.detach().clone())
    handle = dict(model.named_modules())[consumer_name].register_forward_hook(capture)
    try:
        with torch.inference_mode(), post_bn_channel_mask(model, spec) if spec else nullcontext():
            output = model([image])[0]
    finally:
        handle.remove()
        restore_rng(rng)
        try:
            guard.verify(model)
        except RuntimeError:
            guard.restore(model)
            raise
        if not rng_states_equal(rng, snapshot_rng()):
            raise RuntimeError("Equivalence RNG restoration failed")
    if len(captured) != 1:
        raise RuntimeError("Consumer must execute exactly once")
    return output, captured[0]
