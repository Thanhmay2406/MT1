from __future__ import annotations

import hashlib
import random
import copy
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any
from weakref import WeakSet


_ACTIVE_MASKS = WeakSet()
LOSS_KEYS = ("loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg")


@dataclass(frozen=True)
class InterventionSpec:
    canonical_id: str
    group_id: str
    norm_name: str
    channel_index: int


class ModelStateGuard:
    """Keep an exact device-local snapshot for cheap per-forward checks."""

    def __init__(self, model: Any):
        import torch

        self._parameter_names = tuple(name for name, _ in model.named_parameters())
        self._buffer_names = tuple(name for name, _ in model.named_buffers())
        self._parameters = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
        }
        self._buffers = {
            name: buffer.detach().clone()
            for name, buffer in model.named_buffers()
        }
        self._training = {name: module.training for name, module in model.named_modules()}
        self._requires_grad = {name: p.requires_grad for name, p in model.named_parameters()}
        self._grads = {name: None if p.grad is None else p.grad.detach().clone() for name, p in model.named_parameters()}
        self._hooks = {name: {key: dict(value) for key, value in vars(module).items()
                            if "hook" in key and isinstance(value, dict)} for name, module in model.named_modules()}
        self.digest = model_state_digest(model)
        self._torch = torch

    def verify(self, model: Any, *, check_gradients=True) -> None:
        parameters = dict(model.named_parameters())
        buffers = dict(model.named_buffers())
        if tuple(parameters) != self._parameter_names or tuple(buffers) != self._buffer_names:
            raise RuntimeError("Model parameter or buffer topology changed during intervention")
        for name, reference in self._parameters.items():
            if not self._torch.equal(parameters[name].detach(), reference):
                raise RuntimeError(f"Model parameter changed during intervention: {name}")
        for name, reference in self._buffers.items():
            if not self._torch.equal(buffers[name].detach(), reference):
                raise RuntimeError(f"Model buffer changed during intervention: {name}")
        training = {name: module.training for name, module in model.named_modules()}
        if training != self._training:
            raise RuntimeError("Model module training flags changed during intervention")
        for name, parameter in parameters.items():
            if parameter.requires_grad != self._requires_grad[name]:
                raise RuntimeError(f"requires_grad changed: {name}")
            if check_gradients:
                expected = self._grads[name]
                if (expected is None) != (parameter.grad is None) or (expected is not None and not self._torch.equal(expected, parameter.grad)):
                    raise RuntimeError(f"Gradient state changed: {name}")
        hooks = {name: {key: dict(value) for key, value in vars(module).items()
                       if "hook" in key and isinstance(value, dict)} for name, module in model.named_modules()}
        if hooks != self._hooks:
            raise RuntimeError("Model hook state changed during intervention")

    def restore(self, model):
        with self._torch.no_grad():
            for name, p in model.named_parameters():
                p.copy_(self._parameters[name])
                p.requires_grad_(self._requires_grad[name])
                p.grad = None if self._grads[name] is None else self._grads[name].clone()
            for name, b in model.named_buffers():
                b.copy_(self._buffers[name])
        for name, module in model.named_modules():
            module.training = self._training[name]
            for key, hooks in self._hooks[name].items():
                getattr(module, key).clear()
                getattr(module, key).update(hooks)


def snapshot_rng() -> dict[str, Any]:
    import numpy as np
    import torch

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state().clone(),
        "cuda": [state.clone() for state in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else None,
    }


def restore_rng(state: dict[str, Any]) -> None:
    import numpy as np
    import torch

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def rng_states_equal(a, b):
    import numpy as np
    import torch
    return (a["python"] == b["python"] and a["numpy"][0] == b["numpy"][0]
            and np.array_equal(a["numpy"][1], b["numpy"][1]) and a["numpy"][2:] == b["numpy"][2:]
            and torch.equal(a["torch"], b["torch"])
            and ((a["cuda"] is None and b["cuda"] is None) or
                 (a["cuda"] is not None and b["cuda"] is not None and len(a["cuda"]) == len(b["cuda"]) and
                  all(torch.equal(x, y) for x, y in zip(a["cuda"], b["cuda"])))))


def rng_identity(state):
    digest = hashlib.sha256()
    digest.update(repr(state["python"]).encode())
    digest.update(str(state["numpy"][0]).encode())
    digest.update(state["numpy"][1].tobytes())
    digest.update(repr(state["numpy"][2:]).encode())
    digest.update(state["torch"].cpu().numpy().tobytes())
    if state["cuda"] is not None:
        for value in state["cuda"]:
            digest.update(value.cpu().numpy().tobytes())
    return digest.hexdigest()


def validate_loss_components(losses):
    import torch
    if set(losses) != set(LOSS_KEYS):
        raise ValueError("Detector loss must contain exactly the four unweighted components")
    for key in LOSS_KEYS:
        value = losses[key]
        if not torch.is_tensor(value) or value.shape != () or value.dtype != torch.float32 or not torch.isfinite(value):
            raise ValueError(f"Loss component must be a finite float32 scalar: {key}")


@contextmanager
def loss_scoring_context(model, *, allow_gradient_updates=False):
    import torch
    outer = ModelStateGuard(model)
    rng = snapshot_rng()
    if model in _ACTIVE_MASKS:
        raise RuntimeError("Cannot enter loss context with an active intervention")
    try:
        transform = getattr(model, "transform", None)
        if transform is not None and (tuple(transform.min_size) != (800,) or transform.max_size != 1333):
            raise RuntimeError("Loss transform must use fixed 800/1333 resize")
        model.train()
        for module in model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()
        configured = ModelStateGuard(model)
        try:
            yield configured
        finally:
            configured.verify(model, check_gradients=not allow_gradient_updates)
    finally:
        outer.restore(model)
        restore_rng(rng)
        outer.verify(model)
        if not rng_states_equal(rng, snapshot_rng()):
            raise RuntimeError("Loss context RNG restoration failed")


def paired_loss_diagnostic(model, image, target, spec, *, forward_counts=None):
    import torch
    if image.dtype != torch.float32 or any(p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError("Loss diagnostic requires float32")
    counts = forward_counts if forward_counts is not None else {"loss_original": 0, "loss_intervened": 0}
    with loss_scoring_context(model) as guard:
        paired_rng = snapshot_rng()
        with torch.no_grad(), torch.autocast(device_type=image.device.type, enabled=False):
            counts["loss_original"] += 1
            original = model([image.clone()], [copy.deepcopy(target)])
            validate_loss_components(original)
            guard.verify(model)
            restore_rng(paired_rng)
            with post_bn_channel_mask(model, spec):
                counts["loss_intervened"] += 1
                intervened = model([image.clone()], [copy.deepcopy(target)])
            validate_loss_components(intervened)
            guard.verify(model)
        deltas = {key: float(intervened[key] - original[key]) for key in LOSS_KEYS}
        return {"status": "ok", "scope": "eligible_images_full_gt", "components": deltas,
                "original": {key: float(original[key]) for key in LOSS_KEYS},
                "intervened": {key: float(intervened[key]) for key in LOSS_KEYS},
                "total_delta": sum(deltas.values()), "rng_coupling": "common_initial_state",
                "forward_counts": {key: counts[key] for key in ("loss_original", "loss_intervened")}}


def model_state_digest(model: Any) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        value = parameter.detach().cpu().contiguous()
        digest.update(b"parameter\0")
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(repr(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    for name, buffer in model.named_buffers():
        value = buffer.detach().cpu().contiguous()
        digest.update(b"buffer\0")
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(repr(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    for name, module in model.named_modules():
        digest.update(b"training\0")
        digest.update(name.encode())
        digest.update(b"1" if module.training else b"0")
    return digest.hexdigest()


def _assert_eval_mode(model: Any) -> None:
    if model.training or any(module.training for module in model.modules()):
        raise RuntimeError("E3 intervention requires model and all modules in eval mode")


@contextmanager
def post_bn_channel_mask(model: Any, spec: InterventionSpec):
    if model in _ACTIVE_MASKS:
        raise RuntimeError("An intervention mask is already active on this model")
    modules = dict(model.named_modules())
    if spec.norm_name not in modules:
        raise KeyError(f"Post-BN module not found: {spec.norm_name}")
    calls = 0

    def mask(_module, _inputs, output):
        nonlocal calls
        import torch

        calls += 1
        if not torch.is_tensor(output) or output.ndim != 4:
            raise TypeError("Post-BN output must be a 4D tensor")
        if spec.channel_index < 0 or spec.channel_index >= output.shape[1]:
            raise ValueError(f"Channel index {spec.channel_index} outside post-BN width {output.shape[1]}")
        masked = output.clone()
        masked[:, spec.channel_index, :, :] = 0
        return masked

    handle = modules[spec.norm_name].register_forward_hook(mask)
    _ACTIVE_MASKS.add(model)
    failed = False
    try:
        yield
    except BaseException:
        failed = True
        raise
    finally:
        handle.remove()
        _ACTIVE_MASKS.discard(model)
        if not failed and calls != 1:
            raise RuntimeError("Intervention hook must fire exactly once in one forward")


def run_intervened_forward(model: Any, image: Any, spec: InterventionSpec, state_guard: ModelStateGuard | None = None) -> Any:
    import torch

    _assert_eval_mode(model)
    state_guard = state_guard or ModelStateGuard(model)
    state_guard.verify(model)
    rng_state = snapshot_rng()
    try:
        with torch.inference_mode(), post_bn_channel_mask(model, spec):
            prediction = model([image])[0]
    finally:
        restore_rng(rng_state)
        if not rng_states_equal(rng_state, snapshot_rng()):
            raise RuntimeError("Intervention RNG restoration failed")
        try:
            state_guard.verify(model)
        except RuntimeError:
            state_guard.restore(model)
            raise
    return prediction
