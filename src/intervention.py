from __future__ import annotations

import hashlib
import random
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any


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
        self.digest = model_state_digest(model)
        self._torch = torch

    def verify(self, model: Any) -> None:
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
    try:
        yield
    finally:
        handle.remove()
        if calls > 1:
            raise RuntimeError("Intervention hook fired more than once in one forward")


def run_intervened_forward(model: Any, image: Any, spec: InterventionSpec, state_guard: ModelStateGuard | None = None) -> Any:
    import torch

    _assert_eval_mode(model)
    if state_guard is None:
        before_digest = model_state_digest(model)
        before_training = {name: module.training for name, module in model.named_modules()}
    else:
        state_guard.verify(model)
        before_digest = state_guard.digest
    rng_state = snapshot_rng()
    try:
        with torch.inference_mode(), post_bn_channel_mask(model, spec):
            prediction = model([image])[0]
    finally:
        restore_rng(rng_state)
    if state_guard is None:
        after_training = {name: module.training for name, module in model.named_modules()}
        after_digest = model_state_digest(model)
        if before_training != after_training:
            raise RuntimeError("Model module training flags changed during intervention")
        if before_digest != after_digest:
            raise RuntimeError("Model parameters or buffers changed during intervention")
    else:
        state_guard.verify(model)
    return prediction
