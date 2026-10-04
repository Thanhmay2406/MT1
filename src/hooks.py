from __future__ import annotations

from typing import Any, Sequence


class PostBNHookBank:
    def __init__(self, model: Any, module_names: Sequence[str]):
        self.model = model
        self.module_names = list(module_names)
        self.activations: dict[str, Any] = {}
        self.handles: list[Any] = []

    def _hook(self, name: str):
        def capture(_module, _inputs, output):
            import torch

            if not torch.is_tensor(output) or output.ndim != 4:
                raise TypeError(f"Post-BN output must be a 4D tensor: {name}")
            self.activations[name] = output
            if output.requires_grad:
                output.retain_grad()

        return capture

    def __enter__(self):
        modules = dict(self.model.named_modules())
        for name in self.module_names:
            if name not in modules:
                raise KeyError(f"Post-BN module not found: {name}")
            self.handles.append(modules[name].register_forward_hook(self._hook(name)))
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.activations.clear()

    def clear(self) -> None:
        for activation in self.activations.values():
            activation.grad = None

    def activation(self, module_name: str):
        if module_name not in self.activations:
            raise KeyError(f"No captured activation for {module_name}")
        return self.activations[module_name]
