from __future__ import annotations

import os
import random
import json
import subprocess
import sys
from functools import lru_cache
from typing import Any


DETERMINISTIC_SEED = 20260905
CUBLAS_WORKSPACE_CONFIG = ":4096:8"
AOT_AUTOGRAD_DONATED_BUFFER = False
_STARTUP_HASHSEED = os.environ.get("PYTHONHASHSEED")


@lru_cache(maxsize=None)
def startup_hashseed_verified(seed):
    if _STARTUP_HASHSEED != str(seed):
        return False
    expression = "import json; print(json.dumps([hash('mt1_startup_seed'), hash(b'mt1_startup_seed_bytes')]))"
    env = {**os.environ, "PYTHONHASHSEED": str(seed)}
    try:
        expected = json.loads(subprocess.check_output([sys.executable, "-c", expression], env=env, text=True, timeout=5))
    except (OSError, subprocess.SubprocessError, ValueError):
        return False
    return expected == [hash("mt1_startup_seed"), hash(b"mt1_startup_seed_bytes")]


def configure_determinism(seed: int = DETERMINISTIC_SEED) -> dict[str, Any]:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (None, CUBLAS_WORKSPACE_CONFIG):
        raise RuntimeError("Conflicting CUBLAS_WORKSPACE_CONFIG")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", CUBLAS_WORKSPACE_CONFIG)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    import torch
    from torch._functorch import config as aot_config

    if not hasattr(aot_config, "donated_buffer"):
        raise RuntimeError("Torch does not expose the required retained-backward buffer policy")
    # Compiled ROIAlign must preserve saved buffers across per-object backward calls.
    aot_config.donated_buffer = AOT_AUTOGRAD_DONATED_BUFFER

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return determinism_metadata(seed, device=None)


def determinism_metadata(seed: int = DETERMINISTIC_SEED, device: str | None = None) -> dict[str, Any]:
    import torch
    from torch._functorch import config as aot_config

    return {
        "seed": int(seed),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "aot_autograd_donated_buffer": getattr(aot_config, "donated_buffer", None),
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "pythonhashseed_startup_verified": startup_hashseed_verified(seed),
        "pythonhashseed_verification": "startup_environment_and_hash_probe",
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "device": device,
    }
