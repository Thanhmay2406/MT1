from __future__ import annotations

import hashlib
import json
import math
import platform
import sys
from pathlib import Path
from typing import Any


E6_SCHEMA_VERSION = "causal_audit_e6_reproducibility/v1"
E6_DEFAULT_ATOL = 1e-6


def canonical_json_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def compare_json(reference: Any, candidate: Any, *, atol: float, rtol: float) -> dict:
    if atol < 0 or rtol < 0 or not math.isfinite(atol) or not math.isfinite(rtol):
        raise ValueError("Comparison tolerances must be finite and non-negative")
    mismatches: list[dict] = []
    compared = 0

    def visit(left: Any, right: Any, path: str) -> None:
        nonlocal compared
        if _is_number(left) and _is_number(right):
            compared += 1
            if isinstance(left, float) or isinstance(right, float):
                if not math.isfinite(float(left)) or not math.isfinite(float(right)):
                    if left != right:
                        mismatches.append({"path": path, "reference": left, "candidate": right, "type": "non_finite"})
                elif not math.isclose(float(left), float(right), rel_tol=rtol, abs_tol=atol):
                    mismatches.append({"path": path, "reference": left, "candidate": right, "type": "numeric"})
            elif left != right:
                mismatches.append({"path": path, "reference": left, "candidate": right, "type": "integer"})
            return
        if type(left) is not type(right):
            mismatches.append({"path": path, "reference": left, "candidate": right, "type": "type"})
            return
        if isinstance(left, dict):
            compared += 1
            left_keys, right_keys = set(left), set(right)
            for key in sorted(left_keys - right_keys):
                mismatches.append({"path": f"{path}.{key}", "reference": left[key], "candidate": None, "type": "missing"})
            for key in sorted(right_keys - left_keys):
                mismatches.append({"path": f"{path}.{key}", "reference": None, "candidate": right[key], "type": "extra"})
            for key in sorted(left_keys & right_keys):
                visit(left[key], right[key], f"{path}.{key}")
            return
        if isinstance(left, list):
            compared += 1
            if len(left) != len(right):
                mismatches.append({"path": path, "reference": len(left), "candidate": len(right), "type": "length"})
            for index, (left_item, right_item) in enumerate(zip(left, right)):
                visit(left_item, right_item, f"{path}[{index}]")
            return
        compared += 1
        if left != right:
            mismatches.append({"path": path, "reference": left, "candidate": right, "type": "value"})

    visit(reference, candidate, "$" )
    return {"passed": not mismatches, "mismatch_count": len(mismatches), "mismatches": mismatches, "compared_fields": compared}


_DOWNSTREAM_PROVENANCE_FIELDS = {
    "e2": ("eligibility_sha256",),
    "e3": ("eligibility_sha256", "importance_sha256"),
    "e4": ("eligibility_sha256", "importance_sha256", "intervention_sha256", "e4_sample_sha256"),
    "e5": ("eligibility_sha256", "importance_sha256", "intervention_sha256", "equivalence_sha256"),
}


def normalize_stage_artifact(stage: str, candidate: dict, reference: dict) -> dict:
    normalized = dict(candidate)
    for field in _DOWNSTREAM_PROVENANCE_FIELDS.get(stage, ()):
        if field in reference and field in normalized:
            normalized[field] = reference[field]
    return normalized


def compare_stage_artifact(stage: str, reference: dict, candidate: dict, *, atol: float, rtol: float) -> dict:
    result = compare_json(reference, normalize_stage_artifact(stage, candidate, reference), atol=atol, rtol=rtol)
    return {"stage": stage, **result}


def source_identity(repo_root: str | Path) -> str:
    root = Path(repo_root)
    paths = sorted([*root.glob("src/*.py"), *(root / "scripts").glob("e[1-5]_*.py")])
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def runtime_fingerprint(*, repo_root: str | Path, device: str) -> dict:
    import torch
    import torchvision
    from PIL import __version__ as pillow_version

    resolved_device = str(device)
    cuda_name = None
    if resolved_device.startswith("cuda") and torch.cuda.is_available():
        cuda_name = torch.cuda.get_device_name(torch.device(resolved_device))
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": getattr(torch, "__version__", None),
        "torchvision": getattr(torchvision, "__version__", None),
        "pillow": pillow_version,
        "cuda_available": bool(torch.cuda.is_available()),
        "device": resolved_device,
        "cuda_device_name": cuda_name,
        "source_identity_sha256": source_identity(repo_root),
    }


def environment_gate(current: dict, manifest: dict | None) -> dict:
    if manifest is None:
        return {"verified": False, "status": "unverified", "reason": "missing_reference_environment_identity"}
    expected = manifest.get("fingerprint", manifest)
    comparison = compare_json(expected, current, atol=0.0, rtol=0.0)
    return {"verified": comparison["passed"], "status": "verified" if comparison["passed"] else "mismatch", "comparison": comparison}


def build_e6_artifact(
    *,
    hashes: dict,
    reference_artifact_hashes: dict,
    rerun_artifact_hashes: dict,
    stage_comparisons: list[dict],
    environment: dict,
    configuration: dict,
    payload_reproducible: bool,
    environment_provenance_verified: bool,
    atol: float,
    runtime: dict,
) -> dict:
    structural_pass = all(item["passed"] for item in stage_comparisons)
    reproducibility_pass = structural_pass and payload_reproducible and environment_provenance_verified
    return {
        "schema_version": E6_SCHEMA_VERSION,
        **hashes,
        "reference_artifact_hashes": reference_artifact_hashes,
        "rerun_artifact_hashes": rerun_artifact_hashes,
        "stage_comparisons": stage_comparisons,
        "environment": environment,
        "configuration": configuration,
        "tolerance": {"atol": float(atol), "rtol": 0.0},
        "payload_reproducible": bool(payload_reproducible),
        "environment_provenance_verified": bool(environment_provenance_verified),
        "reproducibility_pass": bool(reproducibility_pass),
        "complete": True,
        "mismatch_count": sum(int(item["mismatch_count"]) for item in stage_comparisons),
        "runtime": runtime,
    }
