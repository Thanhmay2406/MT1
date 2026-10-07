from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import sys
import subprocess
from pathlib import Path
from typing import Any


E6_SCHEMA_VERSION = "causal_audit_e6_reproducibility/v2"
E6_DEFAULT_ATOL = 0.0


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
        if type(left) is not type(right):
            mismatches.append({"path": path, "reference": left, "candidate": right, "type": "type"})
            return
        if _is_number(left) and _is_number(right):
            compared += 1
            if isinstance(left, float) or isinstance(right, float):
                if not math.isfinite(float(left)) or not math.isfinite(float(right)):
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
    if atol != 0 or rtol != 0:
        raise ValueError("Stage reproducibility requires exact canonical equality")
    normalized = normalize_stage_artifact(stage, candidate, reference)
    result = compare_json(reference, normalized, atol=0., rtol=0.)
    equal = canonical_json_hash(reference) == canonical_json_hash(normalized)
    return {"stage": stage, **result, "passed": result["passed"] and equal,
            "canonical_payload_equal": equal, "reference_payload_sha256": canonical_json_hash(reference),
            "candidate_payload_sha256": canonical_json_hash(normalized)}


def source_identity(repo_root: str | Path) -> str:
    root = Path(repo_root)
    paths = sorted([*root.glob("src/*.py"), *(root / "scripts").glob("e[1-6]_*.py"),
                    *[(root / "scripts" / name) for name in ("create_dataset_manifest.py", "create_e4_sample.py", "create_environment_manifest.py")],
                    *root.glob("requirements*.txt")])
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
    import importlib.metadata
    from determinism import determinism_metadata

    resolved_device = str(device)
    cuda_name = None
    package_lock = sorted((dist.metadata["Name"].lower(), dist.version) for dist in importlib.metadata.distributions() if dist.metadata["Name"])
    driver = None
    if resolved_device.startswith("cuda") and torch.cuda.is_available():
        cuda_name = torch.cuda.get_device_name(torch.device(resolved_device))
        try:
            driver = subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True, timeout=5).strip().splitlines()
        except (OSError, subprocess.SubprocessError):
            driver = "unverified"
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "torchvision": str(torchvision.__version__),
        "pillow": pillow_version,
        "cuda_available": bool(torch.cuda.is_available()),
        "device": resolved_device,
        "cuda_device_name": cuda_name,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "source_identity_sha256": source_identity(repo_root),
        "cuda_runtime": torch.version.cuda,
        "cuda_driver": driver,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cudnn": torch.backends.cudnn.version(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_devices": [{"name": torch.cuda.get_device_properties(i).name,
                          "uuid": str(getattr(torch.cuda.get_device_properties(i), "uuid", None)),
                          "capability": list(torch.cuda.get_device_capability(i))} for i in range(torch.cuda.device_count())],
        "package_versions": {name: importlib.metadata.version(name) for name in ("numpy", "torch", "torchvision", "pillow", "munkres")},
        "package_lock": [{"name": name, "version": version} for name, version in package_lock],
        "package_lock_sha256": canonical_json_hash(package_lock),
        "precision": {"default_dtype": str(torch.get_default_dtype()), "matmul_precision": torch.get_float32_matmul_precision(),
                      "cuda_matmul_tf32": torch.backends.cuda.matmul.allow_tf32, "cudnn_tf32": torch.backends.cudnn.allow_tf32,
                      "cpu_autocast": torch.is_autocast_enabled("cpu"), "cuda_autocast": torch.is_autocast_enabled("cuda")},
        "determinism": determinism_metadata(device=resolved_device),
        "documentation_identities": {str(p.relative_to(repo_root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(repo_root).glob("docs/**/*.md"))},
    }


def environment_gate(current: dict, manifest: dict | None) -> dict:
    if manifest is None:
        return {"verified": False, "status": "unverified", "reason": "missing_reference_environment_identity"}
    expected = manifest.get("fingerprint", manifest)
    if "manifest_sha256" in manifest and manifest["manifest_sha256"] != canonical_json_hash({k: v for k, v in manifest.items() if k != "manifest_sha256"}):
        return {"verified": False, "status": "unverified", "reason": "reference_environment_self_hash_mismatch"}
    if not current.get("determinism", {}).get("pythonhashseed_startup_verified"):
        return {"verified": False, "status": "unverified", "reason": "python_hash_seed_not_verified_at_startup"}
    if current.get("cuda_driver") == "unverified":
        return {"verified": False, "status": "unverified", "reason": "cuda_driver_unverified"}
    comparison = compare_json(expected, current, atol=0.0, rtol=0.0)
    return {"verified": comparison["passed"], "status": "verified" if comparison["passed"] else "mismatch", "comparison": comparison}


def scientific_environment_gate(current, manifest):
    identity = environment_gate(current, manifest)
    if not identity["verified"]:
        return identity
    reference = (str(current.get("python", "")).startswith("3.12.13")
                 and current.get("torch") == "2.10.0+cu128"
                 and current.get("torchvision") == "0.25.0+cu128"
                 and current.get("cuda_runtime") == "12.8"
                 and current.get("cuda_device_count") == 1
                 and str(current.get("device", "")).startswith("cuda")
                 and current.get("cuda_device_name") == "Tesla T4")
    devices = current.get("cuda_devices", [])
    reference = reference and len(devices) == 1 and devices[0].get("uuid") not in (None, "None", "", "unverified")
    reference = reference and isinstance(current.get("cuda_driver"), list) and bool(current["cuda_driver"])
    return {**identity, "verified": bool(reference), "status": "verified" if reference else "unverified",
            "reason": None if reference else "not_reference_scientific_stack", "environment_identity_matches": True}


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
    protocol_amendment_id: str | None = None,
) -> dict:
    if atol != 0:
        raise ValueError("E6 requires an exact payload comparison")
    structural_pass = {item["stage"] for item in stage_comparisons} == {"e1", "e2", "e3", "e4", "e5"} and len(stage_comparisons) == 5 and all(item["passed"] for item in stage_comparisons)
    reproducibility_pass = structural_pass and payload_reproducible and environment_provenance_verified
    artifact = {
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
    if protocol_amendment_id is not None:
        artifact["protocol_amendment_id"] = protocol_amendment_id
    return artifact
