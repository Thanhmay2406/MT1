from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from artifacts import read_json_artifact, write_json_artifact
from channel_manifest import FROZEN_CHANNEL_MANIFEST_SHA256, load_channel_manifest, validate_channel_manifest
from dataset_manifest import DATASET_MANIFEST_SCHEMA, assert_dataset_manifest_sha256, load_dataset_manifest, validate_dataset_manifest
from integrity import assert_file_sha256, sha256_file
from reproducibility import E6_DEFAULT_ATOL, build_e6_artifact, canonical_json_hash, compare_stage_artifact, scientific_environment_gate, runtime_fingerprint
from determinism import CUBLAS_WORKSPACE_CONFIG, DETERMINISTIC_SEED

CHECKPOINT_SHA256 = "953a2b8d8e412227a89b9dd42c0899b33de28f281110af54829ec531db16dda0"
PROBE_SHA256 = "1016ed7eacda87c6b368c880a5564799e22e7fa757e11cd9cf93845b396d811d"
DATASET_MANIFEST_SHA256 = "fce7c5bc78d606c641873220f124d4656f1c32d96c7ca1e2dbc7b0b4d4f6536f"
E1_SCHEMA = "causal_audit_e1_eligibility/v2"
E2_SCHEMA = "causal_audit_e2_importance/v2"
E3_SCHEMA = "causal_audit_e3_damage/v2"
E4_SCHEMA = "causal_audit_e4_equivalence/v2"
E5_SCHEMA = "causal_audit_e5_statistics/v2"
AMENDMENT_ID = "e2_taylor_determinism_amendment_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="E6 reproducibility audit for E1-E5", allow_abbrev=False)
    for name in ("checkpoint", "probe", "images-root", "eligibility", "importance", "intervention", "equivalence", "statistics", "channel-manifest", "e4-sample", "dataset-manifest"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--output-dir", required=True, help="New directory retaining E1-E5 reruns and the E6 report; must not exist")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--atol", type=float, default=E6_DEFAULT_ATOL)
    parser.add_argument("--environment-manifest")
    return parser.parse_args()


def _paths(args: argparse.Namespace) -> dict[str, Path]:
    paths = {name.replace("-", "_"): Path(getattr(args, name.replace("-", "_"))) for name in ("checkpoint", "probe", "images-root", "eligibility", "importance", "intervention", "equivalence", "statistics", "channel-manifest", "e4-sample", "dataset-manifest")}
    paths["output_dir"] = Path(args.output_dir).expanduser().absolute()
    paths["output"] = paths["output_dir"] / "e6_reproducibility.json"
    return paths


def _validate_baselines(args: argparse.Namespace) -> dict:
    paths = _paths(args)
    if paths["output_dir"].exists() or paths["output_dir"].is_symlink():
        raise FileExistsError(f"Output directory already exists; choose a new directory: {paths['output_dir']}")
    if not math.isfinite(args.atol) or args.atol < 0:
        raise ValueError("--atol must be finite and non-negative")
    if args.environment_manifest and not Path(args.environment_manifest).is_file():
        raise FileNotFoundError(f"Environment manifest not found: {args.environment_manifest}")
    if args.environment_manifest:
        environment_manifest = json.loads(Path(args.environment_manifest).read_text(encoding="utf-8"))
        if environment_manifest.get("schema_version") != "causal_audit_environment_manifest/v1" or environment_manifest.get("protocol_amendment_id") != AMENDMENT_ID:
            raise ValueError("Unsupported deterministic environment manifest")
        if environment_manifest.get("checkpoint_sha256") != CHECKPOINT_SHA256 or environment_manifest.get("probe_sha256") != PROBE_SHA256 or environment_manifest.get("dataset_manifest_sha256") != DATASET_MANIFEST_SHA256:
            raise ValueError("Environment manifest frozen-input provenance mismatch")
    checkpoint_sha256 = assert_file_sha256(paths["checkpoint"], CHECKPOINT_SHA256)
    probe_sha256 = assert_file_sha256(paths["probe"], PROBE_SHA256)
    channel_manifest_sha256 = assert_file_sha256(paths["channel_manifest"], FROZEN_CHANNEL_MANIFEST_SHA256)
    dataset_manifest_sha256 = assert_dataset_manifest_sha256(paths["dataset_manifest"], DATASET_MANIFEST_SHA256)
    probe = json.loads(paths["probe"].read_text(encoding="utf-8"))
    dataset = load_dataset_manifest(paths["dataset_manifest"])
    if dataset.get("schema_version") != DATASET_MANIFEST_SCHEMA:
        raise ValueError("Unsupported dataset manifest schema")
    dataset_verification = validate_dataset_manifest(dataset, paths["dataset_manifest"].parent, probe, probe_path=paths["probe"])
    stages = {
        "e1": read_json_artifact(paths["eligibility"]),
        "e2": read_json_artifact(paths["importance"]),
        "e3": read_json_artifact(paths["intervention"]),
        "e4": read_json_artifact(paths["equivalence"]),
        "e5": read_json_artifact(paths["statistics"]),
    }
    expected_schemas = {"e1": E1_SCHEMA, "e2": E2_SCHEMA, "e3": E3_SCHEMA, "e4": E4_SCHEMA, "e5": E5_SCHEMA}
    for stage, expected in expected_schemas.items():
        if stages[stage].get("schema_version") != expected:
            raise ValueError(f"Unsupported {stage.upper()} schema")
    e1_sha256 = sha256_file(paths["eligibility"])
    e2_sha256 = sha256_file(paths["importance"])
    e3_sha256 = sha256_file(paths["intervention"])
    e4_sha256 = sha256_file(paths["equivalence"])
    if stages["e1"].get("checkpoint_sha256") != checkpoint_sha256 or stages["e1"].get("probe_sha256") != probe_sha256:
        raise ValueError("E1 frozen provenance mismatch")
    if stages["e2"].get("checkpoint_sha256") != checkpoint_sha256 or stages["e2"].get("probe_sha256") != probe_sha256 or stages["e2"].get("eligibility_sha256") != e1_sha256:
        raise ValueError("E2 frozen provenance mismatch")
    if stages["e3"].get("checkpoint_sha256") != checkpoint_sha256 or stages["e3"].get("probe_sha256") != probe_sha256 or stages["e3"].get("eligibility_sha256") != e1_sha256 or stages["e3"].get("importance_sha256") != e2_sha256 or stages["e3"].get("dataset_manifest_sha256") != dataset_manifest_sha256 or stages["e3"].get("channel_manifest_sha256") != channel_manifest_sha256:
        raise ValueError("E3 frozen provenance mismatch")
    if stages["e4"].get("checkpoint_sha256") != checkpoint_sha256 or stages["e4"].get("probe_sha256") != probe_sha256 or stages["e4"].get("eligibility_sha256") != e1_sha256 or stages["e4"].get("importance_sha256") != e2_sha256 or stages["e4"].get("intervention_sha256") != e3_sha256 or stages["e4"].get("dataset_manifest_sha256") != dataset_manifest_sha256 or stages["e4"].get("channel_manifest_sha256") != channel_manifest_sha256:
        raise ValueError("E4 frozen provenance mismatch")
    if stages["e5"].get("checkpoint_sha256") != checkpoint_sha256 or stages["e5"].get("probe_sha256") != probe_sha256 or stages["e5"].get("eligibility_sha256") != e1_sha256 or stages["e5"].get("importance_sha256") != e2_sha256 or stages["e5"].get("intervention_sha256") != e3_sha256 or stages["e5"].get("equivalence_sha256") != e4_sha256 or stages["e5"].get("dataset_manifest_sha256") != dataset_manifest_sha256 or stages["e5"].get("channel_manifest_sha256") != channel_manifest_sha256:
        raise ValueError("E5 frozen provenance mismatch")
    if len(stages["e1"].get("images", [])) != 300 or stages["e1"].get("image_count") != 300:
        raise ValueError("E1 must contain the complete 300-image ledger")
    for row, image in zip(stages["e1"]["images"], probe["images"]):
        if row.get("image_id") != image.get("id") or row.get("file_name") != image.get("file_name"):
            raise ValueError("E1 image order does not match the frozen probe")
    if stages["e2"].get("channel_count") != 7552 or stages["e2"].get("group_count") != 32:
        raise ValueError("E2 must contain 7552 channels and 32 groups")
    if args.environment_manifest:
        determinism = stages["e2"].get("determinism")
        if not isinstance(determinism, dict) or determinism.get("seed") != DETERMINISTIC_SEED or determinism.get("deterministic_algorithms") is not True or determinism.get("cudnn_deterministic") is not True or determinism.get("cudnn_benchmark") is not False or determinism.get("cublas_workspace_config") != CUBLAS_WORKSPACE_CONFIG:
            raise ValueError("E2 is missing the frozen deterministic Taylor metadata")
    channel_manifest = load_channel_manifest(paths["channel_manifest"])
    validate_channel_manifest(channel_manifest, stages["e2"], checkpoint_sha256, probe_sha256)
    channel_ids = [f"{entry['producer_name']}|channel={int(entry['local_channel_index'])}" for entry in channel_manifest["entries"]]
    if len(set(channel_ids)) != 384:
        raise ValueError("Frozen channel manifest must contain 384 unique channels")
    if not stages["e3"].get("complete") or stages["e3"].get("completed_pair_count") != 115200:
        raise ValueError("E3 must be complete with 115200 pairs")
    e3_pairs = stages["e3"].get("pairs", [])
    e3_keys = {(row.get("canonical_id"), int(row.get("image_id"))) for row in e3_pairs}
    expected_e3_keys = {(channel_id, int(image["id"])) for channel_id in channel_ids for image in probe["images"]}
    if len(e3_keys) != len(e3_pairs) or e3_keys != expected_e3_keys or any(row.get("status") != "ok" or row.get("state_restored") is not True or row.get("rng_restored") is not True for row in e3_pairs):
        raise ValueError("E3 channel-image coverage is incomplete or duplicated")
    if not stages["e4"].get("complete") or stages["e4"].get("completed_pair_count") != 128:
        raise ValueError("E4 must be complete with 128 pairs")
    e4_keys = {(row.get("canonical_id"), int(row.get("image_id"))) for row in stages["e4"].get("pairs", [])}
    if len(e4_keys) != len(stages["e4"].get("pairs", [])) or any(row.get("status") != "ok" for row in stages["e4"].get("pairs", [])):
        raise ValueError("E4 contains duplicate channel-image pairs")
    if not stages["e5"].get("complete") or stages["e5"].get("channel_count") != 384 or stages["e5"].get("group_count") != 32:
        raise ValueError("E5 must be complete with 384 channels and 32 groups")
    if stages["e5"].get("bootstrap", {}).get("seed") != 20260905 or stages["e5"].get("bootstrap", {}).get("replicates") != 10000:
        raise ValueError("E5 bootstrap configuration is not frozen")
    if stages["e5"].get("permutation", {}).get("seed") != 20260905 or stages["e5"].get("permutation", {}).get("replicates") != 100000:
        raise ValueError("E5 permutation configuration is not frozen")
    if stages["e4"].get("intervention_sha256") != sha256_file(paths["intervention"]):
        raise ValueError("E4 does not reference supplied E3 artifact")
    e4_sample = json.loads(paths["e4_sample"].read_text(encoding="utf-8"))
    if e4_sample.get("schema_version") != "causal_audit_e4_equivalence_sample/v2" or e4_sample.get("channel_count") != 8 or e4_sample.get("image_count") != 16:
        raise ValueError("E4 sample must contain the frozen 8x16 sample")
    sample_channel_ids = {row.get("canonical_id") for row in e4_sample.get("channels", [])}
    sample_image_ids = {int(row.get("image_id")) for row in e4_sample.get("images", [])}
    expected_e4_keys = {(channel_id, image_id) for channel_id in sample_channel_ids for image_id in sample_image_ids}
    if len(sample_channel_ids) != 8 or len(sample_image_ids) != 16 or e4_keys != expected_e4_keys:
        raise ValueError("E4 sample coverage is incomplete or duplicated")
    from equivalence import select_e4_sample
    selected, selected_images = select_e4_sample(channel_manifest["entries"], probe)
    if [r["canonical_id"] for r in e4_sample["channels"]] != [f"{r['producer_name']}|channel={int(r['local_channel_index'])}" for r in selected]:
        raise ValueError("E6 requires the corrected outcome-blind E4 selector")
    if [int(r["image_id"]) for r in e4_sample["images"]] != [int(r["id"]) for r in selected_images]:
        raise ValueError("E6 requires first 16 probe images for E4")
    e5_ids = [row.get("canonical_id") for row in stages["e5"].get("channel_rows", [])]
    if len(e5_ids) != 384 or len(set(e5_ids)) != 384 or set(e5_ids) != set(channel_ids):
        raise ValueError("E5 channel rows do not match the frozen 384-channel universe")
    for row in stages["e5"].get("channel_rows", []):
        if not math.isfinite(float(row.get("damage"))):
            raise ValueError("E5 contains non-finite damage")
        for method in ("gxa", "activation", "l1", "taylor"):
            if not math.isfinite(float(row["scores"][method]["raw"])):
                raise ValueError("E5 contains non-finite importance")
    hashes = {
        "checkpoint_sha256": checkpoint_sha256,
        "probe_sha256": probe_sha256,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "channel_manifest_sha256": channel_manifest_sha256,
        "e4_sample_sha256": sha256_file(paths["e4_sample"]),
    }
    return {"dataset_verification": dataset_verification, "paths": paths, "probe": probe, "stages": stages, "frozen_sample": e4_sample, "hashes": hashes, "reference_hashes": {stage: sha256_file(paths[name]) for stage, name in (("e1", "eligibility"), ("e2", "importance"), ("e3", "intervention"), ("e4", "equivalence"), ("e5", "statistics"))}}


def dry_run(args: argparse.Namespace) -> int:
    data = _validate_baselines(args)
    print("E6_DRY_RUN_OK")
    print(f"output_dir={data['paths']['output_dir']}")
    print(f"output={data['paths']['output']}")
    for stage, _, path in _stage_commands(args, data, data["paths"]["output_dir"], args.device):
        print(f"{stage}_output={path}")
    print(f"e4_sample_output={data['paths']['output_dir'] / 'rerun_e4_sample.json'}")
    print("stage_count=5")
    print("channel_count=384")
    print("e3_pair_count=115200")
    print("e4_pair_count=128")
    print("bootstrap_replicates=10000")
    print("permutation_replicates=100000")
    return 0


def _rebind_e4_sample(source: Path, destination: Path, *, eligibility_sha256: str, importance_sha256: str, intervention_sha256: str, expected_sample_identity_sha256: str | None = None) -> Path:
    sample = json.loads(source.read_text(encoding="utf-8"))
    if expected_sample_identity_sha256 is not None and sample.get("sample_identity_sha256") != expected_sample_identity_sha256:
        raise ValueError("E4 sample identity changed")
    sample["eligibility_sha256"] = eligibility_sha256
    sample["importance_sha256"] = importance_sha256
    sample["intervention_sha256"] = intervention_sha256
    sample.pop("sample_sha256", None)
    sample["sample_sha256"] = canonical_json_hash({key: value for key, value in sample.items() if key not in {"sample_sha256", "sample_identity_sha256"}})
    write_json_artifact(destination, sample)
    return destination


def _run_stage(stage: str, command: list[str], artifact_path: Path, env: dict[str, str]) -> dict:
    print(f"E6_STAGE_START stage={stage.upper()}", flush=True)
    started = time.time()
    process = subprocess.Popen(command, cwd=REPO_ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    tail: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        clean = line.rstrip("\n")
        print(f"[{stage.upper()}] {clean}", flush=True)
        tail.append(clean)
        del tail[:-80]
    returncode = process.wait()
    elapsed = time.time() - started
    print(f"E6_STAGE_DONE stage={stage.upper()} elapsed={elapsed:.1f}s returncode={returncode}", flush=True)
    if returncode != 0 or not artifact_path.is_file():
        raise RuntimeError(f"{stage.upper()} rerun failed (returncode={returncode}):\n" + "\n".join(tail[-80:]))
    return {"stage": stage, "returncode": returncode, "elapsed_seconds": elapsed, "log_tail": tail}


def _stage_commands(args: argparse.Namespace, data: dict, work: Path, device: str, e4_sample: Path | None = None) -> list[tuple[str, list[str], Path]]:
    p = data["paths"]
    python = sys.executable
    common = ["--checkpoint", str(p["checkpoint"].resolve()), "--probe", str(p["probe"].resolve()), "--images-root", str(p["images_root"].resolve()), "--device", device]
    e1 = work / "rerun_e1_eligibility.json"
    e2 = work / "rerun_e2_importance.json"
    e3 = work / "rerun_e3_damage.json"
    e4 = work / "rerun_e4_equivalence.json"
    e5 = work / "rerun_e5_statistics.json"
    sample = e4_sample or p["e4_sample"].resolve()
    return [
        ("e1", [python, str(REPO_ROOT / "scripts/e1_eligibility.py"), *common, "--output", str(e1), "--overwrite"], e1),
        ("e2", [python, str(REPO_ROOT / "scripts/e2_score.py"), *common, "--eligibility", str(e1), "--seed", str(DETERMINISTIC_SEED), "--output", str(e2), "--overwrite"], e2),
        ("e3", [python, str(REPO_ROOT / "scripts/e3_intervene.py"), *common, "--eligibility", str(e1), "--importance", str(e2), "--channel-manifest", str(p["channel_manifest"].resolve()), "--dataset-manifest", str(p["dataset_manifest"].resolve()), "--output", str(e3), "--overwrite"], e3),
        ("e4", [python, str(REPO_ROOT / "scripts/e4_equivalence.py"), *common, "--eligibility", str(e1), "--importance", str(e2), "--intervention", str(e3), "--channel-manifest", str(p["channel_manifest"].resolve()), "--e4-sample", str(sample), "--dataset-manifest", str(p["dataset_manifest"].resolve()), "--output", str(e4), "--overwrite"], e4),
        ("e5", [python, str(REPO_ROOT / "scripts/e5_analyze.py"), "--checkpoint", str(p["checkpoint"].resolve()), "--probe", str(p["probe"].resolve()), "--eligibility", str(e1), "--importance", str(e2), "--intervention", str(e3), "--equivalence", str(e4), "--channel-manifest", str(p["channel_manifest"].resolve()), "--dataset-manifest", str(p["dataset_manifest"].resolve()), "--output", str(e5), "--overwrite"], e5),
    ]


def run_real(args: argparse.Namespace) -> int:
    from determinism import configure_determinism
    configure_determinism()
    if args.atol != 0:
        raise ValueError("E6 requires exact canonical payloads; tolerance overrides are not permitted")
    os.environ["PYTHONHASHSEED"] = str(DETERMINISTIC_SEED)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG
    data = _validate_baselines(args)
    device = args.device
    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    work = data["paths"]["output_dir"]
    work.mkdir(parents=True, exist_ok=False)
    print(f"output_dir={work}", flush=True)
    print(f"output={data['paths']['output']}", flush=True)
    stage_results = []
    rerun_paths = {}
    started = time.time()
    commands = _stage_commands(args, data, work, device)
    stage_env = os.environ.copy()
    stage_env["PYTHONHASHSEED"] = str(DETERMINISTIC_SEED)
    stage_env["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG
    for index, (stage, command, artifact_path) in enumerate(commands):
        if stage == "e4":
            _rebind_e4_sample(
                data["paths"]["e4_sample"],
                work / "rerun_e4_sample.json",
                eligibility_sha256=sha256_file(rerun_paths["e1"]),
                importance_sha256=sha256_file(rerun_paths["e2"]),
                intervention_sha256=sha256_file(rerun_paths["e3"]),
                expected_sample_identity_sha256=data["frozen_sample"]["sample_identity_sha256"],
            )
            command = _stage_commands(args, data, work, device, work / "rerun_e4_sample.json")[index][1]
        rerun_paths[stage] = artifact_path
        stage_results.append(_run_stage(stage, command, artifact_path, stage_env))
    comparisons = []
    for stage, name in (("e1", "eligibility"), ("e2", "importance"), ("e3", "intervention"), ("e4", "equivalence"), ("e5", "statistics")):
        candidate = read_json_artifact(rerun_paths[stage])
        bindings = {"eligibility_sha256": "e1", "importance_sha256": "e2", "intervention_sha256": "e3", "equivalence_sha256": "e4"}
        for field, upstream in bindings.items():
            if field in data["stages"][stage] and candidate.get(field) != sha256_file(rerun_paths[upstream]):
                raise RuntimeError(f"Rerun upstream provenance mismatch: {stage}.{field}")
        if stage == "e4" and "e4_sample_sha256" in data["stages"][stage] and candidate.get("e4_sample_sha256") != sha256_file(work / "rerun_e4_sample.json"):
            raise RuntimeError("Rerun E4 sample provenance mismatch")
        comparisons.append(compare_stage_artifact(stage, data["stages"][stage], candidate, atol=args.atol, rtol=0.0))
    environment = runtime_fingerprint(repo_root=REPO_ROOT, device=device)
    reference_environment = json.loads(Path(args.environment_manifest).read_text(encoding="utf-8")) if args.environment_manifest else None
    environment_status = scientific_environment_gate(environment, reference_environment)
    artifact = build_e6_artifact(
        hashes=data["hashes"],
        reference_artifact_hashes=data["reference_hashes"],
        rerun_artifact_hashes={stage: sha256_file(path) for stage, path in rerun_paths.items()},
        stage_comparisons=comparisons,
        environment=environment_status,
        configuration={"device": device, "matching_iou_threshold": 0.5, "bootstrap_seed": 20260905, "bootstrap_replicates": 10000, "permutation_seed": 20260905, "permutation_replicates": 100000},
        payload_reproducible=all(item["passed"] for item in comparisons),
        environment_provenance_verified=bool(environment_status["verified"]),
        atol=args.atol,
        runtime={"elapsed_seconds": time.time() - started, "stages": stage_results},
        protocol_amendment_id=AMENDMENT_ID if args.environment_manifest else None,
    )
    artifact["dataset_verification"] = data.get("dataset_verification")
    artifact["scientific_completion"] = False
    artifact["execution"] = {"command": sys.argv, "interpreter": sys.executable,
                             "reference_environment_is_required": True}
    write_json_artifact(data["paths"]["output"], artifact)
    print("E6_OK" if artifact["reproducibility_pass"] else "E6_FAIL")
    print(f"output={data['paths']['output']}")
    print(f"payload_reproducible={artifact['payload_reproducible']}")
    print(f"environment_provenance_verified={artifact['environment_provenance_verified']}")
    print(f"reproducibility_pass={artifact['reproducibility_pass']}")
    return 0 if artifact["reproducibility_pass"] else 2


def main() -> int:
    args = parse_args()
    return dry_run(args) if args.dry_run else run_real(args)


if __name__ == "__main__":
    raise SystemExit(main())
