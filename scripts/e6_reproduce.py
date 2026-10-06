from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
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
from reproducibility import E6_DEFAULT_ATOL, build_e6_artifact, environment_gate, compare_stage_artifact, runtime_fingerprint

CHECKPOINT_SHA256 = "953a2b8d8e412227a89b9dd42c0899b33de28f281110af54829ec531db16dda0"
PROBE_SHA256 = "1016ed7eacda87c6b368c880a5564799e22e7fa757e11cd9cf93845b396d811d"
DATASET_MANIFEST_SHA256 = "fce7c5bc78d606c641873220f124d4656f1c32d96c7ca1e2dbc7b0b4d4f6536f"
E1_SCHEMA = "causal_audit_e1_eligibility/v1"
E2_SCHEMA = "causal_audit_e2_importance/v1"
E3_SCHEMA = "causal_audit_e3_damage/v1"
E4_SCHEMA = "causal_audit_e4_equivalence/v1"
E5_SCHEMA = "causal_audit_e5_statistics/v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="E6 reproducibility audit for E1-E5")
    for name in ("checkpoint", "probe", "images-root", "eligibility", "importance", "intervention", "equivalence", "statistics", "channel-manifest", "e4-sample", "dataset-manifest", "output"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--atol", type=float, default=E6_DEFAULT_ATOL)
    parser.add_argument("--environment-manifest")
    return parser.parse_args()


def _paths(args: argparse.Namespace) -> dict[str, Path]:
    return {name.replace("-", "_"): Path(getattr(args, name.replace("-", "_"))) for name in ("checkpoint", "probe", "images-root", "eligibility", "importance", "intervention", "equivalence", "statistics", "channel-manifest", "e4-sample", "dataset-manifest", "output")}


def _validate_baselines(args: argparse.Namespace) -> dict:
    paths = _paths(args)
    if paths["output"].exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists; pass --overwrite: {paths['output']}")
    if not math.isfinite(args.atol) or args.atol < 0:
        raise ValueError("--atol must be finite and non-negative")
    if args.environment_manifest and not Path(args.environment_manifest).is_file():
        raise FileNotFoundError(f"Environment manifest not found: {args.environment_manifest}")
    checkpoint_sha256 = assert_file_sha256(paths["checkpoint"], CHECKPOINT_SHA256)
    probe_sha256 = assert_file_sha256(paths["probe"], PROBE_SHA256)
    channel_manifest_sha256 = assert_file_sha256(paths["channel_manifest"], FROZEN_CHANNEL_MANIFEST_SHA256)
    dataset_manifest_sha256 = assert_dataset_manifest_sha256(paths["dataset_manifest"], DATASET_MANIFEST_SHA256)
    probe = json.loads(paths["probe"].read_text(encoding="utf-8"))
    dataset = load_dataset_manifest(paths["dataset_manifest"])
    if dataset.get("schema_version") != DATASET_MANIFEST_SCHEMA:
        raise ValueError("Unsupported dataset manifest schema")
    validate_dataset_manifest(dataset, paths["dataset_manifest"].parent, probe, probe_path=paths["probe"])
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
    if e4_sample.get("schema_version") != "causal_audit_e4_equivalence_sample/v1" or e4_sample.get("channel_count") != 8 or e4_sample.get("image_count") != 16:
        raise ValueError("E4 sample must contain the frozen 8x16 sample")
    sample_channel_ids = {row.get("canonical_id") for row in e4_sample.get("channels", [])}
    sample_image_ids = {int(row.get("image_id")) for row in e4_sample.get("images", [])}
    expected_e4_keys = {(channel_id, image_id) for channel_id in sample_channel_ids for image_id in sample_image_ids}
    if len(sample_channel_ids) != 8 or len(sample_image_ids) != 16 or e4_keys != expected_e4_keys:
        raise ValueError("E4 sample coverage is incomplete or duplicated")
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
    return {"paths": paths, "probe": probe, "stages": stages, "hashes": hashes, "reference_hashes": {stage: sha256_file(paths[name]) for stage, name in (("e1", "eligibility"), ("e2", "importance"), ("e3", "intervention"), ("e4", "equivalence"), ("e5", "statistics"))}}


def dry_run(args: argparse.Namespace) -> int:
    data = _validate_baselines(args)
    print("E6_DRY_RUN_OK")
    print("stage_count=5")
    print("channel_count=384")
    print("e3_pair_count=115200")
    print("e4_pair_count=128")
    print("bootstrap_replicates=10000")
    print("permutation_replicates=100000")
    return 0


def _stage_commands(args: argparse.Namespace, data: dict, work: Path, device: str) -> list[tuple[str, list[str], Path]]:
    p = data["paths"]
    python = sys.executable
    common = ["--checkpoint", str(p["checkpoint"].resolve()), "--probe", str(p["probe"].resolve()), "--images-root", str(p["images_root"].resolve()), "--device", device]
    e1 = work / "rerun_e1_eligibility.json"
    e2 = work / "rerun_e2_importance.json"
    e3 = work / "rerun_e3_damage.json"
    e4 = work / "rerun_e4_equivalence.json"
    e5 = work / "rerun_e5_statistics.json"
    return [
        ("e1", [python, str(REPO_ROOT / "scripts/e1_eligibility.py"), *common, "--output", str(e1), "--overwrite"], e1),
        ("e2", [python, str(REPO_ROOT / "scripts/e2_score.py"), *common, "--eligibility", str(e1), "--output", str(e2), "--overwrite"], e2),
        ("e3", [python, str(REPO_ROOT / "scripts/e3_intervene.py"), *common, "--eligibility", str(e1), "--importance", str(e2), "--channel-manifest", str(p["channel_manifest"].resolve()), "--dataset-manifest", str(p["dataset_manifest"].resolve()), "--output", str(e3), "--overwrite"], e3),
        ("e4", [python, str(REPO_ROOT / "scripts/e4_equivalence.py"), *common, "--eligibility", str(e1), "--importance", str(e2), "--intervention", str(e3), "--channel-manifest", str(p["channel_manifest"].resolve()), "--e4-sample", str(p["e4_sample"].resolve()), "--dataset-manifest", str(p["dataset_manifest"].resolve()), "--output", str(e4), "--overwrite"], e4),
        ("e5", [python, str(REPO_ROOT / "scripts/e5_analyze.py"), "--checkpoint", str(p["checkpoint"].resolve()), "--probe", str(p["probe"].resolve()), "--eligibility", str(e1), "--importance", str(e2), "--intervention", str(e3), "--equivalence", str(e4), "--channel-manifest", str(p["channel_manifest"].resolve()), "--dataset-manifest", str(p["dataset_manifest"].resolve()), "--output", str(e5), "--overwrite"], e5),
    ]


def run_real(args: argparse.Namespace) -> int:
    if args.atol < 0:
        raise ValueError("--atol must be non-negative")
    data = _validate_baselines(args)
    device = args.device
    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    with tempfile.TemporaryDirectory(prefix="e6_reproduce-") as temporary:
        work = Path(temporary)
        stage_results = []
        rerun_paths = {}
        started = time.time()
        for stage, command, artifact_path in _stage_commands(args, data, work, device):
            completed = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True, check=False)
            rerun_paths[stage] = artifact_path
            if completed.returncode != 0 or not artifact_path.is_file():
                raise RuntimeError(f"{stage.upper()} rerun failed: {completed.stderr[-2000:]}")
            stage_results.append({"stage": stage, "returncode": completed.returncode, "stdout": completed.stdout[-4000:], "stderr": completed.stderr[-4000:]})
        comparisons = []
        for stage, name in (("e1", "eligibility"), ("e2", "importance"), ("e3", "intervention"), ("e4", "equivalence"), ("e5", "statistics")):
            comparisons.append(compare_stage_artifact(stage, data["stages"][stage], read_json_artifact(rerun_paths[stage]), atol=args.atol, rtol=0.0))
        environment = runtime_fingerprint(repo_root=REPO_ROOT, device=device)
        reference_environment = json.loads(Path(args.environment_manifest).read_text(encoding="utf-8")) if args.environment_manifest else None
        environment_status = environment_gate(environment, reference_environment)
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
        )
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
