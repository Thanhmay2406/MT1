from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from artifacts import build_e5_artifact, read_e3_artifact, read_e4_artifact, read_json_artifact, write_json_artifact
from bootstrap import BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, hierarchical_bootstrap
from channel_manifest import FROZEN_CHANNEL_MANIFEST_SHA256, load_channel_manifest, validate_channel_manifest
from dataset_manifest import DATASET_MANIFEST_SCHEMA, assert_dataset_manifest_sha256, load_dataset_manifest, validate_dataset_manifest
from integrity import assert_file_sha256, sha256_file
from permutation import PERMUTATION_REPLICATES, PERMUTATION_SEED, permutation_tests
from statistics import METHODS, group_statistics, macro_average, paired_macro_differences, partial_rank_macro

CHECKPOINT_SHA256 = "953a2b8d8e412227a89b9dd42c0899b33de28f281110af54829ec531db16dda0"
PROBE_SHA256 = "1016ed7eacda87c6b368c880a5564799e22e7fa757e11cd9cf93845b396d811d"
DATASET_MANIFEST_SHA256 = "fce7c5bc78d606c641873220f124d4656f1c32d96c7ca1e2dbc7b0b4d4f6536f"
E1_SCHEMA = "causal_audit_e1_eligibility/v1"
E2_SCHEMA = "causal_audit_e2_importance/v1"
E3_SCHEMA = "causal_audit_e3_damage/v1"
E4_SCHEMA = "causal_audit_e4_equivalence/v1"
EXPECTED_IMAGES = 300
EXPECTED_ANNOTATIONS = 372
MATCHING_IOU_THRESHOLD = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="E5 hierarchical causal-audit statistics")
    for name in ("checkpoint", "probe", "eligibility", "importance", "intervention", "equivalence", "channel-manifest", "dataset-manifest", "output"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--bootstrap-replicates", type=int, default=BOOTSTRAP_REPLICATES)
    parser.add_argument("--permutation-replicates", type=int, default=PERMUTATION_REPLICATES)
    return parser.parse_args()


def _load_probe(path: Path) -> dict:
    probe = json.loads(path.read_text(encoding="utf-8"))
    if len(probe.get("images", [])) != EXPECTED_IMAGES or len(probe.get("annotations", [])) != EXPECTED_ANNOTATIONS:
        raise ValueError("Probe size does not match frozen probe")
    return probe


def _validate_e1(e1: dict, probe: dict, checkpoint_sha256: str, probe_sha256: str) -> None:
    if e1.get("schema_version") != E1_SCHEMA or e1.get("checkpoint_sha256") != checkpoint_sha256 or e1.get("probe_sha256") != probe_sha256:
        raise ValueError("E1 schema or provenance mismatch")
    rows = e1.get("images", [])
    if len(rows) != len(probe["images"]):
        raise ValueError("E1 image ledger is incomplete")
    for row, image in zip(rows, probe["images"]):
        if row.get("image_id") != image.get("id") or row.get("file_name") != image.get("file_name"):
            raise ValueError("E1/probe image order mismatch")


def _validate_e3(e3: dict, probe: dict, channel_ids: list[str], e2_sha256: str, e1_sha256: str) -> dict[tuple[str, int], dict]:
    if e3.get("schema_version") != E3_SCHEMA or not e3.get("complete"):
        raise ValueError("E3 must be complete")
    if e3.get("eligibility_sha256") != e1_sha256 or e3.get("importance_sha256") != e2_sha256:
        raise ValueError("E3 E1/E2 provenance mismatch")
    expected = {(channel, int(image["id"])) for channel in channel_ids for image in probe["images"]}
    pairs = e3.get("pairs", [])
    indexed = {(row.get("canonical_id"), int(row.get("image_id"))): row for row in pairs}
    if len(indexed) != len(pairs) or set(indexed) != expected or len(pairs) != 115200:
        raise ValueError("E3 pairs are incomplete or duplicated")
    if any(row.get("status") != "ok" or row.get("state_restored") is not True or row.get("rng_restored") is not True for row in pairs):
        raise ValueError("E3 contains an unverified pair")
    return indexed


def _validate_e4(e4: dict, e3_sha256: str) -> None:
    if e4.get("schema_version") != E4_SCHEMA or not e4.get("complete"):
        raise ValueError("E4 must be complete")
    if e4.get("intervention_sha256") != e3_sha256:
        raise ValueError("E4 does not reference supplied E3 artifact")
    if e4.get("channel_count") != 8 or e4.get("image_count") != 16 or e4.get("completed_pair_count") != 128:
        raise ValueError("E4 bounded sample is incomplete")


def _preflight(args: argparse.Namespace) -> dict:
    names = ("checkpoint", "probe", "eligibility", "importance", "intervention", "equivalence", "channel-manifest", "dataset-manifest", "output")
    paths = {name.replace("-", "_"): Path(getattr(args, name.replace("-", "_"))) for name in names}
    if paths["output"].exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists; pass --overwrite: {paths['output']}")
    if args.bootstrap_replicates < 1 or args.permutation_replicates < 1:
        raise ValueError("Replicate counts must be positive")
    checkpoint_sha256 = assert_file_sha256(paths["checkpoint"], CHECKPOINT_SHA256)
    probe_sha256 = assert_file_sha256(paths["probe"], PROBE_SHA256)
    channel_manifest_sha256 = assert_file_sha256(paths["channel_manifest"], FROZEN_CHANNEL_MANIFEST_SHA256)
    dataset_manifest_sha256 = assert_dataset_manifest_sha256(paths["dataset_manifest"], DATASET_MANIFEST_SHA256)
    probe = _load_probe(paths["probe"])
    dataset_manifest = load_dataset_manifest(paths["dataset_manifest"])
    if dataset_manifest.get("schema_version") != DATASET_MANIFEST_SCHEMA:
        raise ValueError("Unsupported dataset manifest schema")
    validate_dataset_manifest(dataset_manifest, paths["dataset_manifest"].parent, probe, probe_path=paths["probe"])
    e1 = read_json_artifact(paths["eligibility"])
    _validate_e1(e1, probe, checkpoint_sha256, probe_sha256)
    e1_sha256 = sha256_file(paths["eligibility"])
    e2 = read_json_artifact(paths["importance"])
    e2_sha256 = sha256_file(paths["importance"])
    if e2.get("schema_version") != E2_SCHEMA or e2.get("checkpoint_sha256") != checkpoint_sha256 or e2.get("probe_sha256") != probe_sha256 or e2.get("eligibility_sha256") != e1_sha256:
        raise ValueError("E2 schema or provenance mismatch")
    channel_manifest = load_channel_manifest(paths["channel_manifest"])
    validate_channel_manifest(channel_manifest, e2, checkpoint_sha256, probe_sha256)
    channel_ids = [f"{entry['producer_name']}|channel={int(entry['local_channel_index'])}" for entry in channel_manifest["entries"]]
    e3 = read_e3_artifact(paths["intervention"])
    e3_sha256 = sha256_file(paths["intervention"])
    if e3.get("checkpoint_sha256") != checkpoint_sha256 or e3.get("probe_sha256") != probe_sha256 or e3.get("dataset_manifest_sha256") != dataset_manifest_sha256 or e3.get("channel_manifest_sha256") != channel_manifest_sha256:
        raise ValueError("E3 frozen-input provenance mismatch")
    e3_pairs = _validate_e3(e3, probe, channel_ids, e2_sha256, e1_sha256)
    e4 = read_e4_artifact(paths["equivalence"])
    e4_sha256 = sha256_file(paths["equivalence"])
    _validate_e4(e4, e3_sha256)
    if e4.get("checkpoint_sha256") != checkpoint_sha256 or e4.get("probe_sha256") != probe_sha256 or e4.get("dataset_manifest_sha256") != dataset_manifest_sha256 or e4.get("channel_manifest_sha256") != channel_manifest_sha256:
        raise ValueError("E4 frozen-input provenance mismatch")
    return {"paths": paths, "probe": probe, "e1": e1, "e2": e2, "e3": e3, "e4": e4, "e3_pairs": e3_pairs, "hashes": {"checkpoint_sha256": checkpoint_sha256, "probe_sha256": probe_sha256, "eligibility_sha256": e1_sha256, "importance_sha256": e2_sha256, "intervention_sha256": e3_sha256, "equivalence_sha256": e4_sha256, "dataset_manifest_sha256": dataset_manifest_sha256, "channel_manifest_sha256": channel_manifest_sha256}, "channel_ids": channel_ids}


def _build_channel_rows(data: dict) -> tuple[list[dict], list[int]]:
    eligible_rows = [row for row in data["e1"]["images"] if row.get("matches")]
    eligible_image_ids = [int(row["image_id"]) for row in eligible_rows]
    e2_by_id = {row["canonical_id"]: row for row in data["e2"]["channels"]}
    rows = []
    for canonical_id in data["channel_ids"]:
        e2_row = e2_by_id.get(canonical_id)
        if e2_row is None:
            raise ValueError(f"E2 channel missing: {canonical_id}")
        damage_by_image = {}
        for image_id in eligible_image_ids:
            pair = data["e3_pairs"][(canonical_id, image_id)]
            object_rows = pair.get("object_damage", [])
            if not object_rows:
                raise ValueError(f"Eligible image has no object damage: {canonical_id}, {image_id}")
            damage = sum(float(item["damage"]) for item in object_rows) / len(object_rows)
            if not math.isfinite(damage):
                raise ValueError("Non-finite damage")
            if not math.isclose(damage, float(pair["image_damage"]), rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"E3 image damage aggregation mismatch: {canonical_id}, {image_id}")
            damage_by_image[str(image_id)] = damage
        rows.append({"canonical_id": canonical_id, "group_id": e2_row["group_id"], "stage": e2_row["stage"], "block": e2_row["block"], "hidden_conv": e2_row["hidden_conv"], "scores": e2_row["scores"], "damage_by_image": damage_by_image, "damage": sum(damage_by_image.values()) / len(damage_by_image), "eligible_image_count": len(damage_by_image)})
    return rows, eligible_image_ids


def _analyze(data: dict, bootstrap_replicates: int, permutation_replicates: int) -> dict:
    channel_rows, eligible_image_ids = _build_channel_rows(data)
    group_stats = group_statistics(channel_rows, METHODS)
    macro_stats = {method: {"spearman": macro_average(group_stats, "spearman", method), "kendall": macro_average(group_stats, "kendall", method)} for method in METHODS}
    paired = paired_macro_differences(group_stats, METHODS)
    hypotheses = {"H1": {"method": "gxa", "null": "macro_spearman <= 0", "observed": macro_stats["gxa"]["spearman"]["value"], "supported": False, "requires": "bootstrap_ci_lower > 0 and E6 reproducibility PASS"}, "H2": {"baseline": "l1", "observed": partial_rank_macro(channel_rows, "l1")["value"]}, "H3": {"baseline": "taylor", "observed": partial_rank_macro(channel_rows, "taylor")["value"]}, "H4": {"baseline": "activation", "observed": partial_rank_macro(channel_rows, "activation")["value"]}}
    bootstrap = hierarchical_bootstrap(channel_rows, eligible_image_ids, replicates=bootstrap_replicates, seed=BOOTSTRAP_SEED)
    permutation = permutation_tests(channel_rows, replicates=permutation_replicates, seed=PERMUTATION_SEED)
    for hypothesis, baseline in (("H2", "l1"), ("H3", "taylor"), ("H4", "activation")):
        hypotheses[hypothesis]["permutation"] = permutation["statistics"][baseline]
    return {"channel_rows": channel_rows, "group_statistics": group_stats, "macro_statistics": macro_stats, "paired_differences": paired, "hypotheses": hypotheses, "bootstrap": bootstrap, "permutation": permutation}


def dry_run(args: argparse.Namespace) -> int:
    data = _preflight(args)
    print("E5_DRY_RUN_OK")
    print("group_count=32")
    print("channel_count=384")
    print(f"eligible_image_count={sum(bool(row.get('matches')) for row in data['e1']['images'])}")
    print(f"bootstrap_replicates={args.bootstrap_replicates}")
    print(f"permutation_replicates={args.permutation_replicates}")
    print(f"e4_equivalence_status={data['e4']['equivalence_status']}")
    return 0


def run_real(args: argparse.Namespace) -> int:
    data = _preflight(args)
    analysis = _analyze(data, args.bootstrap_replicates, args.permutation_replicates)
    artifact = build_e5_artifact(**data["hashes"], equivalence_status=data["e4"]["equivalence_status"], matching_iou_threshold=MATCHING_IOU_THRESHOLD, eligible_image_count=data["e2"]["eligible_image_count"], eligible_instance_count=data["e2"]["eligible_instance_count"], **analysis)
    write_json_artifact(data["paths"]["output"], artifact)
    print("E5_OK")
    print(f"output={data['paths']['output']}")
    print("group_count=32")
    print("channel_count=384")
    print(f"bootstrap_replicates={args.bootstrap_replicates}")
    print(f"permutation_replicates={args.permutation_replicates}")
    return 0


def main() -> int:
    args = parse_args()
    return dry_run(args) if args.dry_run else run_real(args)


if __name__ == "__main__":
    raise SystemExit(main())
