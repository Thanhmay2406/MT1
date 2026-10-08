from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from determinism import CUBLAS_WORKSPACE_CONFIG, DETERMINISTIC_SEED
from channel_manifest import FROZEN_CHANNEL_MANIFEST_SHA256
from dataset_manifest import assert_dataset_manifest_sha256
from artifacts import write_json_artifact
from integrity import assert_file_sha256, sha256_file
from reproducibility import canonical_json_hash, runtime_fingerprint

CHECKPOINT_SHA256 = "953a2b8d8e412227a89b9dd42c0899b33de28f281110af54829ec531db16dda0"
PROBE_SHA256 = "1016ed7eacda87c6b368c880a5564799e22e7fa757e11cd9cf93845b396d811d"
DATASET_MANIFEST_SHA256 = "fce7c5bc78d606c641873220f124d4656f1c32d96c7ca1e2dbc7b0b4d4f6536f"


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze the E2 deterministic environment identity")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--probe", required=True)
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--channel-manifest", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    from determinism import configure_determinism
    configure_determinism()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Environment manifest already exists: {output}")
    checkpoint_sha256 = assert_file_sha256(args.checkpoint, CHECKPOINT_SHA256)
    probe_sha256 = assert_file_sha256(args.probe, PROBE_SHA256)
    dataset_sha256 = assert_dataset_manifest_sha256(args.dataset_manifest, DATASET_MANIFEST_SHA256)
    channel_sha256 = assert_file_sha256(args.channel_manifest, FROZEN_CHANNEL_MANIFEST_SHA256)
    fingerprint = runtime_fingerprint(repo_root=REPO_ROOT, device=args.device)
    payload = {
        "schema_version": "causal_audit_environment_manifest/v1",
        "protocol_amendment_id": "e2_taylor_determinism_amendment_v1",
        "fingerprint": fingerprint,
        "determinism": {"seed": DETERMINISTIC_SEED, "cublas_workspace_config": CUBLAS_WORKSPACE_CONFIG, "pythonhashseed": str(DETERMINISTIC_SEED)},
        "checkpoint_sha256": checkpoint_sha256,
        "probe_sha256": probe_sha256,
        "dataset_manifest_sha256": dataset_sha256,
        "channel_manifest_sha256": channel_sha256,
    }
    payload["manifest_sha256"] = canonical_json_hash(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json_artifact(output, payload)
    print("ENVIRONMENT_MANIFEST_OK")
    print(f"output={output}")
    print(f"manifest_sha256={payload['manifest_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
