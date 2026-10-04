from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from dataset_manifest import build_dataset_manifest, write_dataset_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Create amended frozen dataset manifest")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--probe", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest = build_dataset_manifest(args.dataset_root, args.probe)
    write_dataset_manifest(manifest, args.output)
    print("DATASET_MANIFEST_CREATED")
    print(f"output={args.output}")
    print(f"manifest_sha256={manifest['manifest_sha256']}")
    print(f"train_images={manifest['splits']['train']['image_count']}")
    print(f"valid_images={manifest['splits']['valid']['image_count']}")
    print(f"probe_images={manifest['probe_reference']['image_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
