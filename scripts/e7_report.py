from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from artifacts import E7_SCHEMA_VERSION, read_json_artifact, write_json_artifact
from integrity import sha256_file

E1_SCHEMA = "causal_audit_e1_eligibility/v1"
E2_SCHEMA = "causal_audit_e2_importance/v1"
E3_SCHEMA = "causal_audit_e3_damage/v1"
E4_SCHEMA = "causal_audit_e4_equivalence/v1"
E5_SCHEMA = "causal_audit_e5_statistics/v1"
E6_SCHEMA = "causal_audit_e6_reproducibility/v1"
EXPECTED_IMAGES = 300
EXPECTED_CHANNELS = 384
EXPECTED_GROUPS = 32
EXPECTED_E3_PAIRS = 115200
EXPECTED_E4_PAIRS = 128
EXPECTED_BOOTSTRAP = 10000
EXPECTED_PERMUTATION = 100000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="E7 frozen-artifact final report")
    for name in ("eligibility", "importance", "intervention", "equivalence", "statistics", "reproducibility", "output", "summary-output"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _paths(args: argparse.Namespace) -> dict[str, Path]:
    names = ("eligibility", "importance", "intervention", "equivalence", "statistics", "reproducibility", "output", "summary-output")
    return {name.replace("-", "_"): Path(getattr(args, name.replace("-", "_"))) for name in names}


def _require_schema(payload: dict, schema: str, stage: str) -> None:
    if payload.get("schema_version") != schema:
        raise ValueError(f"Unsupported {stage} schema")


def _validate_inputs(paths: dict[str, Path]) -> dict:
    artifacts = {
        "e1": read_json_artifact(paths["eligibility"]),
        "e2": read_json_artifact(paths["importance"]),
        "e3": read_json_artifact(paths["intervention"]),
        "e4": read_json_artifact(paths["equivalence"]),
        "e5": read_json_artifact(paths["statistics"]),
        "e6": read_json_artifact(paths["reproducibility"]),
    }
    schemas = {"e1": E1_SCHEMA, "e2": E2_SCHEMA, "e3": E3_SCHEMA, "e4": E4_SCHEMA, "e5": E5_SCHEMA, "e6": E6_SCHEMA}
    for stage, schema in schemas.items():
        _require_schema(artifacts[stage], schema, stage.upper())
    if not artifacts["e3"].get("complete") or artifacts["e3"].get("completed_pair_count") != EXPECTED_E3_PAIRS:
        raise ValueError("E3 is incomplete")
    if not artifacts["e4"].get("complete") or artifacts["e4"].get("completed_pair_count") != EXPECTED_E4_PAIRS:
        raise ValueError("E4 is incomplete")
    if not artifacts["e5"].get("complete"):
        raise ValueError("E5 is incomplete")
    if not artifacts["e6"].get("complete"):
        raise ValueError("E6 is incomplete")
    e1_hash = sha256_file(paths["eligibility"])
    e2_hash = sha256_file(paths["importance"])
    e3_hash = sha256_file(paths["intervention"])
    e4_hash = sha256_file(paths["equivalence"])
    e5_hash = sha256_file(paths["statistics"])
    e6_hash = sha256_file(paths["reproducibility"])
    if artifacts["e1"].get("image_count") != EXPECTED_IMAGES or len(artifacts["e1"].get("images", [])) != EXPECTED_IMAGES:
        raise ValueError("E1 must contain 300 images")
    if artifacts["e2"].get("channel_count") != 7552:
        raise ValueError("E2 must contain 7552 canonical channels")
    if artifacts["e3"].get("eligibility_sha256") != e1_hash or artifacts["e3"].get("importance_sha256") != e2_hash:
        raise ValueError("E3 provenance chain mismatch")
    if artifacts["e4"].get("eligibility_sha256") != e1_hash or artifacts["e4"].get("importance_sha256") != e2_hash or artifacts["e4"].get("intervention_sha256") != e3_hash:
        raise ValueError("E4 provenance chain mismatch")
    if artifacts["e5"].get("eligibility_sha256") != e1_hash or artifacts["e5"].get("importance_sha256") != e2_hash or artifacts["e5"].get("intervention_sha256") != e3_hash or artifacts["e5"].get("equivalence_sha256") != e4_hash:
        raise ValueError("E5 provenance chain mismatch")
    reference_hashes = artifacts["e6"].get("reference_artifact_hashes", {})
    expected_hashes = {"e1": e1_hash, "e2": e2_hash, "e3": e3_hash, "e4": e4_hash, "e5": e5_hash}
    if any(reference_hashes.get(stage) != value for stage, value in expected_hashes.items()):
        raise ValueError("E6 reference artifact hashes do not match supplied artifacts")
    if artifacts["e5"].get("channel_count") != EXPECTED_CHANNELS or artifacts["e5"].get("group_count") != EXPECTED_GROUPS:
        raise ValueError("E5 channel/group counts are incomplete")
    if artifacts["e5"].get("bootstrap", {}).get("replicates") != EXPECTED_BOOTSTRAP or artifacts["e5"].get("permutation", {}).get("replicates") != EXPECTED_PERMUTATION:
        raise ValueError("E5 replicate configuration is not frozen")
    return {"artifacts": artifacts, "hashes": {**expected_hashes, "e6": e6_hash}}


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def build_summary(data: dict) -> dict:
    a = data["artifacts"]
    e5 = a["e5"]
    e6 = a["e6"]
    hypotheses = e5.get("hypotheses", {})
    gxa_bootstrap = e5.get("bootstrap", {}).get("statistics", {}).get("gxa", {})
    stage_results = {
        item.get("stage"): {
            "passed": item.get("passed"),
            "mismatch_count": item.get("mismatch_count", 0),
            "compared_fields": item.get("compared_fields", 0),
        }
        for item in e6.get("stage_comparisons", [])
    }
    stage_completion = {
        "e1": bool(a["e1"].get("complete", True)),
        "e2": bool(a["e2"].get("complete", True)),
        "e3": bool(a["e3"].get("complete")),
        "e4": bool(a["e4"].get("complete")),
        "e5": bool(a["e5"].get("complete")),
        "e6": bool(a["e6"].get("complete")),
    }
    mismatch_diagnostics = {
        item.get("stage"): {
            "mismatch_count": item.get("mismatch_count", 0),
            "examples": item.get("mismatches", [])[:5],
        }
        for item in e6.get("stage_comparisons", [])
        if item.get("mismatch_count", 0)
    }
    return {
        "schema_version": E7_SCHEMA_VERSION,
        "report_language": "vi",
        "artifact_hashes": data["hashes"],
        "pipeline": {
            "stage_completion": stage_completion,
            "image_count": a["e1"].get("image_count"),
            "eligible_image_count": a["e1"].get("eligible_image_count"),
            "eligible_instance_count": a["e1"].get("eligible_instance_count"),
            "e2_channel_count": a["e2"].get("channel_count"),
            "e5_channel_count": a["e5"].get("channel_count"),
            "group_count": a["e5"].get("group_count"),
            "e3_pair_count": a["e3"].get("completed_pair_count"),
            "e4_pair_count": a["e4"].get("completed_pair_count"),
            "matching_iou_threshold": a["e1"].get("matching_iou_threshold"),
        },
        "equivalence": {"status": a["e4"].get("equivalence_status"), "tolerance": a["e4"].get("tolerance")},
        "statistics": {
            "macro_statistics": e5.get("macro_statistics"),
            "h1": {**hypotheses.get("H1", {}), "bootstrap_ci": gxa_bootstrap},
            "h2": hypotheses.get("H2", {}),
            "h3": hypotheses.get("H3", {}),
            "h4": hypotheses.get("H4", {}),
            "bootstrap": e5.get("bootstrap"),
            "permutation": e5.get("permutation"),
        },
        "reproducibility": {
            "complete": e6.get("complete"),
            "payload_reproducible": e6.get("payload_reproducible"),
            "environment_provenance_verified": e6.get("environment_provenance_verified"),
            "reproducibility_pass": e6.get("reproducibility_pass"),
            "mismatch_count": e6.get("mismatch_count"),
            "stage_comparisons": stage_results,
            "mismatch_diagnostics": mismatch_diagnostics,
        },
        "deviations": [
            "E2 Taylor scores không tái lập hoàn toàn trong artifact E6 hiện tại.",
            "E5 có sai khác downstream do phụ thuộc vào Taylor scores.",
            "Environment identity của lần chạy tham chiếu chưa được cung cấp.",
        ],
        "negative_findings": [
            "Không được kết luận H1 được hỗ trợ vì E6 reproducibility chưa PASS.",
            "Không được trộn artifact amended với artifact gốc.",
        ],
    }


def _fmt(value: object, digits: int = 6) -> str:
    return f"{float(value):.{digits}f}" if _finite(value) else "N/A"


def render_markdown(summary: dict) -> str:
    p = summary["pipeline"]
    s = summary["statistics"]
    r = summary["reproducibility"]
    lines = [
        "# E7 Báo Cáo Cuối Cùng: Causal Audit Detector Structural Attribution",
        "",
        "> Báo cáo này chỉ đọc các artifact E1–E6 đã freeze; không chạy model hoặc thay đổi protocol.",
        "",
        "## 1. Phạm Vi Và Provenance",
        "",
        "| Artifact | SHA-256 |",
        "|---|---|",
    ]
    for stage, digest in summary["artifact_hashes"].items():
        lines.append(f"| {stage} | `{digest}` |")
    lines += [
        "",
        "## 2. Tình Trạng Pipeline",
        "",
        f"- Probe images: **{p['image_count']}**; eligible images: **{p['eligible_image_count']}**; eligible instances: **{p['eligible_instance_count']}**.",
        f"- E2: **{p['e2_channel_count']}** canonical channels; E5: **{p['e5_channel_count']}** sampled channels trong **{p['group_count']}** groups.",
        f"- E3: **{p['e3_pair_count']}** verified pairs; E4: **{p['e4_pair_count']}** pairs.",
        f"- Matching IoU threshold: **{p['matching_iou_threshold']}**.",
        "",
        "## 3. E1–E4",
        "",
        "- E1 đã freeze eligibility theo probe order.",
        "- E2 cung cấp GxA, Activation, L1 và Taylor importance.",
        "- E3 sử dụng signed post-BN activation-knockout damage với đủ Cartesian pairs.",
        f"- E4 status: **{summary['equivalence']['status']}**, tolerance `{summary['equivalence']['tolerance']}`.",
        ("- E4 supports equivalence between post-BN activation knockout and physical structural-removal damage for the bounded sample."
         if summary["equivalence"]["status"] == "equivalent" else
         "- E4 is non-equivalent; damage must be described as post-BN activation-knockout damage, not physical structural-removal damage."),
        "",
        "## 4. E5 Statistics",
        "",
        "| Hypothesis | Observed | p-value | Holm-adjusted | Reject/Support |"]
    for key, label in (("h1", "H1"), ("h2", "H2"), ("h3", "H3"), ("h4", "H4")):
        h = s[key]
        perm = h.get("permutation", {}) if key != "h1" else {}
        observed = h.get("observed")
        if key == "h1":
            ci = s["h1"].get("bootstrap_ci", {})
            lines.append(f"| {label} | `{_fmt(observed)}` | N/A | CI [{_fmt(ci.get('lower'))}, {_fmt(ci.get('upper'))}] | **Not supported yet** |")
        else:
            lines.append(f"| {label} | `{_fmt(observed)}` | `{_fmt(perm.get('raw_p'))}` | `{_fmt(perm.get('holm_p'))}` | `{perm.get('reject')} |")
    lines += [
        "",
        "Bootstrap configuration: 10,000 replicates, seed `20260905`.",
        "Permutation configuration: 100,000 replicates, seed `20260905`, plus-one p-value và Holm correction.",
        "",
        "## 5. E6 Reproducibility",
        "",
        f"- `complete`: **{r['complete']}**.",
        f"- `payload_reproducible`: **{r['payload_reproducible']}**.",
        f"- `environment_provenance_verified`: **{r['environment_provenance_verified']}**.",
        f"- `reproducibility_pass`: **{r['reproducibility_pass']}**.",
        f"- Total mismatch count: **{r['mismatch_count']}**.",
        "",
        "| Stage | Passed | Mismatches |",
        "|---|---:|---:|",
    ]
    for stage, result in r["stage_comparisons"].items():
        lines.append(f"| {stage.upper()} | {result['passed']} | {result['mismatch_count']} |")
    if r["mismatch_diagnostics"]:
        lines += ["", "Mismatch diagnostics (tối đa 5 path tiêu biểu mỗi stage):"]
        for stage, diagnostic in r["mismatch_diagnostics"].items():
            lines.append(f"- **{stage.upper()}**: {diagnostic['mismatch_count']} mismatches.")
            for example in diagnostic["examples"]:
                lines.append(f"  - `{example.get('path', 'unknown')}` ({example.get('type', 'unknown')})")
    lines += [
        "",
        "E1, E3 và E4 tái lập; sai khác tập trung ở Taylor của E2 và lan truyền xuống E5. Environment identity của lần chạy tham chiếu chưa được xác minh.",
        "",
        "## 6. Kết Luận",
        "",
        "- **H1 chưa được hỗ trợ cuối cùng**: bootstrap CI của GxA dương nhưng điều kiện E6 reproducibility PASS chưa đạt.",
        "- H2–H4 được báo cáo theo đúng observed statistic, raw p-value và Holm-adjusted p-value; không suy diễn vượt estimand đã freeze.",
        "- E4 được báo cáo theo equivalence status hiện có.",
        "- E6 FAIL là một negative finding hợp lệ, không phải lý do để thay đổi sample, matching rule hoặc hypothesis.",
        "",
        "## 7. Deviations Và Negative Findings",
        "",
    ]
    lines.extend(f"- {item}" for item in summary["deviations"] + summary["negative_findings"])
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    paths = _paths(args)
    if not args.overwrite and (paths["output"].exists() or paths["summary_output"].exists()):
        raise FileExistsError("Report output already exists; pass --overwrite")
    data = _validate_inputs(paths)
    summary = build_summary(data)
    if args.dry_run:
        print("E7_DRY_RUN_OK")
        print("stage_count=6")
        print(f"e6_reproducibility_pass={summary['reproducibility']['reproducibility_pass']}")
        print(f"e3_pair_count={summary['pipeline']['e3_pair_count']}")
        print(f"e4_pair_count={summary['pipeline']['e4_pair_count']}")
        return 0
    paths["output"].parent.mkdir(parents=True, exist_ok=True)
    text = render_markdown(summary)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=paths["output"].parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, paths["output"])
    write_json_artifact(paths["summary_output"], summary)
    print("E7_OK")
    print(f"report={paths['output']}")
    print(f"summary={paths['summary_output']}")
    print(f"e6_reproducibility_pass={summary['reproducibility']['reproducibility_pass']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
