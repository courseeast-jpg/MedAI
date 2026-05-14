"""Canonical terminology source preflight validator.

The validator checks local terminology source shape using file metadata only.
It never imports terminology data, never reads LICENSE_ACK_PRIVATE.json, and
never prints licensed terminology rows.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_MANIFEST = REPO_ROOT / "config" / "terminology_sources.example.json"
LOCAL_MANIFEST = REPO_ROOT / "config" / "terminology_sources.local.json"
REPORT_DIR = REPO_ROOT / "reports" / "terminology_sources_preflight"
REPORT_JSON = REPORT_DIR / "terminology_sources_preflight_report.json"
REPORT_MD = REPORT_DIR / "terminology_sources_preflight_report.md"

PRIMARY_REQUIRED_STATUSES = {"local_private_required"}
FUTURE_GATED_STATUSES = {"local_private_required_future_block"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate canonical MedAI terminology source locations.")
    parser.add_argument("--manifest", default=None, help="Optional manifest path. Defaults to local manifest, then example.")
    parser.add_argument("--repo-root", default=str(REPO_ROOT), help="Repository root for relative path resolution.")
    parser.add_argument("--report-dir", default=None, help="Optional report output directory.")
    return parser.parse_args()


def safe_rel(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.name


def load_manifest(repo_root: Path, manifest_arg: str | None = None) -> tuple[dict[str, Any], Path]:
    if manifest_arg:
        manifest_path = Path(manifest_arg)
        if not manifest_path.is_absolute():
            manifest_path = repo_root / manifest_path
    else:
        local = repo_root / "config" / "terminology_sources.local.json"
        example = repo_root / "config" / "terminology_sources.example.json"
        manifest_path = local if local.exists() else example
    return json.loads(manifest_path.read_text(encoding="utf-8")), manifest_path


def metadata_matches(canonical_path: Path, file_name: str) -> list[Path]:
    if not canonical_path.exists():
        return []
    return [path for path in canonical_path.rglob(file_name) if path.is_file()]


def pattern_matches(canonical_path: Path, pattern: str) -> list[Path]:
    if not canonical_path.exists():
        return []
    return [path for path in canonical_path.rglob(pattern) if path.is_file()]


def duplicate_loinc_candidates(repo_root: Path) -> list[str]:
    terminology_root = repo_root / "terminology_data"
    if not terminology_root.exists():
        return []
    candidates = set()
    for path in terminology_root.iterdir():
        if path.is_dir() and "loinc" in path.name.lower():
            candidates.add(safe_rel(path, repo_root))
    for path in terminology_root.rglob("Loinc.csv"):
        if path.is_file():
            try:
                top = path.relative_to(terminology_root).parts[0]
            except (ValueError, IndexError):
                continue
            candidates.add(f"terminology_data/{top}")
    return sorted(candidates, key=str.lower)


def validate_source(name: str, spec: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    canonical_rel = spec["canonical_path"]
    canonical = repo_root / canonical_rel
    presence_only = bool(spec.get("presence_only", False))
    required_files = list(spec.get("required_files_anywhere", []))
    required_patterns = list(spec.get("required_file_patterns", []))
    missing: list[str] = []
    detected: list[dict[str, Any]] = []

    if presence_only:
        exists = canonical.exists()
        if not exists:
            missing.append(canonical_rel)
        return {
            "source": name,
            "role": spec.get("role"),
            "status": spec.get("status"),
            "canonical_path": canonical_rel,
            "canonical_exists": exists,
            "presence_only": True,
            "contents_read": False,
            "required_checks": [],
            "missing_requirements": missing,
            "ready": exists,
        }

    canonical_exists = canonical.exists() and canonical.is_dir()
    if not canonical_exists:
        missing.append(canonical_rel)

    checks = []
    for file_name in required_files:
        matches = metadata_matches(canonical, file_name)
        if not matches:
            missing.append(file_name)
        checks.append({"type": "file_name", "value": file_name, "match_count": len(matches)})
        detected.extend({"relative_path": safe_rel(match, repo_root), "size_bytes": match.stat().st_size} for match in matches[:5])

    for pattern in required_patterns:
        matches = pattern_matches(canonical, pattern)
        if not matches:
            missing.append(pattern)
        checks.append({"type": "pattern", "value": pattern, "match_count": len(matches)})
        detected.extend({"relative_path": safe_rel(match, repo_root), "size_bytes": match.stat().st_size} for match in matches[:5])

    return {
        "source": name,
        "role": spec.get("role"),
        "status": spec.get("status"),
        "canonical_path": canonical_rel,
        "canonical_exists": canonical_exists,
        "presence_only": False,
        "contents_read": False,
        "required_checks": checks,
        "detected_metadata_samples": detected[:12],
        "missing_requirements": missing,
        "ready": canonical_exists and not missing,
    }


def readiness_flags(results: list[dict[str, Any]]) -> dict[str, bool]:
    by_source = {result["source"]: result for result in results}
    return {
        "ready_for_loinc_preflight": bool(by_source.get("loinc", {}).get("ready")),
        "ready_for_rxnorm_preflight": bool(by_source.get("rxnorm", {}).get("ready")),
        "ready_for_snomed_us_preflight": bool(by_source.get("snomed_ct_us", {}).get("ready")),
        "umls_present_but_future_gated": bool(by_source.get("umls", {}).get("ready")),
        "license_ack_present_presence_only": bool(by_source.get("license_ack_private", {}).get("ready")),
    }


def build_report(repo_root: Path, manifest_arg: str | None = None) -> dict[str, Any]:
    manifest, manifest_path = load_manifest(repo_root, manifest_arg)
    results = [validate_source(name, spec, repo_root) for name, spec in manifest.items()]
    duplicate_candidates = duplicate_loinc_candidates(repo_root)
    duplicate_warnings = []
    if len(duplicate_candidates) > 1:
        duplicate_warnings.append(
            {
                "source": "loinc",
                "candidate_count": len(duplicate_candidates),
                "candidate_paths": duplicate_candidates,
                "action": "warning_only_select_canonical_before_import_changes",
            }
        )

    missing_primary = [
        result["source"]
        for result in results
        if result.get("status") in PRIMARY_REQUIRED_STATUSES and not result.get("ready")
    ]
    conclusion = "terminology_sources_preflight_ready" if not missing_primary else "blocked_missing_required_primary_sources"

    return {
        "report_id": "medai_term_data_02_sources_preflight",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_source": safe_rel(manifest_path, repo_root),
        "manifest_local_used": manifest_path.name == "terminology_sources.local.json",
        "conclusion": conclusion,
        "import_performed": False,
        "runtime_db_or_index_created": False,
        "license_ack_contents_read": False,
        "licensed_rows_printed": False,
        "absolute_paths_in_report": False,
        "external_api_used": False,
        "canonical_sources": results,
        "readiness": readiness_flags(results),
        "duplicate_candidate_warnings": duplicate_warnings,
        "missing_required_primary_sources": missing_primary,
        "umls_future_gated": True,
        "snomed_runtime_integration_enabled": False,
        "b07_behavior_changed": False,
        "clinical_logic_changed": False,
        "ocr_extractor_safety_gates_changed": False,
        "privacy_safety_assertions": {
            "relative_paths_only": True,
            "license_ack_presence_only": True,
            "metadata_only": True,
            "source_rows_not_read": True,
            "terminology_data_not_modified": True,
        },
        "next_recommended_gated_block": (
            "Operator approval for a separate, parser-specific import or adapter validation block; "
            "do not infer import readiness beyond this preflight."
        ),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    table = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        table.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(table)


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    source_rows = [
        [
            result["source"],
            result["role"],
            result["canonical_path"],
            result["status"],
            result["ready"],
            ", ".join(result["missing_requirements"]) or "none",
        ]
        for result in payload["canonical_sources"]
    ]
    readiness_rows = [[name, value] for name, value in payload["readiness"].items()]
    warning_rows = [
        [warning["source"], warning["candidate_count"], "<br>".join(warning["candidate_paths"]), warning["action"]]
        for warning in payload["duplicate_candidate_warnings"]
    ]
    safety_rows = [
        ["import_performed", payload["import_performed"]],
        ["runtime_db_or_index_created", payload["runtime_db_or_index_created"]],
        ["license_ack_contents_read", payload["license_ack_contents_read"]],
        ["licensed_rows_printed", payload["licensed_rows_printed"]],
        ["absolute_paths_in_report", payload["absolute_paths_in_report"]],
        ["external_api_used", payload["external_api_used"]],
        ["b07_behavior_changed", payload["b07_behavior_changed"]],
        ["clinical_logic_changed", payload["clinical_logic_changed"]],
        ["ocr_extractor_safety_gates_changed", payload["ocr_extractor_safety_gates_changed"]],
    ]

    lines = [
        "# Terminology Sources Preflight",
        "",
        "This report validates canonical local terminology source locations using metadata only. It does not import terminology data, create runtime indexes, read private license acknowledgment contents, or print licensed terminology rows.",
        "",
        f"- Manifest source: `{payload['manifest_source']}`",
        f"- Conclusion: `{payload['conclusion']}`",
        f"- Import performed: `{payload['import_performed']}`",
        "",
        "## Canonical Source Table",
        "",
        markdown_table(["Source", "Role", "Canonical path", "Status", "Ready", "Missing"], source_rows),
        "",
        "## Readiness Table",
        "",
        markdown_table(["Readiness flag", "Value"], readiness_rows),
        "",
        "## Duplicate Candidate Warnings",
        "",
        markdown_table(["Source", "Candidate count", "Candidate paths", "Action"], warning_rows)
        if warning_rows
        else "No duplicate candidate warnings detected.",
        "",
        "## Privacy / Safety Assertions",
        "",
        markdown_table(["Assertion", "Value"], safety_rows),
        "",
        "## Next Recommended Gated Block",
        "",
        payload["next_recommended_gated_block"],
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    report_dir = Path(args.report_dir).resolve() if args.report_dir else repo_root / "reports" / "terminology_sources_preflight"
    report = build_report(repo_root, args.manifest)
    json_path = report_dir / "terminology_sources_preflight_report.json"
    md_path = report_dir / "terminology_sources_preflight_report.md"
    write_json(json_path, report)
    write_markdown(md_path, report)
    print(
        json.dumps(
            {
                "conclusion": report["conclusion"],
                "report_json": safe_rel(json_path, repo_root),
                "report_md": safe_rel(md_path, repo_root),
                "import_performed": report["import_performed"],
                "readiness": report["readiness"],
                "missing_required_primary_sources": report["missing_required_primary_sources"],
                "duplicate_warning_count": len(report["duplicate_candidate_warnings"]),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["conclusion"] == "terminology_sources_preflight_ready" else 1


if __name__ == "__main__":
    sys.exit(main())
