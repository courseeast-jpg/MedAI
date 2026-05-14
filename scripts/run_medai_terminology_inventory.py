"""Report-only inventory for local MedAI terminology_data packages.

This script inspects file metadata only. It does not import terminology data,
does not read licensed terminology rows, and does not read
LICENSE_ACK_PRIVATE.json contents.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = REPO_ROOT / "reports" / "terminology_data_inventory"
REPORT_JSON = REPORT_DIR / "terminology_data_inventory_report.json"
REPORT_MD = REPORT_DIR / "terminology_data_inventory_report.md"

KEY_FILE_PATTERNS = {
    "loinc": ("Loinc.csv", "Loinc_*.zip"),
    "rxnorm": ("RXNCONSO.RRF", "RXNREL.RRF", "RXNSAT.RRF", "RXNSAB.RRF", "RXNSTY.RRF"),
    "umls": ("MRCONSO.RRF", "MRSTY.RRF", "MRSAB.RRF", "MRREL.RRF", "mmsys.zip", "*.nlm"),
    "snomed_ct": (
        "sct2_Concept*.txt",
        "sct2_Description*.txt",
        "sct2_Relationship*.txt",
        "release_package_information.json",
    ),
    "license_ack_private": ("LICENSE_ACK_PRIVATE.json",),
}

REQUIRED_GITIGNORE_PATTERNS = (
    "terminology_data/",
    "LICENSE_ACK_PRIVATE.json",
    "*.RRF",
    "*.rrf",
    "*.nlm",
    "*.zip",
    "data/terminology/",
    "*.sqlite",
    "*.sqlite3",
    "*.db",
)

LICENSE_ACK_NAME = "LICENSE_ACK_PRIVATE.json"


@dataclass(frozen=True)
class FileEntry:
    relative_path: str
    parent: str
    name: str
    extension: str
    size_bytes: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a sanitized MedAI terminology_data inventory.")
    parser.add_argument(
        "--terminology-root",
        default="terminology_data",
        help="Terminology root relative to the repository root, or an absolute local path.",
    )
    parser.add_argument("--repo-root", default=str(REPO_ROOT), help="Repository root used for relative report paths.")
    parser.add_argument("--report-dir", default=None, help="Optional report output directory.")
    return parser.parse_args()


def resolve_terminology_root(value: str) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return candidate.resolve()


def safe_rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        # Keep reports path-only and non-absolute even if the operator passes a
        # different local root. The configured root label avoids local drive leaks.
        return path.name


def iter_file_entries(root: Path) -> list[FileEntry]:
    entries: list[FileEntry] = []
    if not root.exists():
        return entries
    for path in sorted(root.rglob("*"), key=lambda p: safe_rel(p).lower()):
        if not path.is_file():
            continue
        rel = safe_rel(path)
        parent = Path(rel).parent.as_posix()
        suffix = path.suffix.lower() if path.suffix else "<none>"
        size = path.stat().st_size
        entries.append(
            FileEntry(
                relative_path=rel,
                parent=parent,
                name=path.name,
                extension=suffix,
                size_bytes=size,
            )
        )
    return entries


def count_folders(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(1 for path in root.rglob("*") if path.is_dir())


def detect_key_files(root: Path, entries: Iterable[FileEntry]) -> dict[str, list[dict[str, object]]]:
    by_system: dict[str, list[dict[str, object]]] = {system: [] for system in KEY_FILE_PATTERNS}
    for entry in entries:
        if entry.name == LICENSE_ACK_NAME:
            by_system["license_ack_private"].append(
                {
                    "relative_path": entry.relative_path,
                    "presence_only": True,
                    "contents_read": False,
                    "size_bytes": entry.size_bytes,
                }
            )
            continue

        entry_path = root / Path(entry.relative_path).relative_to(safe_rel(root))
        for system, patterns in KEY_FILE_PATTERNS.items():
            if system == "license_ack_private":
                continue
            if any(entry_path.match(pattern) or entry.name.lower() == pattern.lower() for pattern in patterns):
                by_system[system].append(
                    {
                        "relative_path": entry.relative_path,
                        "file_name": entry.name,
                        "size_bytes": entry.size_bytes,
                    }
                )
    return by_system


def folder_rows(root: Path, entries: list[FileEntry]) -> list[dict[str, object]]:
    grouped: dict[str, dict[str, object]] = defaultdict(lambda: {"file_count": 0, "total_size_bytes": 0, "extensions": Counter()})
    if not root.exists():
        return []

    root_rel = safe_rel(root)
    for entry in entries:
        rel = Path(entry.relative_path)
        parts = rel.parts
        top = root_rel
        if len(parts) >= 3:
            top = f"{parts[0]}/{parts[1]}"
        grouped[top]["file_count"] += 1
        grouped[top]["total_size_bytes"] += entry.size_bytes
        grouped[top]["extensions"][entry.extension] += 1

    rows = []
    for folder, data in sorted(grouped.items(), key=lambda item: item[0].lower()):
        rows.append(
            {
                "folder": folder,
                "file_count": data["file_count"],
                "total_size_bytes": data["total_size_bytes"],
                "extensions": dict(sorted(data["extensions"].items())),
            }
        )
    return rows


def classify_packages(folder_inventory: list[dict[str, object]], key_files: dict[str, list[dict[str, object]]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    folder_names = [str(row["folder"]) for row in folder_inventory]

    def add(system: str, label: str, status: str, candidates: list[str], recommendation: str) -> None:
        rows.append(
            {
                "system": system,
                "label": label,
                "status": status,
                "candidate_folders": candidates,
                "recommendation": recommendation,
            }
        )

    loinc_candidates = [f for f in folder_names if "loinc" in f.lower()]
    rxnorm_candidates = [f for f in folder_names if "rxnorm" in f.lower()]
    rxnorm_full = [f for f in rxnorm_candidates if "full" in f.lower() and "prescrib" not in f.lower()]
    rxnorm_prescribe = [f for f in rxnorm_candidates if "prescrib" in f.lower()]
    umls_candidates = [f for f in folder_names if "umls" in f.lower()]
    snomed_us = [f for f in folder_names if "snomed" in f.lower() and ("us" in f.lower() or "managedserviceus" in f.lower())]
    snomed_int = [f for f in folder_names if "snomed" in f.lower() and ("international" in f.lower() or "intl" in f.lower())]

    add(
        "loinc",
        "LOINC main package",
        "present" if key_files["loinc"] else "missing",
        loinc_candidates,
        "Use the folder containing Loinc.csv as the canonical LOINC source.",
    )
    add(
        "rxnorm",
        "RxNorm full monthly release",
        "present" if any(item["file_name"] == "RXNCONSO.RRF" for item in key_files["rxnorm"]) else "missing",
        rxnorm_full or rxnorm_candidates,
        "Use the full monthly release as canonical; keep prescribable subset auxiliary.",
    )
    add(
        "rxnorm_prescribable_subset",
        "RxNorm prescribable subset",
        "auxiliary_present" if rxnorm_prescribe else "not_detected",
        rxnorm_prescribe,
        "Treat as auxiliary only; avoid replacing the full RxNorm release with this subset.",
    )
    add(
        "umls",
        "UMLS full distribution or generated RRF subset",
        "present" if key_files["umls"] else "missing",
        umls_candidates,
        "Keep separate from RxNorm/LOINC imports unless a future licensed UMLS import is explicitly approved.",
    )
    add(
        "snomed_ct_us",
        "SNOMED CT US Edition",
        "present" if snomed_us else "missing",
        snomed_us,
        "Prefer US Edition as primary SNOMED source for US clinical workflows.",
    )
    add(
        "snomed_ct_international",
        "SNOMED CT International",
        "secondary_present" if snomed_int else "not_detected",
        snomed_int,
        "Treat International Edition as secondary unless a future import plan selects it explicitly.",
    )
    add(
        "license_ack_private",
        "Private license acknowledgment",
        "present" if key_files["license_ack_private"] else "missing",
        ["terminology_data"] if key_files["license_ack_private"] else [],
        "Presence only was checked; contents were not read.",
    )
    return rows


def duplicate_warnings(package_inventory: list[dict[str, object]]) -> list[str]:
    warnings: list[str] = []
    for row in package_inventory:
        candidates = row.get("candidate_folders") or []
        if isinstance(candidates, list) and len(candidates) > 1:
            warnings.append(
                f"{row['system']} has {len(candidates)} candidate folders; select one canonical folder before import changes."
            )
    return warnings


def load_gitignore_lines() -> list[str]:
    path = REPO_ROOT / ".gitignore"
    if not path.exists():
        return []
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def gitignore_inventory(lines: list[str]) -> dict[str, object]:
    line_set = set(lines)
    checks = []
    for pattern in REQUIRED_GITIGNORE_PATTERNS:
        direct = pattern in line_set
        protected_by_scope = False
        if pattern not in {"terminology_data/", "data/terminology/"}:
            protected_by_scope = "terminology_data/" in line_set or "data/terminology/" in line_set
        wildcard_license = pattern == "LICENSE_ACK_PRIVATE.json" and any("LICENSE_ACK_PRIVATE" in line for line in lines)
        protected = direct or protected_by_scope or wildcard_license
        checks.append(
            {
                "pattern": pattern,
                "present_directly": direct,
                "protected": protected,
                "protection_note": "direct" if direct else ("covered_by_existing_scope_or_wildcard" if protected else "missing"),
            }
        )
    return {
        "checks": checks,
        "missing_patterns": [check["pattern"] for check in checks if not check["protected"]],
        "direct_pattern_recommendations": [check["pattern"] for check in checks if not check["present_directly"]],
    }


def format_bytes(size: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{size} B"


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    table = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        table.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(table)


def build_report(terminology_root: Path) -> dict[str, object]:
    entries = iter_file_entries(terminology_root)
    folder_count = count_folders(terminology_root)
    total_size = sum(entry.size_bytes for entry in entries)
    extension_counts = Counter(entry.extension for entry in entries)
    key_files = detect_key_files(terminology_root, entries)
    key_file_counts = {system: len(files) for system, files in key_files.items()}
    folders = folder_rows(terminology_root, entries)
    packages = classify_packages(folders, key_files)
    package_inventory_public = [
        {
            "system": row["system"],
            "label": row["label"],
            "status": row["status"],
            "candidate_count": len(row.get("candidate_folders") or []),
            "recommendation": row["recommendation"],
        }
        for row in packages
    ]
    gitignore = gitignore_inventory(load_gitignore_lines())

    return {
        "report_id": "medai_terminology_data_inventory",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "terminology_root": safe_rel(terminology_root),
        "report_scope": "metadata_only_no_import",
        "safety": {
            "terminology_import_performed": False,
            "terminology_files_modified": False,
            "license_ack_contents_read": False,
            "licensed_rows_printed": False,
            "absolute_paths_in_report": False,
            "external_api_used": False,
        },
        "summary": {
            "root_exists": terminology_root.exists(),
            "folder_count": folder_count,
            "file_count": len(entries),
            "total_size_human": format_bytes(total_size),
            "extension_counts": dict(sorted(extension_counts.items())),
        },
        "key_file_counts": key_file_counts,
        "package_inventory": package_inventory_public,
        "duplicate_parallel_folder_warnings": duplicate_warnings(packages),
        "gitignore_protection": gitignore,
        "canonical_folder_recommendations": [
            "LOINC: use the folder containing Loinc.csv as canonical.",
            "RxNorm: use the full monthly release as canonical; keep prescribable subset auxiliary.",
            "UMLS: keep separate until a licensed UMLS-specific import block is approved.",
            "SNOMED CT: prefer US Edition as primary; keep International Edition secondary unless explicitly selected.",
        ],
        "next_recommended_medai_codebase_update_plan": [
            "No import behavior changes should be made from this inventory alone.",
            "Confirm canonical folders with the operator before any import or adapter expansion.",
            "If SNOMED or UMLS support is approved later, create a separate gated parser/import block with synthetic tests first.",
            "Keep reports public-safe and keep licensed files and runtime indexes untracked.",
        ],
    }


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_markdown(path: Path, payload: dict[str, object]) -> None:
    summary = payload["summary"]
    safety = payload["safety"]
    packages = payload["package_inventory"]
    gitignore = payload["gitignore_protection"]
    key_file_counts = payload["key_file_counts"]

    product_rows = [
        [row["system"], row["label"], row["status"], row["candidate_count"], row["recommendation"]]
        for row in packages
    ]
    key_rows = []
    for system, count in key_file_counts.items():
        key_rows.append([system, count, "presence only" if system == "license_ack_private" else "metadata only"])
    gitignore_rows = [
        [check["pattern"], check["present_directly"], check["protected"], check["protection_note"]]
        for check in gitignore["checks"]
    ]

    lines = [
        "# MedAI Terminology Data Inventory",
        "",
        "This is a report-only inventory. It does not import terminology data, modify terminology files, read private license acknowledgment contents, or print licensed terminology rows.",
        "",
        "## Summary",
        "",
        f"- Terminology root: `{payload['terminology_root']}`",
        f"- Root exists: `{summary['root_exists']}`",
        f"- Folders: `{summary['folder_count']}`",
        f"- Files: `{summary['file_count']}`",
        f"- Total size: `{summary['total_size_human']}`",
        f"- External API used: `{safety['external_api_used']}`",
        f"- Import performed: `{safety['terminology_import_performed']}`",
        f"- License acknowledgment contents read: `{safety['license_ack_contents_read']}`",
        "",
        "## Product Readiness Table",
        "",
        markdown_table(["System", "Package", "Status", "Candidate count", "Recommendation"], product_rows),
        "",
        "## Key File Detection",
        "",
        markdown_table(["System", "Detected key files", "Mode"], key_rows),
        "",
        "## Canonical Folder Recommendations",
        "",
    ]
    lines.extend(f"- {item}" for item in payload["canonical_folder_recommendations"])
    lines.extend(["", "## Duplicate / Parallel Folder Warnings", ""])
    warnings = payload["duplicate_parallel_folder_warnings"]
    lines.extend(f"- {item}" for item in warnings) if warnings else lines.append("- No duplicate or parallel folder warnings detected.")
    lines.extend(["", "## .gitignore Protection", ""])
    lines.append(markdown_table(["Pattern", "Present directly", "Protected", "Note"], gitignore_rows))
    lines.extend(["", "## Missing .gitignore Patterns", ""])
    if gitignore["missing_patterns"]:
        lines.extend(f"- {pattern}" for pattern in gitignore["missing_patterns"])
    else:
        lines.append("- No unprotected required patterns detected.")
    lines.extend(["", "## Direct .gitignore Pattern Recommendations", ""])
    if gitignore["direct_pattern_recommendations"]:
        lines.extend(f"- Consider adding `{pattern}` explicitly for clearer future protection." for pattern in gitignore["direct_pattern_recommendations"])
    else:
        lines.append("- All requested patterns are present directly.")
    lines.extend(["", "## Next Recommended MedAI Codebase Update Plan", ""])
    lines.extend(f"- {item}" for item in payload["next_recommended_medai_codebase_update_plan"])
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    global REPO_ROOT, REPORT_DIR, REPORT_JSON, REPORT_MD
    args = parse_args()
    REPO_ROOT = Path(args.repo_root).resolve()
    if args.report_dir:
        REPORT_DIR = Path(args.report_dir).resolve()
        REPORT_JSON = REPORT_DIR / "terminology_data_inventory_report.json"
        REPORT_MD = REPORT_DIR / "terminology_data_inventory_report.md"
    else:
        REPORT_DIR = REPO_ROOT / "reports" / "terminology_data_inventory"
        REPORT_JSON = REPORT_DIR / "terminology_data_inventory_report.json"
        REPORT_MD = REPORT_DIR / "terminology_data_inventory_report.md"
    terminology_root = resolve_terminology_root(args.terminology_root)
    report = build_report(terminology_root)
    write_json(REPORT_JSON, report)
    write_markdown(REPORT_MD, report)
    detected = {
        row["system"]: row["status"]
        for row in report["package_inventory"]
        if row["status"] not in {"missing", "not_detected"}
    }
    print(
        json.dumps(
            {
                "conclusion": "terminology_data_inventory_report_ready",
                "report_json": safe_rel(REPORT_JSON),
                "report_md": safe_rel(REPORT_MD),
                "packages_detected": detected,
                "terminology_import_performed": False,
                "license_ack_contents_read": False,
                "absolute_paths_in_report": False,
                "missing_gitignore_patterns": report["gitignore_protection"]["missing_patterns"],
                "direct_gitignore_pattern_recommendations": report["gitignore_protection"]["direct_pattern_recommendations"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
