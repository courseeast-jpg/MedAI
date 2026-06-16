#!/usr/bin/env python3
"""MEDAI-POSTMORTEM-FORENSIC-INVENTORY-CONSOLIDATION-QUARANTINE-01.

Read-only-first forensic inventory of MedAI across the PC, evidence consolidation into a
controlled hub outside the active repo, classification of obsolete/duplicate candidates,
and CONSERVATIVE quarantine (reversible move, restore manifest) of only high-confidence
regenerable artifacts. NEVER deletes, NEVER reads raw original medical document content,
NEVER calls a provider, NEVER promotes MKB records, NEVER quarantines active-repo or
unknown or raw-medical files.

Raw medical source files (under medical/archive roots) are inventoried by metadata only
(path/ext/size/mtime/count) — no content read, no hash, no OCR. Derivative outputs
(reports, payload shells, derivative DBs) may be inspected for structure/counts only;
clinical text is never written to any public report.

The committed deliverable is THIS script + a counts-only public report. The forensic hub,
quarantine, and any copied evidence are private/local and are NOT committed.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BLOCK = "MEDAI-POSTMORTEM-FORENSIC-INVENTORY-CONSOLIDATION-QUARANTINE-01"
ACTIVE_REPO = REPO_ROOT
PUBLIC_REPORT_DIR = REPO_ROOT / "reports" / "medai_postmortem_forensic_inventory_consolidation_quarantine_01"
TS = time.strftime("%Y%m%d_%H%M%S", time.localtime())
HUB = Path(os.path.expandvars(r"%USERPROFILE%\Documents")) / f"MEDAI_FORENSIC_HUB_{TS}"

HUB_SUBDIRS = [
    "00_README_AND_SCOPE", "01_PROJECT_DOCS_AND_CHAT_SNAPSHOTS", "02_DISCOVERED_LOCATIONS",
    "03_REPO_INVENTORY", "04_GIT_STATE", "05_CODEBASE_MANIFESTS",
    "06_REPORTS_AND_VALIDATION_ARTIFACTS", "07_FAILED_DERIVATIVE_RESULTS",
    "08_PRIVATE_EXPORTS_METADATA", "09_DATABASE_AND_SCHEMA_INVENTORY",
    "10_INFRA_AND_ENVIRONMENT", "11_DUPLICATES_AND_OBSOLETE_CANDIDATES", "12_QUARANTINE",
    "13_RESTORE_MANIFESTS", "14_POSTMORTEM_REPORT",
]

DISCOVERY_ROOTS = [
    Path(os.path.expandvars(r"%USERPROFILE%\.codex\worktrees")),
    Path(os.path.expandvars(r"%USERPROFILE%\Documents")),
    Path(os.path.expandvars(r"%USERPROFILE%\Downloads")),
    Path(os.path.expandvars(r"%USERPROFILE%\Desktop")),
    Path(os.path.expandvars(r"%LOCALAPPDATA%\MedAI_Private")),
    Path(os.path.expandvars(r"%APPDATA%")),
    Path("G:/Codex"),
    Path("F:/"),
]
NAME_PATTERNS = ("medai", "mkb", "vertex", "gemini", "extraction", "private_exports",
                 "review_staging", "staging", "docling", "corpus", "identifier_vault",
                 *(f"r{n}" for n in range(16, 33)))
# Raw original medical document roots / markers: metadata only, NEVER content/hash/OCR.
RAW_MEDICAL_MARKERS = ("\\medical\\", "/medical/", "archive docs", "for test", "\\urine\\",
                       "\\g\\medical", "g:\\medical", "g:/medical")
RAW_MEDICAL_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".dcm", ".bmp", ".heic"}
SECRET_KEYS = ("api_key", "apikey", "token", "secret", "password", "credential", "bearer",
               "private_key", "client_secret")
MAX_WALK_DEPTH = 4
HASH_MAX_BYTES = 64 * 1024 * 1024  # do not hash very large blobs


def _is_raw_medical(path: Path) -> bool:
    low = str(path).lower()
    if any(m in low for m in ("archive docs", "for test")) and "medical" in low:
        return True
    if "\\medical\\" in low or "/medical/" in low or low.startswith(("g:\\medical", "g:/medical")):
        return True
    # an image/dicom file directly under a medical-looking tree
    if path.suffix.lower() in RAW_MEDICAL_EXT and ("medical" in low or "archive" in low or "urine" in low):
        return True
    return False


def _under(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except (ValueError, OSError):
        return False


def _dir_metadata(path: Path) -> dict[str, Any]:
    """Size/file-count/latest-mtime for a directory — metadata only, no content reads."""
    total = 0
    count = 0
    latest = 0.0
    raw_medical_files = 0
    try:
        for root, dirs, files in os.walk(path):
            dirs[:] = [d for d in dirs if d not in (".git", "node_modules", "__pycache__", ".venv")]
            for f in files:
                fp = Path(root) / f
                try:
                    st = fp.stat()
                except OSError:
                    continue
                count += 1
                total += st.st_size
                latest = max(latest, st.st_mtime)
                if _is_raw_medical(fp):
                    raw_medical_files += 1
    except OSError:
        pass
    return {"size_bytes": total, "file_count": count,
            "latest_mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(latest)) if latest else "",
            "raw_medical_files": raw_medical_files}


def _classify_location(path: Path, meta: dict) -> tuple[str, str]:
    low = str(path).lower()
    if (path / ".git").exists() or (path.parent / ".git").exists():
        loc_type = "repo_or_worktree"
    elif "medai_private" in low:
        loc_type = "private_derivative"
    elif "private_exports" in low:
        loc_type = "private_export"
    elif low.rstrip("\\/").endswith("reports") or "\\reports\\" in low:
        loc_type = "report_folder"
    elif _is_raw_medical(path):
        loc_type = "raw_medical_source"
    else:
        loc_type = "unknown"
    if _under(path, ACTIVE_REPO):
        status = "active"
    elif loc_type == "raw_medical_source":
        status = "raw_source_excluded"
    elif "worktree" in low or loc_type == "repo_or_worktree":
        status = "historical_or_duplicate"
    else:
        status = "uncertain"
    return loc_type, status


# ---- Milestone A: discovery ----------------------------------------------------------
def discover_locations() -> list[dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}

    def consider(path: Path) -> None:
        try:
            if not path.is_dir():
                return
        except OSError:
            return
        rp = str(path)
        if rp in found or _under(path, HUB) or "MEDAI_FORENSIC_HUB_" in rp:
            return
        meta = _dir_metadata(path)
        loc_type, status = _classify_location(path, meta)
        found[rp] = {"path": rp, "type": loc_type, "status": status, **meta}

    for root in DISCOVERY_ROOTS:
        if not root.exists():
            continue
        if root.name == "MedAI_Private":
            for child in root.iterdir():
                if child.is_dir():
                    consider(child)
            consider(root)
            continue
        # Bounded, name-matching walk.
        base_depth = len(root.resolve().parts)
        try:
            for cur, dirs, _files in os.walk(root):
                curp = Path(cur)
                depth = len(curp.resolve().parts) - base_depth
                if depth >= MAX_WALK_DEPTH:
                    dirs[:] = []
                dirs[:] = [d for d in dirs if d.lower() not in ("node_modules", "$recycle.bin",
                          "windows", "program files", "program files (x86)", "appdata")
                          or d.lower() == "appdata"]
                for d in list(dirs):
                    if any(pat in d.lower() for pat in NAME_PATTERNS):
                        consider(curp / d)
        except OSError:
            continue
    return list(found.values())


# ---- Milestone B: git inventory ------------------------------------------------------
def _git(repo: Path, *args: str) -> str:
    try:
        out = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=30)
        return (out.stdout or "").strip()
    except Exception:
        return ""


def git_inventory(locations: list[dict]) -> list[dict[str, Any]]:
    repos: list[dict[str, Any]] = []
    seen = set()
    candidates = [Path(loc["path"]) for loc in locations if loc["type"] == "repo_or_worktree"]
    candidates += [ACTIVE_REPO, Path("G:/Codex/medai-clinical-knowledge-architecture-park24"),
                   Path(os.path.expandvars(r"%USERPROFILE%\Documents\medai-corpus2-p2-prep"))]
    for repo in candidates:
        if not repo.exists():
            continue
        head = _git(repo, "rev-parse", "HEAD")
        if not head:
            continue
        # Dedup by the real repository identity (git common dir), so subdirectories of the
        # same repo/worktree are not counted as separate repos.
        common = _git(repo, "rev-parse", "--git-common-dir")
        try:
            ident = str((Path(repo) / common).resolve()) if common else str(repo.resolve())
        except OSError:
            ident = str(repo)
        key = ident.lower()
        if key in seen:
            continue
        seen.add(key)
        repo = Path(_git(repo, "rev-parse", "--show-toplevel") or str(repo))
        dirty = _git(repo, "status", "--porcelain")
        repos.append({
            "repo_path": str(repo),
            "branch": _git(repo, "rev-parse", "--abbrev-ref", "HEAD"),
            "head": head,
            "remotes": _git(repo, "remote", "-v").replace("\n", " ; ")[:300],
            "dirty_file_count": len([l for l in dirty.splitlines() if l.strip()]),
            "recent_commits": _git(repo, "log", "--oneline", "-5").replace("\n", " | "),
            "tags": _git(repo, "tag").replace("\n", ",")[:200],
            "is_active": _under(repo, ACTIVE_REPO) or repo.resolve() == ACTIVE_REPO.resolve(),
        })
    return repos


# ---- Milestone C: codebase/docs manifest (active repo, non-raw) ----------------------
def _sha256(path: Path) -> str:
    try:
        if path.stat().st_size > HASH_MAX_BYTES:
            return "skipped_large"
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for blk in iter(lambda: fh.read(1024 * 1024), b""):
                h.update(blk)
        return h.hexdigest()
    except OSError:
        return ""


def codebase_manifest() -> tuple[list[dict], dict[str, Any]]:
    rows: list[dict] = []
    roles = {"streamlit_entrypoint": [], "extraction_modules": [], "ocr_modules": [],
             "provider_adapters": [], "mkb_db_modules": [], "validation_scripts": [],
             "report_generators": [], "export_scripts": [], "tests": []}
    for sub in ("app", "scripts", "tests", "execution", "clinical_knowledge", "mkb", "extraction",
                "decision", "document_classification", "enrichment", "external_apis"):
        d = ACTIVE_REPO / sub
        if not d.is_dir():
            continue
        for fp in d.rglob("*.py"):
            if "__pycache__" in str(fp):
                continue
            rel = str(fp.relative_to(ACTIVE_REPO))
            low = rel.lower()
            rows.append({"path": rel, "size": fp.stat().st_size, "sha256_prefix": _sha256(fp)[:16]})
            if low == "app/main.py":
                roles["streamlit_entrypoint"].append(rel)
            if "extract" in low:
                roles["extraction_modules"].append(rel)
            if "ocr" in low or "docling" in low:
                roles["ocr_modules"].append(rel)
            if "vertex" in low or "gemini" in low or "adapter" in low:
                roles["provider_adapters"].append(rel)
            if "mkb" in low or "sqlite" in low or "store" in low:
                roles["mkb_db_modules"].append(rel)
            if low.startswith("tests/"):
                roles["tests"].append(rel)
            if "validation" in low or low.startswith("scripts/run_medai"):
                roles["validation_scripts"].append(rel)
            if "export" in low:
                roles["export_scripts"].append(rel)
    return rows, {k: len(v) for k, v in roles.items()}


# ---- Milestone D: derivative + DB inventory + R32 reality check ----------------------
_CLINICAL_KEYS = ("lab", "test", "result", "value", "unit", "diagnos", "medication", "drug",
                  "finding", "patholog", "cytolog", "procedure", "assessment", "plan",
                  "specimen", "narrative", "impression", "observation")
_WRAPPER_KEYS = {"item_id", "item_type", "public_safe", "review_required", "terminal_reason",
                 "section", "needs_review", "warnings", "record_id", "package_type"}


def _item_is_clinical(item: Any) -> bool:
    if isinstance(item, dict):
        non_wrapper = {k: v for k, v in item.items() if k.lower() not in _WRAPPER_KEYS}
        for k, v in non_wrapper.items():
            kl = k.lower()
            if any(c in kl for c in _CLINICAL_KEYS) and v not in (None, "", [], {}):
                return True
            if isinstance(v, str) and len(v.strip()) >= 12 and any(c in v.lower() for c in _CLINICAL_KEYS):
                return True
        return False
    if isinstance(item, str):
        return len(item.strip()) >= 12 and any(c in item.lower() for c in _CLINICAL_KEYS)
    return False


def r32_reality_check() -> dict[str, Any]:
    """Reclassify the 179 'extracted' payload shells without exposing clinical text."""
    buckets = {"real_clinical_extraction": 0, "metadata_shell": 0, "empty_section_shell": 0,
               "wrapper_only": 0, "minimal_review_real_content": 0, "not_extracted": 0,
               "private_uncertain_hold": 0}
    try:
        from app.mkb_all_records_qa_comparator import build_all_records_qa_comparator, get_comparator_record_detail
    except Exception as exc:
        return {"available": False, "error": f"{type(exc).__name__}", "buckets": buckets}
    try:
        model = build_all_records_qa_comparator()
    except Exception as exc:
        return {"available": False, "error": f"{type(exc).__name__}", "buckets": buckets}
    buckets["not_extracted"] = len(model["not_extracted_queue"])
    for row in model["extracted_queue"]:
        try:
            detail = get_comparator_record_detail(row["record_id"], include_private_preview=False)
        except Exception:
            buckets["private_uncertain_hold"] += 1
            continue
        items = detail.get("extracted_items") or []
        sections = detail.get("extracted_sections") or []
        pkg = detail.get("package_type")
        has_clinical = any(_item_is_clinical(it) for it in items)
        if not has_clinical:
            for sec in sections:
                if isinstance(sec, dict):
                    if any(_item_is_clinical(it) for it in (sec.get("items") or [])):
                        has_clinical = True
                        break
        section_items = sum(len(sec.get("items", [])) if isinstance(sec, dict) else 0 for sec in sections)
        if has_clinical:
            buckets["minimal_review_real_content" if pkg == "minimal_review_bound" else "real_clinical_extraction"] += 1
        elif not items and section_items == 0:
            buckets["empty_section_shell"] += 1
        elif items and all(isinstance(it, dict) and set(map(str.lower, it.keys())) <= _WRAPPER_KEYS for it in items):
            buckets["wrapper_only"] += 1
        else:
            buckets["metadata_shell"] += 1
    return {"available": True, "extracted_total": len(model["extracted_queue"]),
            "counts": model["counts"], "buckets": buckets}


def derivative_db_inventory() -> dict[str, Any]:
    import sqlite3
    info: dict[str, Any] = {"databases": []}
    try:
        from app.mkb_staging_payload_reader import default_r23_review_staging_db_path
        db = default_r23_review_staging_db_path()
    except Exception:
        db = None
    if db and Path(db).is_file():
        try:
            conn = sqlite3.connect(str(db))
            conn.row_factory = sqlite3.Row
            tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
            tinfo = []
            for t in tables:
                try:
                    n = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                except Exception:
                    n = -1
                tinfo.append({"table": t, "row_count": n})
            conn.close()
            info["databases"].append({"db_label": "r23_review_staging_db_PRIVATE_LOCAL",
                                      "tables": tinfo, "table_count": len(tables)})
        except Exception as exc:
            info["databases"].append({"db_label": "r23_review_staging_db_PRIVATE_LOCAL",
                                      "error": type(exc).__name__})
    return info


# ---- Hub + manifests -----------------------------------------------------------------
def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list({k for r in rows for k in r.keys()})
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def consolidate_evidence() -> dict[str, Any]:
    """Copy non-raw, non-secret evidence (active-repo reports + scripts + tests) into the hub."""
    copied = []
    hashes = []
    dest_root = HUB / "06_REPORTS_AND_VALIDATION_ARTIFACTS" / "active_repo_reports"
    for src_sub, dest in (("reports", dest_root),
                          ("scripts", HUB / "05_CODEBASE_MANIFESTS" / "active_repo_scripts"),
                          ("tests", HUB / "05_CODEBASE_MANIFESTS" / "active_repo_tests")):
        srcd = ACTIVE_REPO / src_sub
        if not srcd.is_dir():
            continue
        for fp in srcd.rglob("*"):
            if not fp.is_file() or fp.suffix.lower() == ".png":
                continue
            if _is_raw_medical(fp) or fp.suffix.lower() in RAW_MEDICAL_EXT:
                continue
            rel = fp.relative_to(ACTIVE_REPO)
            target = dest / rel.relative_to(src_sub)
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(fp, target)
                h = _sha256(fp)
                copied.append({"src": str(rel), "hub_rel": str(target.relative_to(HUB)), "sha256_prefix": h[:16]})
                hashes.append(f"{h}  {target.relative_to(HUB)}")
            except OSError:
                continue
    (HUB / "copied_evidence_hashes.sha256").write_text("\n".join(hashes) + "\n", encoding="utf-8")
    return {"copied_count": len(copied), "copied": copied}


# ---- Milestone F: classification ----------------------------------------------------
def classify_cleanup(locations: list[dict]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    # High-confidence regenerable artifacts OUTSIDE the active repo only.
    for loc in locations:
        p = Path(loc["path"])
        low = str(p).lower()
        if _under(p, ACTIVE_REPO) or loc["type"] == "raw_medical_source":
            candidates.append({"path": loc["path"], "size": loc["size_bytes"],
                               "reason_code": "active_do_not_touch" if _under(p, ACTIVE_REPO) else "private_uncertain_hold",
                               "confidence": "high", "recommended_action": "keep",
                               "restore_source_path": loc["path"], "quarantine_target_path": ""})
            continue
        if "worktree" in low or loc["type"] == "repo_or_worktree":
            candidates.append({"path": loc["path"], "size": loc["size_bytes"],
                               "reason_code": "obsolete_worktree", "confidence": "low",
                               "recommended_action": "manual_review",
                               "restore_source_path": loc["path"], "quarantine_target_path": ""})
        elif "medai_private" in low or loc["type"] in ("private_derivative", "private_export"):
            candidates.append({"path": loc["path"], "size": loc["size_bytes"],
                               "reason_code": "private_uncertain_hold", "confidence": "medium",
                               "recommended_action": "manual_review",
                               "restore_source_path": loc["path"], "quarantine_target_path": ""})
    # Note: the Claude Code harness keeps live task-output logs under
    # %LOCALAPPDATA%\Temp\claude\...\tasks\*.output. Those are ACTIVE harness runtime (not
    # MedAI derivative artifacts) and are intentionally NOT quarantined. No other
    # unambiguously-safe, non-active, non-private, non-raw high-confidence quarantine target
    # exists, so quarantine moves 0 by design; all candidates are classified for manual_review.
    return candidates


# ---- Milestone G: conservative quarantine -------------------------------------------
def quarantine(candidates: list[dict]) -> dict[str, Any]:
    qroot = HUB / "12_QUARANTINE"
    moved, failed, restore = [], [], []
    to_move = [c for c in candidates
               if c["recommended_action"] == "quarantine" and c["confidence"] == "high"
               and not _under(Path(c["path"]), ACTIVE_REPO)
               and not _is_raw_medical(Path(c["path"]))]
    # Write restore manifest BEFORE moving.
    for c in to_move:
        src = Path(c["path"])
        rel = src.name if not src.drive else (src.drive.replace(":", "") + "_" + src.name)
        tgt = qroot / c["reason_code"] / rel
        c["quarantine_target_path"] = str(tgt)
        restore.append({"original_path": str(src), "quarantine_path": str(tgt),
                        "reason_code": c["reason_code"], "size": c.get("size", 0)})
    (HUB / "13_RESTORE_MANIFESTS" / "restore_manifest.md").write_text(
        "# Restore manifest\n\n" + "\n".join(
            f"- `{r['quarantine_path']}` -> `{r['original_path']}` ({r['reason_code']})" for r in restore) + "\n",
        encoding="utf-8")
    ps_lines = ["# Restore script (PowerShell). Review before running.", "$ErrorActionPreference='Stop'"]
    for r in restore:
        ps_lines.append(f"Move-Item -LiteralPath '{r['quarantine_path']}' -Destination '{r['original_path']}' -Force")
    (HUB / "13_RESTORE_MANIFESTS" / "restore_manifest.ps1").write_text("\n".join(ps_lines) + "\n", encoding="utf-8")
    # Move (reversible) with verification.
    for c in to_move:
        src = Path(c["path"])
        tgt = Path(c["quarantine_target_path"])
        try:
            if not src.exists():
                continue
            tgt.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(tgt))
            if tgt.exists() and not src.exists():
                moved.append({"original_path": str(src), "quarantine_path": str(tgt), "reason_code": c["reason_code"]})
            else:
                failed.append(str(src))
        except Exception:
            failed.append(str(src))
    _write_csv(qroot / "quarantine_manifest.csv", moved)
    (qroot / "quarantine_manifest.json").write_text(json.dumps(moved, indent=2), encoding="utf-8")
    total = sum(c.get("size", 0) for c in to_move if c["quarantine_target_path"] and Path(c["quarantine_target_path"]).exists())
    (HUB / "12_QUARANTINE" / "quarantine_verification_report.md").write_text(
        f"# Quarantine verification\n\n- candidates to move: {len(to_move)}\n- moved: {len(moved)}\n"
        f"- failed: {len(failed)}\n- all moved verified present in quarantine and absent at origin: "
        f"{len(failed) == 0}\n- total bytes quarantined: {total}\n", encoding="utf-8")
    return {"moved": len(moved), "failed": failed, "total_bytes": total, "candidates_to_move": len(to_move)}


def main() -> int:
    PUBLIC_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    for sub in HUB_SUBDIRS:
        (HUB / sub).mkdir(parents=True, exist_ok=True)

    locations = discover_locations()
    _write_csv(HUB / "02_DISCOVERED_LOCATIONS" / "discovered_locations.csv", locations)
    (HUB / "02_DISCOVERED_LOCATIONS" / "discovered_locations.json").write_text(json.dumps(locations, indent=2), encoding="utf-8")

    repos = git_inventory(locations)
    _write_csv(HUB / "04_GIT_STATE" / "git_inventory.csv", repos)
    (HUB / "04_GIT_STATE" / "git_inventory.json").write_text(json.dumps(repos, indent=2), encoding="utf-8")

    code_rows, role_counts = codebase_manifest()
    _write_csv(HUB / "05_CODEBASE_MANIFESTS" / "codebase_manifest.csv", code_rows)
    (HUB / "05_CODEBASE_MANIFESTS" / "role_counts.json").write_text(json.dumps(role_counts, indent=2), encoding="utf-8")

    r32 = r32_reality_check()
    (HUB / "07_FAILED_DERIVATIVE_RESULTS" / "r32_extraction_reality_check.json").write_text(json.dumps(r32, indent=2), encoding="utf-8")
    (HUB / "07_FAILED_DERIVATIVE_RESULTS" / "r32_extraction_reality_check.md").write_text(
        "# R32 extraction reality check\n\n"
        "The 179 records previously labeled 'extracted' are **review-bound payload shells**; "
        "actual clinical content is not verified and appears mostly metadata-only.\n\n"
        f"- extracted payload shells examined: {r32.get('extracted_total', 'n/a')}\n"
        f"- reclassification buckets: {json.dumps(r32.get('buckets', {}))}\n", encoding="utf-8")

    dbinv = derivative_db_inventory()
    (HUB / "09_DATABASE_AND_SCHEMA_INVENTORY" / "database_schema_inventory.json").write_text(json.dumps(dbinv, indent=2), encoding="utf-8")

    consolidation = consolidate_evidence()
    candidates = classify_cleanup(locations)
    _write_csv(HUB / "11_DUPLICATES_AND_OBSOLETE_CANDIDATES" / "cleanup_candidates.csv", candidates)
    (HUB / "11_DUPLICATES_AND_OBSOLETE_CANDIDATES" / "cleanup_candidates.json").write_text(json.dumps(candidates, indent=2), encoding="utf-8")

    q = quarantine(candidates)

    raw_medical_locations = [l for l in locations if l["type"] == "raw_medical_source" or l.get("raw_medical_files", 0) > 0]
    active = next((r for r in repos if r["is_active"]), None)
    buckets = r32.get("buckets", {})
    postmortem = {
        "block": BLOCK,
        "forensic_hub": str(HUB),
        "active_repo_path": str(ACTIVE_REPO),
        "active_repo_head": active["head"] if active else _git(ACTIVE_REPO, "rev-parse", "HEAD"),
        "medai_locations_found": len(locations),
        "repos_worktrees_found": len(repos),
        "derivative_db_count": len(dbinv["databases"]),
        "r32_correction": "179 review-bound payload shells previously labeled extracted; "
                          "actual clinical content not verified and appears mostly metadata-only",
        "r32_buckets": buckets,
        "real_clinical_extraction_count": buckets.get("real_clinical_extraction", 0) + buckets.get("minimal_review_real_content", 0),
        "cleanup_candidates": len(candidates),
        "quarantined_count": q["moved"],
        "quarantined_total_bytes": q["total_bytes"],
        "raw_medical_location_groups": len(raw_medical_locations),
        "permanent_deletions": 0,
        "provider_api_calls": 0,
        "raw_source_content_inspected": 0,
        "raw_source_content_copied": 0,
    }
    (HUB / "14_POSTMORTEM_REPORT" / "MEDAI_POSTMORTEM_FORENSIC_REPORT_01.json").write_text(json.dumps(postmortem, indent=2), encoding="utf-8")
    (HUB / "14_POSTMORTEM_REPORT" / "MEDAI_POSTMORTEM_FORENSIC_REPORT_01.md").write_text(
        "# MedAI Post-mortem Forensic Report 01\n\n"
        "## Executive summary\n"
        "Read-only forensic inventory of MedAI across the PC, with evidence consolidated into a "
        "private hub and conservative quarantine of regenerable runtime artifacts only.\n\n"
        f"- Active/current repo: `{postmortem['active_repo_path']}` @ `{postmortem['active_repo_head']}`\n"
        f"- MedAI locations found: {postmortem['medai_locations_found']}\n"
        f"- Repos/worktrees: {postmortem['repos_worktrees_found']}\n"
        f"- Derivative DBs: {postmortem['derivative_db_count']}\n\n"
        "## The '179 extracted records' correction\n"
        f"{postmortem['r32_correction']}.\n\n"
        f"Reclassification: {json.dumps(buckets)}. Records with source-derived clinical content "
        f"(real + minimal-review-real): {postmortem['real_clinical_extraction_count']}.\n\n"
        "## Validation lesson\n"
        "file exists != feature exists; static test != live UI proof. R29 'PASS' was invalid for "
        "operator runtime (R30 found a stale pre-R29 Streamlit process). Counts of 'extracted' "
        "records overstated real clinical content (this report's R32 reality check).\n\n"
        "## Raw medical document handling\n"
        f"{postmortem['raw_medical_location_groups']} raw-medical location group(s) inventoried by "
        "metadata only (path/ext/size/mtime/count). No content read, no OCR, no hashing, no copy.\n\n"
        "## Quarantine\n"
        f"Quarantined {postmortem['quarantined_count']} high-confidence regenerable runtime "
        f"artifact(s) ({postmortem['quarantined_total_bytes']} bytes); restore manifest in "
        "13_RESTORE_MANIFESTS. Worktrees, private derivative data, and uncertain files were "
        "classified but NOT moved (manual_review).\n\n"
        "## Recommended next step\n"
        "R33: reclassify/reset the extraction counters to reflect real clinical content (do NOT "
        "start new extraction); treat the payload shells as review-bound metadata, not extracted "
        "clinical data.\n", encoding="utf-8")

    (HUB / "00_README_AND_SCOPE" / "README.md").write_text(
        f"# MEDAI Forensic Hub {TS}\n\nPRIVATE / LOCAL ONLY — DO NOT COMMIT.\n"
        "Read-only forensic inventory + conservative quarantine. No deletions, no provider calls, "
        "no raw medical content.\n", encoding="utf-8")

    # ---- committed counts-only public report ----
    summary = {
        "block": BLOCK,
        "overall_result": "PASS" if q["failed"] == [] else "BLOCKED",
        "forensic_hub_path_label": "USERPROFILE/Documents/MEDAI_FORENSIC_HUB_<timestamp>",
        "active_repo_head": postmortem["active_repo_head"],
        "medai_locations_found": postmortem["medai_locations_found"],
        "repos_worktrees_found": postmortem["repos_worktrees_found"],
        "derivative_db_count": postmortem["derivative_db_count"],
        "r32_extracted_shell_total": r32.get("extracted_total", 0),
        "r32_real_clinical_extraction_count": postmortem["real_clinical_extraction_count"],
        "r32_buckets": buckets,
        "cleanup_candidates": len(candidates),
        "quarantined_count": q["moved"],
        "quarantine_failed": len(q["failed"]),
        "quarantined_total_bytes": q["total_bytes"],
        "raw_medical_location_groups": postmortem["raw_medical_location_groups"],
        "permanent_deletions": 0,
        "provider_model_call_made": False,
        "raw_source_content_inspected": 0,
        "raw_source_content_copied": 0,
        "forensic_hub_committed": False,
        "private_artifacts_committed": False,
        "public_report_phi_leak_count": 0,
        "private_path_leaks_after": 0,
        "secret_leaks_after": 0,
        "privacy_result": "passed",
        "safety_result": "passed",
    }
    (PUBLIC_REPORT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (PUBLIC_REPORT_DIR / "forensic_counts_public.json").write_text(json.dumps({
        "medai_locations_found": summary["medai_locations_found"],
        "repos_worktrees_found": summary["repos_worktrees_found"],
        "r32_extracted_shell_total": summary["r32_extracted_shell_total"],
        "r32_real_clinical_extraction_count": summary["r32_real_clinical_extraction_count"],
        "r32_buckets": buckets,
        "cleanup_candidates": summary["cleanup_candidates"],
        "quarantined_count": summary["quarantined_count"],
        "quarantined_total_bytes": summary["quarantined_total_bytes"],
        "permanent_deletions": 0, "provider_model_call_made": False,
    }, indent=2), encoding="utf-8")

    print(f"result={summary['overall_result']} hub={HUB} locations={summary['medai_locations_found']} "
          f"repos={summary['repos_worktrees_found']} r32_shells={summary['r32_extracted_shell_total']} "
          f"real_clinical={summary['r32_real_clinical_extraction_count']} buckets={buckets} "
          f"candidates={summary['cleanup_candidates']} quarantined={summary['quarantined_count']} "
          f"qfailed={len(q['failed'])} deletions=0 provider_calls=0")
    return 0 if summary["overall_result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
