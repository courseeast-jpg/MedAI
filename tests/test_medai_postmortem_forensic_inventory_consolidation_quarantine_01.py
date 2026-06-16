"""Tests for the MedAI post-mortem forensic inventory public report.

Validates the committed counts-only public report and safety invariants. The forensic hub
itself is private/local and lives outside the repo (not asserted here beyond non-commit).
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = REPO_ROOT / "reports" / "medai_postmortem_forensic_inventory_consolidation_quarantine_01"
PUBLIC_REPORTS = ("summary.json", "forensic_counts_public.json")
SECRET_PATTERNS = [
    re.compile(r"AIza[0-9A-Za-z_-]{20,}"), re.compile(r"ya29\."),
    re.compile(r"Bearer\s+[A-Za-z0-9._-]+", re.IGNORECASE),
    re.compile(r"(?i)(api_key|client_secret|password|private_key)\s*[=:]\s*\S"),
    re.compile(r"[A-Za-z]:\\\\Users\\\\[^\\\\]+\\\\"),  # any raw private Windows user path
    re.compile(r"\bMRN\b"), re.compile(r"\bDOB\b"),
]


def _summary() -> dict:
    return json.loads((REPORT_DIR / "summary.json").read_text(encoding="utf-8"))


def test_public_reports_exist():
    for n in PUBLIC_REPORTS:
        assert (REPORT_DIR / n).exists(), n


def test_overall_pass():
    s = _summary()
    assert s["overall_result"] == "PASS"
    assert s["quarantine_failed"] == 0


def test_r32_reality_correction():
    s = _summary()
    # The 179 "extracted" records are review-bound shells; real clinical extraction is ~0.
    assert s["r32_extracted_shell_total"] == 179
    assert s["r32_buckets"]["not_extracted"] == 317
    assert s["r32_real_clinical_extraction_count"] == (
        s["r32_buckets"]["real_clinical_extraction"] + s["r32_buckets"]["minimal_review_real_content"])
    # Sum of extracted-side buckets equals the 179 shells.
    extracted_side = sum(v for k, v in s["r32_buckets"].items() if k != "not_extracted")
    assert extracted_side == 179


def test_safety_invariants():
    s = _summary()
    assert s["permanent_deletions"] == 0
    assert s["provider_model_call_made"] is False
    assert s["raw_source_content_inspected"] == 0
    assert s["raw_source_content_copied"] == 0
    assert s["forensic_hub_committed"] is False
    assert s["private_artifacts_committed"] is False


def test_counts_present():
    s = _summary()
    assert s["medai_locations_found"] >= 1
    assert s["repos_worktrees_found"] >= 1
    assert s["cleanup_candidates"] >= 0
    assert s["quarantined_count"] >= 0
    assert s["quarantined_total_bytes"] >= 0


def test_forensic_hub_not_committed():
    out = subprocess.run(["git", "ls-files"], cwd=REPO_ROOT, capture_output=True, text=True, timeout=30)
    tracked = out.stdout
    assert "MEDAI_FORENSIC_HUB_" not in tracked
    assert "private_exports/" not in tracked or not any(
        l.startswith("private_exports/") for l in tracked.splitlines())


def test_public_reports_no_secrets_or_private_paths():
    s = _summary()
    assert s["public_report_phi_leak_count"] == 0
    assert s["private_path_leaks_after"] == 0
    assert s["secret_leaks_after"] == 0
    for n in PUBLIC_REPORTS:
        text = (REPORT_DIR / n).read_text(encoding="utf-8", errors="ignore")
        for pat in SECRET_PATTERNS:
            assert not pat.search(text), (n, pat.pattern)
        assert not re.search(r"\b\d{3}[- ]\d{2}[- ]\d{4}\b", text)
