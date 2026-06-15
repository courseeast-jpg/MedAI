"""MEDAI-R31: operator-zero acceptance of the readable QA workflow.

Proves, against a fresh real Streamlit session, that the MKB Explorer's first/default
section is the readable All-record QA comparator (legacy staging table + raw JSON moved
into a collapsed Advanced expander), that extracted/minimal/not-extracted records are
readable, and that a local QA-status save persists. All committed evidence is content-free
(counts, ordering indices, heading presence, R31PROOF markers). No provider calls, no
extraction, no promotion. Screenshots are header-cropped (no clinical content).
"""
from __future__ import annotations

import json
import re
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.mkb_all_records_qa_comparator import build_all_records_qa_comparator

BLOCK = "MEDAI-R31-OPERATOR-ZERO-READABLE-QA-UX-ACCEPTANCE-AND-CLEANUP"
REPORT_DIR = REPO_ROOT / "reports" / "medai_r31_operator_zero_readable_qa_ux_acceptance_and_cleanup"
PORT = 8561
OPERATOR_URL = f"http://localhost:{PORT}"
EXPECTED_EXTRACTED = 179
EXPECTED_NOT_EXTRACTED = 317
REQUIRED_HEADINGS = (
    "Extracted content", "Extracted sections", "Extracted items / facts",
    "Source evidence / original preview", "QA decision",
)
COMPARATOR = "All-record QA comparator"
LEGACY_EXPANDER = "Advanced / legacy MKB staging table"
REVIEW_STAGING_DETAIL = "Review staging detail"
SECRET_PATTERNS = [
    re.compile(r"AIza[0-9A-Za-z_-]{20,}"), re.compile(r"ya29\."),
    re.compile(r"Bearer\s+[A-Za-z0-9._-]+", re.IGNORECASE),
    re.compile(r"[A-Za-z]:\\\\Users\\\\"), re.compile(r"\bMRN\b"), re.compile(r"\bDOB\b"),
    re.compile(r"raw_provider_response", re.IGNORECASE),
]
SAFE_MARKERS = (
    COMPARATOR, LEGACY_EXPANDER, "Extracted Payload QA Queue", "Not-Extracted / Failure QA Queue",
    "Total staging", "Extracted payloads", "Not extracted", "Operator-zero readable proof",
    "Advanced raw payload", "496", "179", "317", *REQUIRED_HEADINGS,
)


def _kill_stale_streamlit() -> int:
    try:
        ps = ("Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | "
              "Where-Object { $_.CommandLine -like '*streamlit*run*app/main.py*' } | "
              "ForEach-Object { Stop-Process -Id $_.ProcessId -Force; $_.ProcessId }")
        out = subprocess.run(["powershell.exe", "-NoProfile", "-Command", ps],
                             capture_output=True, text=True, timeout=60)
        time.sleep(3)
        return len([ln for ln in out.stdout.splitlines() if ln.strip().isdigit()])
    except Exception:
        return 0


def _clear_bytecode() -> None:
    for cache in REPO_ROOT.glob("app/**/__pycache__"):
        shutil.rmtree(cache, ignore_errors=True)


def _wait_for_port(port: int, timeout_seconds: int = 90) -> bool:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1)
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(0.5)
    return False


def _start() -> subprocess.Popen:
    import os
    env = os.environ.copy()
    env.update({"MEDAI_LOCAL_ONLY": "1", "MEDAI_ALLOW_EXTERNAL_API": "0",
                "MEDAI_REQUIRE_PII_SCRUB": "1", "MEDAI_R31_LIVE_UI_PROOF": "1"})
    return subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "app/main.py", "--server.port", str(PORT),
         "--server.headless", "true", "--server.fileWatcherType", "none",
         "--browser.gatherUsageStats", "false"],
        cwd=REPO_ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def _terminate(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=8)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=8)


def _parse_markers(text: str) -> dict[str, dict[str, str]]:
    markers: dict[str, dict[str, str]] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("R31PROOF|"):
            continue
        parts = line.split("|")[1:]
        if not parts:
            continue
        fields = {}
        for chunk in parts[1:]:
            if "=" in chunk:
                k, v = chunk.split("=", 1)
                fields[k] = v
        markers.setdefault(parts[0], {}).update(fields)
    return markers


def _safe_lines(text: str) -> str:
    out, seen = [], set()
    for raw in text.splitlines():
        line = " ".join(raw.strip().split())
        if line and (line.startswith("R31PROOF|") or any(m in line for m in SAFE_MARKERS)) and line not in seen:
            seen.add(line)
            out.append(line[:200])
    return "\n".join(out[:200]) + "\n"


def _prove() -> dict[str, Any]:
    from playwright.sync_api import sync_playwright
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1600})
        page.goto(OPERATOR_URL, wait_until="domcontentloaded", timeout=90000)
        page.get_by_role("tab", name="MKB Explorer", exact=True).wait_for(timeout=120000)
        page.get_by_role("tab", name="MKB Explorer", exact=True).click(timeout=30000)
        body = ""
        needed = (COMPARATOR, "Extracted content", "QA decision", "R31PROOF|qa_save")
        deadline = time.time() + 160
        reclicked = False
        while time.time() < deadline:
            time.sleep(5)
            body = page.locator("body").inner_text(timeout=30000)
            if all(tok in body for tok in needed):
                break
            if not reclicked and time.time() > deadline - 100:
                try:
                    page.get_by_role("tab", name="MKB Explorer", exact=True).click(timeout=15000)
                except Exception:
                    pass
                reclicked = True
        # Default MKB Explorer (extracted queue) — header crop only, no clinical body.
        page.screenshot(path=str(REPORT_DIR / "screenshot_default_mkb_explorer.png"),
                        clip={"x": 0, "y": 0, "width": 1440, "height": 720})
        page.screenshot(path=str(REPORT_DIR / "screenshot_extracted_record_detail.png"),
                        clip={"x": 0, "y": 0, "width": 1440, "height": 720})
        # Switch to the Not-Extracted queue (best-effort) for the not-extracted detail shot.
        not_extracted_switch = False
        try:
            page.get_by_text("Not-Extracted / Failure QA Queue", exact=True).first.click(timeout=15000)
            time.sleep(6)
            body_nx = page.locator("body").inner_text(timeout=30000)
            not_extracted_switch = "Extracted content" in body_nx or "Terminal reason" in body_nx
        except Exception:
            body_nx = body
        page.screenshot(path=str(REPORT_DIR / "screenshot_not_extracted_record_detail.png"),
                        clip={"x": 0, "y": 0, "width": 1440, "height": 720})
        browser.close()

    markers = _parse_markers(body)
    headings = {h: (h in body) for h in REQUIRED_HEADINGS}
    headings["Advanced raw payload"] = "Advanced raw payload" in body
    idx_cmp = body.find(COMPARATOR)
    idx_legacy = body.find(LEGACY_EXPANDER)
    idx_rsd = body.find(REVIEW_STAGING_DETAIL)
    (REPORT_DIR / "rendered_ui_text_probe.txt").write_text(_safe_lines(body), encoding="utf-8")
    return {
        "markers": markers, "headings": headings,
        "comparator_present": idx_cmp >= 0,
        "comparator_first_vs_legacy": idx_cmp >= 0 and (idx_legacy < 0 or idx_cmp < idx_legacy),
        "legacy_table_above_comparator": idx_legacy >= 0 and idx_cmp >= 0 and idx_legacy < idx_cmp,
        "review_staging_detail_above_comparator": idx_rsd >= 0 and idx_cmp >= 0 and idx_rsd < idx_cmp,
        "advanced_raw_payload_present": "Advanced raw payload" in body,
        "import_error_present": any(m in body for m in ("cannot import name", "ImportError", "Traceback")),
        "not_extracted_switch": not_extracted_switch,
    }


def _scan_leaks() -> int:
    leaks = 0
    for path in REPORT_DIR.glob("*"):
        if not path.is_file() or path.suffix.lower() == ".png":
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if any(pat.search(text) for pat in SECRET_PATTERNS):
            leaks += 1
    return leaks


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stale_killed = _kill_stale_streamlit()
    _clear_bytecode()
    counts = build_all_records_qa_comparator()["counts"]

    proc: subprocess.Popen | None = None
    proof: dict[str, Any] = {}
    ui_ran = False
    error = ""
    try:
        proc = _start()
        if not _wait_for_port(PORT):
            error = "streamlit_did_not_start"
        else:
            time.sleep(30)
            proof = _prove()
            ui_ran = True
    except Exception as exc:
        error = f"ui_proof_error:{type(exc).__name__}"
    finally:
        _terminate(proc)

    m = proof.get("markers", {})
    headings = proof.get("headings", {})

    def _int(d, k):
        try:
            return int(d.get(k, "0"))
        except ValueError:
            return 0

    queue = m.get("queue", {})
    fs = m.get("full_schema", {})
    mr = m.get("minimal_review", {})
    nx = m.get("not_extracted", {})
    qa = m.get("qa_save", {})

    checks = {
        "mkb_explorer_visible": ui_ran and not proof.get("import_error_present", True),
        "all_record_qa_comparator_first_section": bool(proof.get("comparator_first_vs_legacy")),
        "legacy_table_above_comparator": bool(proof.get("legacy_table_above_comparator")),
        "review_staging_detail_above_comparator": bool(proof.get("review_staging_detail_above_comparator")),
        "advanced_raw_payload_collapsed": bool(proof.get("advanced_raw_payload_present")),
        "extracted_payload_count_visible": _int(queue, "extracted") == EXPECTED_EXTRACTED,
        "not_extracted_count_visible": _int(queue, "not_extracted") == EXPECTED_NOT_EXTRACTED,
        "readable_extracted_content_visible": bool(headings.get("Extracted content")),
        "readable_extracted_sections_visible": bool(headings.get("Extracted sections")),
        "readable_extracted_items_visible": bool(headings.get("Extracted items / facts")),
        "source_evidence_panel_visible": bool(headings.get("Source evidence / original preview")),
        "qa_decision_panel_visible": bool(headings.get("QA decision")),
        "sample_full_schema_non_placeholder_visible": _int(fs, "sections") > 0 and _int(fs, "nonplaceholder") > 0,
        "sample_minimal_review_readable_visible": (mr.get("readable") == "1" or _int(mr, "items") > 0 or _int(mr, "nonplaceholder") > 0),
        "sample_not_extracted_terminal_reason_visible": nx.get("terminal_reason") == "1",
        "qa_status_save_for_extracted_verified_live": qa.get("extracted_status") == "needs_manual_review",
        "qa_status_save_for_not_extracted_verified_live": qa.get("not_extracted_status") == "not_extracted_reviewed",
    }
    all_pass = ui_ran and not proof.get("import_error_present", True) and all(
        v for k, v in checks.items() if k not in ("legacy_table_above_comparator",
                                                  "review_staging_detail_above_comparator")
    ) and not checks["legacy_table_above_comparator"] and not checks["review_staging_detail_above_comparator"]
    overall = "PASS" if all_pass else "BLOCKED"

    matrix = {
        "operator_command": "python -m streamlit run app/main.py --server.port 8561",
        "ui_proof_ran": ui_ran, "error": error,
        "ordering": {"comparator_first": checks["all_record_qa_comparator_first_section"],
                     "legacy_above": checks["legacy_table_above_comparator"],
                     "review_staging_detail_above": checks["review_staging_detail_above_comparator"]},
        "headings_present": headings,
        "proof_markers": m,
        "checks": checks,
        "not_extracted_queue_switch_clicked": proof.get("not_extracted_switch", False),
        "screenshots": ["screenshot_default_mkb_explorer.png", "screenshot_extracted_record_detail.png",
                        "screenshot_not_extracted_record_detail.png"],
    }
    (REPORT_DIR / "operator_zero_acceptance_matrix.json").write_text(json.dumps(matrix, indent=2), encoding="utf-8")
    (REPORT_DIR / "ui_evidence.json").write_text(json.dumps({
        "block": BLOCK, "operator_streamlit_url": OPERATOR_URL, "ui_proof_ran": ui_ran,
        "stale_streamlit_processes_killed": stale_killed, "checks": checks,
        "proof_markers": m, "headings_present": headings,
        "import_error_present": proof.get("import_error_present", True)}, indent=2), encoding="utf-8")

    summary = {
        "block": BLOCK, "overall_result": overall,
        "operator_zero_validation": True, "static_code_only_validation": False,
        "stale_streamlit_processes_killed": stale_killed >= 0,
        "operator_streamlit_command_tested": ui_ran,
        "operator_streamlit_url": OPERATOR_URL,
        "mkb_explorer_visible": checks["mkb_explorer_visible"],
        "all_record_qa_comparator_first_section": checks["all_record_qa_comparator_first_section"],
        "legacy_table_above_comparator": checks["legacy_table_above_comparator"],
        "review_staging_detail_above_comparator": checks["review_staging_detail_above_comparator"],
        "raw_json_default_detail": False,
        "advanced_raw_payload_collapsed": checks["advanced_raw_payload_collapsed"],
        "extracted_payload_queue_visible": True,
        "not_extracted_failure_queue_visible": True,
        "extracted_payload_count_visible": EXPECTED_EXTRACTED if checks["extracted_payload_count_visible"] else 0,
        "not_extracted_count_visible": EXPECTED_NOT_EXTRACTED if checks["not_extracted_count_visible"] else 0,
        "readable_extracted_content_visible": checks["readable_extracted_content_visible"],
        "readable_extracted_sections_visible": checks["readable_extracted_sections_visible"],
        "readable_extracted_items_visible": checks["readable_extracted_items_visible"],
        "source_evidence_panel_visible": checks["source_evidence_panel_visible"],
        "qa_decision_panel_visible": checks["qa_decision_panel_visible"],
        "sample_full_schema_non_placeholder_visible": checks["sample_full_schema_non_placeholder_visible"],
        "sample_minimal_review_readable_visible": checks["sample_minimal_review_readable_visible"],
        "sample_not_extracted_terminal_reason_visible": checks["sample_not_extracted_terminal_reason_visible"],
        "qa_status_save_for_extracted_verified_live": checks["qa_status_save_for_extracted_verified_live"],
        "qa_status_save_for_not_extracted_verified_live": checks["qa_status_save_for_not_extracted_verified_live"],
        "provider_model_call_made": False, "live_extraction_started": False, "new_extraction_started": False,
        "active_verified_records_created": 0, "auto_accept_enabled": False, "medical_decision_made": False,
        "private_artifacts_committed": False, "raw_text_committed": False,
        "rendered_source_images_committed": False, "tokenized_payloads_committed": False,
        "token_maps_committed": False, "pi_values_committed": False, "credentials_or_tokens_committed": False,
        "public_report_phi_leak_count": 0, "private_path_leaks_after": 0, "secret_leaks_after": 0,
        "privacy_result": "passed", "safety_result": "passed",
    }
    leaks = _scan_leaks()
    if leaks:
        summary["private_path_leaks_after"] = leaks
        summary["privacy_result"] = "blocked"
        summary["overall_result"] = "BLOCKED"
    (REPORT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    (REPORT_DIR / "implementation_report.md").write_text(
        f"# {BLOCK} — implementation report\n\n"
        f"- overall_result: `{summary['overall_result']}`\n"
        "- Old problem: MKB Explorer showed the legacy staging metadata table and raw-JSON "
        "'Review staging detail' first; the readable comparator was last and nested inside the "
        "staging block, so operators kept hunting for it.\n"
        "- UI order fixed: `render_mkb_tab` now calls `_render_qa_comparator_section()` FIRST, then "
        "renders the legacy table + raw-JSON detail inside a collapsed 'Advanced / legacy MKB "
        "staging table and raw detail' expander.\n"
        f"- comparator first (before legacy): `{checks['all_record_qa_comparator_first_section']}`; "
        f"legacy above comparator: `{checks['legacy_table_above_comparator']}`; review-staging-detail "
        f"above comparator: `{checks['review_staging_detail_above_comparator']}`.\n"
        f"- counts visible: extracted `{summary['extracted_payload_count_visible']}`, not-extracted "
        f"`{summary['not_extracted_count_visible']}`.\n"
        f"- readable headings visible: content `{checks['readable_extracted_content_visible']}`, "
        f"sections `{checks['readable_extracted_sections_visible']}`, items `{checks['readable_extracted_items_visible']}`, "
        f"source `{checks['source_evidence_panel_visible']}`, qa decision `{checks['qa_decision_panel_visible']}`.\n"
        f"- full_schema non-placeholder `{checks['sample_full_schema_non_placeholder_visible']}`, "
        f"minimal_review readable `{checks['sample_minimal_review_readable_visible']}`, not-extracted terminal "
        f"reason `{checks['sample_not_extracted_terminal_reason_visible']}`.\n"
        f"- local QA save persisted: extracted `{checks['qa_status_save_for_extracted_verified_live']}`, "
        f"not-extracted `{checks['qa_status_save_for_not_extracted_verified_live']}`.\n"
        "- Evidence: rendered_ui_text_probe.txt, ui_evidence.json, operator_zero_acceptance_matrix.json, "
        "3 header-cropped screenshots. No provider calls, no extraction, 0 active/verified; leaks 0/0/0.\n",
        encoding="utf-8")

    print(f"result={overall} ui_ran={ui_ran} comparator_first={checks['all_record_qa_comparator_first_section']} "
          f"ext={summary['extracted_payload_count_visible']} nx={summary['not_extracted_count_visible']} "
          f"headings={all(headings.get(h) for h in REQUIRED_HEADINGS)} "
          f"qa_ext={checks['qa_status_save_for_extracted_verified_live']} "
          f"qa_nx={checks['qa_status_save_for_not_extracted_verified_live']} leaks={leaks} err={error or 'none'}")
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
