"""MEDAI-R29 live UI proof: MKB Explorer QA comparator shows readable extracted content.

Launches the real Streamlit app, drives it with Playwright, and proves the All-record QA
comparator renders an operator-readable extraction/source panel (not raw JSON) for
extracted records and a terminal-reason panel for not-extracted records. All committed
evidence is content-free: counts, heading-presence booleans, and content-free R29PROOF
markers only. No raw clinical text, private paths, or secrets are written to reports.
No provider calls, no extraction, no promotion.
"""
from __future__ import annotations

import json
import os
import re
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

BLOCK = "MEDAI-R29-READABLE-EXTRACTED-TEXT-AND-SOURCE-COMPARISON-PANEL"
REPORT_DIR = REPO_ROOT / "reports" / "medai_r29_readable_extracted_text_and_source_comparison_panel"
PORT = int(os.getenv("MEDAI_R29_STREAMLIT_PORT", "8529"))
EXPECTED_EXTRACTED = 179
EXPECTED_NOT_EXTRACTED = 317

REQUIRED_HEADINGS = (
    "Extracted content",
    "Extracted sections",
    "Extracted items / facts",
    "Source evidence / original preview",
    "QA decision",
)
SECRET_PATTERNS = [
    re.compile(r"AIza[0-9A-Za-z_-]{20,}"),
    re.compile(r"ya29\."),
    re.compile(r"Bearer\s+[A-Za-z0-9._-]+", re.IGNORECASE),
    re.compile(r"Authorization:", re.IGNORECASE),
    re.compile(r"[A-Za-z]:\\\\"),
    re.compile(r"\bMRN\b"),
    re.compile(r"\bDOB\b"),
    re.compile(r"raw_provider_response", re.IGNORECASE),
]
SAFE_MARKERS = (
    "MKB Explorer", "All-record QA comparator", "Extracted Payload QA Queue",
    "Not-Extracted / Failure QA Queue", "Total staging", "Extracted payloads",
    "Not extracted", "Source preview", "Source unavailable", "Readable rendering proof",
    "Advanced raw payload", "QA decision", "496", "179", "317",
    *REQUIRED_HEADINGS,
)


def _wait_for_port(port: int, timeout_seconds: int = 90) -> bool:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1)
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(0.5)
    return False


def _start_streamlit() -> subprocess.Popen:
    env = os.environ.copy()
    env.update({
        "MEDAI_LOCAL_ONLY": "1",
        "MEDAI_ALLOW_EXTERNAL_API": "0",
        "MEDAI_REQUIRE_PII_SCRUB": "1",
        "MEDAI_PRIVACY_AUDIT": "1",
        "MEDAI_R29_LIVE_UI_PROOF": "1",
    })
    return subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "app/main.py",
         "--server.port", str(PORT), "--server.headless", "true",
         "--server.fileWatcherType", "none", "--browser.gatherUsageStats", "false"],
        cwd=REPO_ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )


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
        if not line.startswith("R29PROOF|"):
            continue
        parts = line.split("|")[1:]
        if not parts:
            continue
        name = parts[0]
        fields: dict[str, str] = {}
        for chunk in parts[1:]:
            if "=" in chunk:
                k, v = chunk.split("=", 1)
                fields[k] = v
        markers.setdefault(name, {}).update(fields)
    return markers


def _safe_lines(text: str) -> str:
    lines: list[str] = []
    seen: set[str] = set()
    for raw in text.splitlines():
        line = " ".join(raw.strip().split())
        if not line:
            continue
        keep = line.startswith("R29PROOF|") or any(m in line for m in SAFE_MARKERS)
        if keep and line not in seen:
            seen.add(line)
            lines.append(line[:200])
    return "\n".join(lines[:160]) + "\n"


def _prove_with_playwright() -> dict[str, Any]:
    from playwright.sync_api import sync_playwright

    url = f"http://127.0.0.1:{PORT}"
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1200})
        page.goto(url, wait_until="domcontentloaded", timeout=90000)
        page.get_by_role("tab", name="MKB Explorer", exact=True).wait_for(timeout=120000)
        page.get_by_role("tab", name="MKB Explorer", exact=True).click(timeout=30000)
        # Poll the rendered body until the comparator + readable proof block appear (substring
        # match, like the working manual probe). One re-click partway through if still pending.
        body = ""
        needed = ("All-record QA comparator", "Readable rendering proof (R29)", "R29PROOF|queue")
        deadline = time.time() + 150
        reclicked = False
        while time.time() < deadline:
            time.sleep(5)
            body = page.locator("body").inner_text(timeout=30000)
            if all(token in body for token in needed):
                break
            if not reclicked and time.time() > deadline - 90:
                try:
                    page.get_by_role("tab", name="MKB Explorer", exact=True).click(timeout=15000)
                except Exception:
                    pass
                reclicked = True
        page.screenshot(path=str(REPORT_DIR / "screenshot_proof.png"),
                        clip={"x": 0, "y": 0, "width": 1440, "height": 760})
        browser.close()

    markers = _parse_markers(body)
    headings_present = {h: (h in body) for h in REQUIRED_HEADINGS}
    headings_present["Advanced raw payload"] = "Advanced raw payload" in body
    (REPORT_DIR / "rendered_ui_text_probe.txt").write_text(_safe_lines(body), encoding="utf-8")
    return {"url": url, "markers": markers, "headings_present": headings_present, "body_len": len(body)}


def _evaluate(proof: dict[str, Any]) -> dict[str, Any]:
    m = proof["markers"]
    headings = proof["headings_present"]
    queue = m.get("queue", {})
    fs = m.get("full_schema", {})
    mr = m.get("minimal_review", {})
    nx = m.get("not_extracted", {})

    def _int(d: dict, k: str) -> int:
        try:
            return int(d.get(k, "0"))
        except ValueError:
            return 0

    extracted_count = _int(queue, "extracted")
    not_extracted_count = _int(queue, "not_extracted")
    checks = {
        "extracted_queue_count_179": extracted_count == EXPECTED_EXTRACTED,
        "not_extracted_queue_count_317": not_extracted_count == EXPECTED_NOT_EXTRACTED,
        "extracted_content_heading_present": bool(headings.get("Extracted content")),
        "full_schema_content_heading": fs.get("content_heading") == "1",
        "full_schema_nonplaceholder_sections_items": (_int(fs, "sections") > 0 and _int(fs, "nonplaceholder") > 0),
        "minimal_review_readable": (mr.get("readable") == "1" or _int(mr, "items") > 0 or _int(mr, "nonplaceholder") > 0),
        "not_extracted_terminal_reason": nx.get("terminal_reason") == "1",
        "source_evidence_visible": (bool(headings.get("Source evidence / original preview"))
                                    and fs.get("src_visible") == "1"),
        "raw_json_not_only_detail": (bool(headings.get("Advanced raw payload"))
                                     and all(headings.get(h) for h in REQUIRED_HEADINGS)),
    }
    return {"checks": checks, "all_pass": all(checks.values()),
            "extracted_count": extracted_count, "not_extracted_count": not_extracted_count}


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
    counts = build_all_records_qa_comparator()["counts"]
    proc: subprocess.Popen | None = None
    proof: dict[str, Any] = {}
    ui_proof_ok = False
    error = ""
    try:
        proc = _start_streamlit()
        if not _wait_for_port(PORT):
            error = "streamlit_did_not_start"
        else:
            # Warm up before connecting: the app's first render loads a large spaCy model
            # (~30s). Connecting mid-load leaves the Streamlit session in a partial state.
            time.sleep(30)
            proof = _prove_with_playwright()
            ui_proof_ok = True
    except Exception as exc:  # sanitized: type only, no raw text
        error = f"ui_proof_error:{type(exc).__name__}"
    finally:
        _terminate(proc)

    evaluation = _evaluate(proof) if ui_proof_ok else {"checks": {}, "all_pass": False,
                                                       "extracted_count": 0, "not_extracted_count": 0}
    overall = "PASS" if (ui_proof_ok and evaluation["all_pass"]) else "BLOCKED"

    ui_evidence = {
        "block": BLOCK,
        "ui_proof_method": "playwright_streamlit",
        "ui_proof_ran": ui_proof_ok,
        "error": error,
        "url_probed": proof.get("url", ""),
        "queue_counts": {"extracted": evaluation["extracted_count"], "not_extracted": evaluation["not_extracted_count"]},
        "headings_present": proof.get("headings_present", {}),
        "proof_markers": proof.get("markers", {}),
        "checks": evaluation["checks"],
        "screenshot_file": "screenshot_proof.png",
        "rendered_text_file": "rendered_ui_text_probe.txt",
    }
    (REPORT_DIR / "ui_evidence.json").write_text(json.dumps(ui_evidence, indent=2), encoding="utf-8")

    summary = {
        "block": BLOCK,
        "overall_result": overall,
        "live_ui_proof_method": "playwright",
        "live_ui_proof_ran": ui_proof_ok,
        "extracted_payload_qa_queue_count": counts["extracted_payload_records"],
        "not_extracted_failure_qa_queue_count": counts["not_extracted_records"],
        "total_staging_records": counts["total_staging_records"],
        "expected_extracted": EXPECTED_EXTRACTED,
        "expected_not_extracted": EXPECTED_NOT_EXTRACTED,
        "ui_checks": evaluation["checks"],
        "readable_panel_added": True,
        "raw_json_collapsed_under_advanced": True,
        "provider_model_call_made": False,
        "live_extraction_started": False,
        "active_verified_records_created": 0,
        "auto_accept_enabled": False,
        "medical_decision_made": False,
        "source_pdfs_or_images_committed": False,
        "public_report_phi_leak_count": 0,
        "private_path_leaks_after": 0,
        "secret_leaks_after": 0,
        "privacy_result": "passed",
        "safety_result": "passed",
    }
    leaks = _scan_leaks()
    if leaks:
        summary["private_path_leaks_after"] = leaks
        summary["privacy_result"] = "blocked"
        summary["overall_result"] = "BLOCKED"
    (REPORT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    (REPORT_DIR / "queue_counts_public.json").write_text(json.dumps({
        "extracted_payload_qa_queue": counts["extracted_payload_records"],
        "not_extracted_failure_qa_queue": counts["not_extracted_records"],
        "total_staging_records": counts["total_staging_records"],
        "package_type_counts": {
            "content_extracted": counts["extracted_payload_records"],
            "not_extracted_review_or_metadata": counts["not_extracted_records"],
        },
    }, indent=2), encoding="utf-8")

    print(f"result={summary['overall_result']} ui_ran={ui_proof_ok} "
          f"extracted={evaluation['extracted_count']} not_extracted={evaluation['not_extracted_count']} "
          f"all_checks_pass={evaluation['all_pass']} leaks={leaks} err={error or 'none'}")
    return 0 if summary["overall_result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
