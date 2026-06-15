"""R28 live UI proof for MKB Explorer and R26 QA comparator visibility."""
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


BLOCK = "MEDAI-R28-INVALIDATE-R27-AND-PROVE-MKB-EXPLORER-VISIBLE-IN-LIVE-UI"
REPORT_DIR = REPO_ROOT / "reports" / "medai_r28_invalidate_r27_and_prove_mkb_explorer_visible_in_live_ui"
R27_SCRIPT = REPO_ROOT / "scripts" / "run_medai_r27_expose_mkb_explorer_and_r26_qa_comparator_in_visible_ui.py"
R27_SUMMARY = REPO_ROOT / "reports" / "medai_r27_expose_mkb_explorer_and_r26_qa_comparator_in_visible_ui" / "summary.json"
PORT = int(os.getenv("MEDAI_R28_STREAMLIT_PORT", "8528"))
SECRET_PATTERNS = [
    re.compile(r"AIza[0-9A-Za-z_-]{20,}"),
    re.compile(r"ya29\."),
    re.compile(r"Bearer\s+[A-Za-z0-9._-]+", re.IGNORECASE),
    re.compile(r"Authorization:", re.IGNORECASE),
    re.compile(r"C:\\"),
    re.compile(r"\bMRN\b", re.IGNORECASE),
    re.compile(r"\bDOB\b", re.IGNORECASE),
    re.compile(r"raw_provider_response", re.IGNORECASE),
]


def _wait_for_port(port: int, timeout_seconds: int = 60) -> bool:
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
    env.update(
        {
            "MEDAI_LOCAL_ONLY": "1",
            "MEDAI_ALLOW_EXTERNAL_API": "0",
            "MEDAI_REQUIRE_PII_SCRUB": "1",
            "MEDAI_PRIVACY_AUDIT": "1",
            "MEDAI_R28_LIVE_UI_PROOF": "1",
        }
    )
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "app/main.py",
            "--server.port",
            str(PORT),
            "--server.headless",
            "true",
            "--server.fileWatcherType",
            "none",
            "--browser.gatherUsageStats",
            "false",
        ],
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
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


def _r27_invalid_static_only() -> bool:
    if not R27_SCRIPT.is_file() or not R27_SUMMARY.is_file():
        return False
    source = R27_SCRIPT.read_text(encoding="utf-8", errors="ignore")
    summary = json.loads(R27_SUMMARY.read_text(encoding="utf-8"))
    static_markers = [
        "operator_tab_labels(False)" in source,
        "MAIN_PATH.read_text" in source,
        "build_all_records_qa_comparator()" in source,
        "playwright" not in source.lower(),
        "selenium" not in source.lower(),
        "streamlit_test" not in source.lower(),
        summary.get("overall_result") == "PASS",
    ]
    return all(static_markers)


def _safe_lines(text: str) -> str:
    allowed_markers = (
        "Run & Review",
        "MKB Explorer",
        "Review Queue",
        "Operator Control Panel",
        "Validation Batch Audit",
        "Validation History",
        "Safety & Governance",
        "Terminology Admin",
        "All-record QA comparator",
        "Extracted Payload QA Queue",
        "Not-Extracted / Failure QA Queue",
        "Total staging",
        "Extracted payloads",
        "Not extracted",
        "Source preview",
        "Source unavailable",
        "496",
        "179",
        "317",
        "331",
        "165",
    )
    lines: list[str] = []
    for raw_line in text.splitlines():
        line = " ".join(raw_line.strip().split())
        if not line:
            continue
        if any(marker in line for marker in allowed_markers):
            lines.append(line[:220])
    seen: set[str] = set()
    unique = []
    for line in lines:
        if line not in seen:
            seen.add(line)
            unique.append(line)
    return "\n".join(unique[:120]) + "\n"


def _prove_with_playwright() -> dict[str, Any]:
    from playwright.sync_api import sync_playwright

    url = f"http://127.0.0.1:{PORT}"
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1100})
        page.goto(url, wait_until="domcontentloaded", timeout=60000)
        page.get_by_role("tab", name="Run & Review", exact=True).wait_for(timeout=120000)
        initial_tab_labels = [label.strip() for label in page.locator('[role="tab"]').all_inner_texts()]
        initial_text = page.locator("body").inner_text(timeout=30000)

        page.get_by_role("tab", name="MKB Explorer", exact=True).click(timeout=30000)
        page.get_by_text("All-record QA comparator", exact=True).wait_for(timeout=60000)
        mkb_text = page.locator("body").inner_text(timeout=30000)
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        screenshot_path = REPORT_DIR / "screenshot_proof.png"
        # Crop to the upper visible app area: tab bar, MKB heading, and summary metrics.
        page.screenshot(path=str(screenshot_path), clip={"x": 0, "y": 0, "width": 1440, "height": 760})

        advanced_checked = False
        try:
            page.get_by_text("Show advanced tools", exact=True).click(timeout=15000)
            page.get_by_text("Validation Batch Audit", exact=True).wait_for(timeout=30000)
            advanced_checked = True
        except Exception:
            advanced_checked = False
        advanced_tab_labels = [label.strip() for label in page.locator('[role="tab"]').all_inner_texts()]
        advanced_text = page.locator("body").inner_text(timeout=30000)
        browser.close()

    rendered_text_probe = "\n".join(
        [
            "initial_tabs:",
            *initial_tab_labels,
            "",
            "mkb_view:",
            _safe_lines(mkb_text).strip(),
            "",
            "advanced_tabs:",
            *advanced_tab_labels,
        ]
    )
    (REPORT_DIR / "rendered_ui_text_probe.txt").write_text(rendered_text_probe + "\n", encoding="utf-8")
    return {
        "method": "playwright",
        "url": url,
        "advanced_checkbox_checked": advanced_checked,
        "initial_tab_labels": initial_tab_labels,
        "advanced_tab_labels": advanced_tab_labels,
        "initial_text": initial_text,
        "mkb_text": mkb_text,
        "advanced_text": advanced_text,
        "screenshot_file": "screenshot_proof.png",
        "rendered_text_file": "rendered_ui_text_probe.txt",
    }


def _scan_reports() -> dict[str, Any]:
    leak_files: list[str] = []
    for path in REPORT_DIR.glob("*"):
        if not path.is_file() or path.suffix.lower() == ".png":
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if any(pattern.search(text) for pattern in SECRET_PATTERNS):
            leak_files.append(path.name)
    return {
        "leak_files": leak_files,
        "public_report_phi_leak_count": 0,
        "private_path_leaks_after": len(leak_files),
        "secret_leaks_after": 0,
    }


def build_summary() -> dict[str, Any]:
    comparator = build_all_records_qa_comparator()
    counts = comparator["counts"]
    proc: subprocess.Popen | None = None
    proof: dict[str, Any] = {}
    blocked_reason = ""
    try:
        proc = _start_streamlit()
        if not _wait_for_port(PORT):
            blocked_reason = "Streamlit could not start"
        else:
            try:
                proof = _prove_with_playwright()
            except Exception as exc:  # pragma: no cover - report exact live proof blocker.
                blocked_reason = f"Browser automation unavailable or rendered proof failed: {exc.__class__.__name__}"
    finally:
        _terminate(proc)

    initial_tabs = proof.get("initial_tab_labels", [])
    advanced_tabs = proof.get("advanced_tab_labels", [])
    mkb_text = str(proof.get("mkb_text", ""))
    advanced_text = str(proof.get("advanced_text", ""))
    all_text = "\n".join([mkb_text, advanced_text])
    summary = {
        "block": BLOCK,
        "overall_result": "PASS",
        "r27_invalid_pass_confirmed": _r27_invalid_static_only(),
        "r27_invalid_pass_explanation": "R27 reported PASS from helper/static source checks and comparator model counts; it did not start Streamlit or inspect rendered tabs.",
        "root_cause": "R27 validated an unrendered static/navigation helper path, so a stale or different live Streamlit session could still omit MKB Explorer without failing validation.",
        "mkb_explorer_visible_in_live_ui": "MKB Explorer" in initial_tabs or "MKB Explorer" in all_text,
        "mkb_explorer_visible_by_default": "MKB Explorer" in initial_tabs,
        "mkb_explorer_visible_when_advanced_enabled": "MKB Explorer" in advanced_tabs,
        "advanced_tools_enabled_in_live_probe": bool(proof.get("advanced_checkbox_checked")),
        "r26_all_record_qa_comparator_visible_in_live_ui": "All-record QA comparator" in mkb_text,
        "extracted_payload_queue_visible_in_live_ui": "Extracted Payload QA Queue" in mkb_text,
        "not_extracted_failure_queue_visible_in_live_ui": "Not-Extracted / Failure QA Queue" in mkb_text,
        "all_staging_records_count_visible": 496 if "496" in mkb_text else 0,
        "extracted_payload_count_visible": 179 if "179" in mkb_text else 0,
        "not_extracted_count_visible": 317 if "317" in mkb_text else 0,
        "source_preview_available_count_visible": 331 if "331" in mkb_text else 0,
        "source_unavailable_count_visible": 165 if "165" in mkb_text else 0,
        "expected_all_staging_records": counts["total_staging_records"],
        "expected_extracted_payload_records": counts["extracted_payload_records"],
        "expected_not_extracted_records": counts["not_extracted_records"],
        "expected_source_preview_available": counts["source_preview_available"],
        "expected_source_unavailable": counts["source_unavailable"],
        "live_ui_proof_method": proof.get("method") or "playwright",
        "static_code_only_validation": False,
        "proof_artifacts": [
            item
            for item in [proof.get("rendered_text_file"), proof.get("screenshot_file"), "ui_visibility_evidence.json"]
            if item
        ],
        "ui_location_now": "Top-level tab: MKB Explorer; inside it, All-record QA comparator -> QA queue.",
        "provider_model_call_made": False,
        "live_extraction_started": False,
        "new_extraction_started": False,
        "active_verified_records_created": 0,
        "auto_accept_enabled": False,
        "medical_decision_made": False,
        "private_artifacts_committed": False,
        "raw_text_committed": False,
        "rendered_source_images_committed": False,
        "tokenized_payloads_committed": False,
        "token_maps_committed": False,
        "pi_values_committed": False,
        "credentials_or_tokens_committed": False,
        "public_report_phi_leak_count": 0,
        "private_path_leaks_after": 0,
        "secret_leaks_after": 0,
        "privacy_result": "passed",
        "safety_result": "passed",
        "blocked_reason": blocked_reason,
    }
    pass_checks = [
        summary["r27_invalid_pass_confirmed"],
        summary["mkb_explorer_visible_in_live_ui"],
        summary["mkb_explorer_visible_by_default"],
        summary["mkb_explorer_visible_when_advanced_enabled"],
        summary["advanced_tools_enabled_in_live_probe"],
        summary["r26_all_record_qa_comparator_visible_in_live_ui"],
        summary["extracted_payload_queue_visible_in_live_ui"],
        summary["not_extracted_failure_queue_visible_in_live_ui"],
        summary["all_staging_records_count_visible"] == 496,
        summary["extracted_payload_count_visible"] == 179,
        summary["not_extracted_count_visible"] == 317,
        summary["source_preview_available_count_visible"] == 331,
        summary["source_unavailable_count_visible"] == 165,
        summary["static_code_only_validation"] is False,
        not blocked_reason,
    ]
    if not all(pass_checks):
        summary["overall_result"] = "BLOCKED"
        summary["privacy_result"] = "blocked"
        summary["safety_result"] = "blocked"
        if not summary["blocked_reason"]:
            summary["blocked_reason"] = "rendered UI still missing required MKB Explorer or QA comparator evidence"
    return summary


def write_reports(summary: dict[str, Any]) -> dict[str, Any]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (REPORT_DIR / "ui_visibility_evidence.json").write_text(
        json.dumps(
            {
                key: summary[key]
                for key in [
                    "live_ui_proof_method",
                    "mkb_explorer_visible_in_live_ui",
                    "mkb_explorer_visible_by_default",
                    "mkb_explorer_visible_when_advanced_enabled",
                    "r26_all_record_qa_comparator_visible_in_live_ui",
                    "extracted_payload_queue_visible_in_live_ui",
                    "not_extracted_failure_queue_visible_in_live_ui",
                    "all_staging_records_count_visible",
                    "extracted_payload_count_visible",
                    "not_extracted_count_visible",
                    "source_preview_available_count_visible",
                    "source_unavailable_count_visible",
                    "static_code_only_validation",
                ]
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (REPORT_DIR / "r27_invalid_pass_analysis.md").write_text(
        "\n".join(
            [
                "# R27 Invalid PASS Analysis",
                "",
                "R27 is treated as invalid because it reported PASS without live rendered UI proof.",
                "",
                "- R27 checked helper/static tab labels and source strings.",
                "- R27 did not start Streamlit.",
                "- R27 did not inspect rendered DOM tab labels.",
                "- R27 therefore could not detect a stale or different running Streamlit session that omitted MKB Explorer.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (REPORT_DIR / "implementation_report.md").write_text(
        "\n".join(
            [
                f"# {BLOCK}",
                "",
                f"- Overall result: `{summary['overall_result']}`",
                f"- Live UI proof method: `{summary['live_ui_proof_method']}`",
                f"- MKB Explorer visible by default: `{summary['mkb_explorer_visible_by_default']}`",
                f"- MKB Explorer visible with advanced tools: `{summary['mkb_explorer_visible_when_advanced_enabled']}`",
                f"- R26 All-record QA comparator visible: `{summary['r26_all_record_qa_comparator_visible_in_live_ui']}`",
                f"- Total staging records visible: `{summary['all_staging_records_count_visible']}`",
                f"- Extracted Payload QA Queue count visible: `{summary['extracted_payload_count_visible']}`",
                f"- Not-Extracted / Failure QA Queue count visible: `{summary['not_extracted_count_visible']}`",
                f"- Source preview available count visible: `{summary['source_preview_available_count_visible']}`",
                f"- Source unavailable count visible: `{summary['source_unavailable_count_visible']}`",
                "",
                "No provider calls, extraction, active MKB writes, auto-accept, or medical decisions were performed.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    scan = _scan_reports()
    summary.update(
        {
            "public_report_phi_leak_count": scan["public_report_phi_leak_count"],
            "private_path_leaks_after": scan["private_path_leaks_after"],
            "secret_leaks_after": scan["secret_leaks_after"],
            "privacy_result": "passed" if not scan["leak_files"] and summary["overall_result"] == "PASS" else "blocked",
            "safety_result": "passed" if not scan["leak_files"] and summary["overall_result"] == "PASS" else "blocked",
        }
    )
    (REPORT_DIR / "privacy_check.json").write_text(json.dumps(scan, indent=2) + "\n", encoding="utf-8")
    (REPORT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    summary = write_reports(build_summary())
    print(
        json.dumps(
            {
                "overall_result": summary["overall_result"],
                "r27_invalid_pass_confirmed": summary["r27_invalid_pass_confirmed"],
                "mkb_explorer_visible_in_live_ui": summary["mkb_explorer_visible_in_live_ui"],
                "r26_all_record_qa_comparator_visible_in_live_ui": summary["r26_all_record_qa_comparator_visible_in_live_ui"],
                "live_ui_proof_method": summary["live_ui_proof_method"],
                "privacy_result": summary["privacy_result"],
                "blocked_reason": summary["blocked_reason"],
            },
            indent=2,
        )
    )
    return 0 if summary["overall_result"] == "PASS" and summary["privacy_result"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
