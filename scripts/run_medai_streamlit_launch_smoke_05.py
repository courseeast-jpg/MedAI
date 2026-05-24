#!/usr/bin/env python3
"""MEDAI-LOCAL-RUNTIME-DEPENDENCY-AND-ONE-DOC-VALIDATION-05.

Streamlit launch smoke. Three layers:

1. Streamlit-free helper import smoke — must always pass.
2. ``app.main`` import smoke — passes when Streamlit is installed.
3. Optional brief Streamlit launch — runs only when ``streamlit`` is
   importable AND the environment allows binding a local port. The
   script binds Streamlit to ``127.0.0.1`` and a free ephemeral port,
   waits a few seconds for the runtime to come up, then terminates
   cleanly. The actual page contents are never fetched and never
   logged.

No external API call. No raw text emitted. No PHI emitted.
"""
from __future__ import annotations

import importlib
import importlib.util
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ["MEDAI_ALLOW_EXTERNAL_API"] = "false"
os.environ.setdefault("MEDAI_LOCAL_ONLY", "true")

REPORT_DIR = REPO_ROOT / "reports" / "medai_streamlit_launch_smoke_05"
JSON_REPORT_PATH = REPORT_DIR / "medai_streamlit_launch_smoke_05_report.json"
MD_REPORT_PATH = REPORT_DIR / "medai_streamlit_launch_smoke_05_report.md"
SHORT_MD_PATH = REPORT_DIR / "MEDAI_STREAMLIT_LAUNCH_SMOKE_05.md"

UI_IMPORT_SMOKE_TARGETS = (
    "app.extracted_information_preview",
    "app.operator_review_actions",
    "execution.extracted_medical_facts",
)

#: Set to "1" or "true" to enable the optional brief launch test. Off by
#: default to avoid binding ports on environments where that is not
#: allowed.
LAUNCH_ENV_FLAG = "MEDAI_STREAMLIT_LAUNCH_SMOKE"


def _streamlit_present() -> bool:
    return importlib.util.find_spec("streamlit") is not None


def _import_smoke_for_module(module_name: str) -> Dict[str, Any]:
    try:
        importlib.import_module(module_name)
        return {"module": module_name, "passed": True, "error": None}
    except Exception as exc:
        return {"module": module_name, "passed": False, "error": f"{type(exc).__name__}"}


def _free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _attempt_streamlit_brief_launch(timeout_seconds: float = 6.0) -> Dict[str, Any]:
    flag = str(os.environ.get(LAUNCH_ENV_FLAG, "")).lower()
    if flag not in {"1", "true", "yes", "on"}:
        return {
            "ran": False,
            "reason": f"{LAUNCH_ENV_FLAG} not set; brief launch skipped by default",
            "passed": None,
        }
    if not _streamlit_present():
        return {
            "ran": False,
            "reason": "streamlit_module_not_present_in_local_environment",
            "passed": None,
        }
    port = _free_local_port()
    args = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(REPO_ROOT / "app" / "main.py"),
        "--server.address",
        "127.0.0.1",
        "--server.port",
        str(port),
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
    ]
    proc = subprocess.Popen(
        args,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={
            **os.environ,
            "MEDAI_ALLOW_EXTERNAL_API": "false",
            "MEDAI_LOCAL_ONLY": "true",
        },
    )
    bound = False
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(0.3)
                s.connect(("127.0.0.1", port))
                bound = True
                break
        except OSError:
            if proc.poll() is not None:
                # Streamlit process died early; bail out of the wait loop.
                break
            time.sleep(0.2)
    try:
        proc.terminate()
        try:
            proc.wait(timeout=4.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=2.0)
    except Exception:
        pass
    return {
        "ran": True,
        "passed": bool(bound),
        "port_attempted": port,
        "exit_code_when_terminated": proc.returncode,
    }


def _privacy_check(payload: Any) -> Dict[str, Any]:
    try:
        from clinical_knowledge.privacy import check_public_report_payload

        result = check_public_report_payload(payload)
        return {
            "passed": bool(result.passed),
            "leak_examples_redacted": list(result.leak_examples_redacted or []),
        }
    except Exception as exc:
        return {"passed": False, "error": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    streamlit_present = _streamlit_present()
    helper_import_results: List[Dict[str, Any]] = [
        _import_smoke_for_module(name) for name in UI_IMPORT_SMOKE_TARGETS
    ]
    helper_import_passed = all(item["passed"] for item in helper_import_results)

    app_main_import = {"ran": False, "passed": None, "reason": ""}
    if streamlit_present:
        try:
            proc = subprocess.run(
                [sys.executable, "-c", "import app.main; print('app_main_import_ok')"],
                cwd=REPO_ROOT,
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
                env={
                    **os.environ,
                    "MEDAI_ALLOW_EXTERNAL_API": "false",
                    "MEDAI_LOCAL_ONLY": "true",
                },
            )
            ok = proc.returncode == 0 and "app_main_import_ok" in (proc.stdout or "")
            app_main_import = {
                "ran": True,
                "passed": bool(ok),
                "exit_code": proc.returncode,
            }
        except Exception as exc:
            app_main_import = {
                "ran": True,
                "passed": False,
                "error": f"{type(exc).__name__}",
            }
    else:
        app_main_import = {
            "ran": False,
            "passed": None,
            "reason": "streamlit_module_not_present_in_local_environment",
        }

    brief_launch = _attempt_streamlit_brief_launch()

    findings: Dict[str, Any] = {
        "streamlit_present": streamlit_present,
        "helper_import_results": helper_import_results,
        "helper_import_passed": helper_import_passed,
        "app_main_import_smoke": app_main_import,
        "streamlit_brief_launch": brief_launch,
        "external_api_used": False,
        "auto_accept_enabled": False,
    }

    overall_passed = helper_import_passed and (
        # When Streamlit is missing, the helper import smoke alone is the
        # acceptance criterion (with a clear environmental note).
        (not streamlit_present)
        or bool(app_main_import.get("passed"))
    )

    report = {
        "block_id": "MEDAI-LOCAL-RUNTIME-DEPENDENCY-AND-ONE-DOC-VALIDATION-05-STREAMLIT-LAUNCH-SMOKE",
        "mode": "streamlit_launch_smoke",
        "branch_expected": "clinical-knowledge-architecture",
        "audit_findings": findings,
        "external_api_used": False,
        "auto_accept_enabled": False,
        "real_pdf_committed": False,
        "raw_text_printed": False,
        "raw_ocr_text_printed": False,
        "raw_filenames_printed": False,
        "private_paths_printed": False,
        "phi_printed": False,
        "passed": bool(overall_passed),
    }

    privacy = _privacy_check(report)
    report["privacy_check_passed"] = bool(privacy.get("passed"))
    report["privacy_check_leak_examples_redacted"] = privacy.get(
        "leak_examples_redacted", []
    )

    JSON_REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    md_lines = [
        "# MEDAI-STREAMLIT-LAUNCH-SMOKE-05 — Report",
        "",
        f"Overall: **{'passed' if overall_passed else 'not_passed'}**",
        "",
        "## Helper import smoke",
        "",
    ]
    for item in helper_import_results:
        md_lines.append(
            f"- `{item['module']}`: passed={item['passed']} error={item['error']}"
        )
    md_lines.extend([
        "",
        "## app.main import smoke",
        "",
        f"- streamlit present: {streamlit_present}",
        f"- ran: {app_main_import.get('ran')}",
        f"- passed: {app_main_import.get('passed')}",
        f"- reason: {app_main_import.get('reason', '')}",
        "",
        "## Streamlit brief launch (optional)",
        "",
        f"- ran: {brief_launch.get('ran')}",
        f"- passed: {brief_launch.get('passed')}",
        f"- reason: {brief_launch.get('reason', '')}",
        "",
        "## Safety",
        "",
        f"- external API used: {report['external_api_used']}",
        f"- auto-accept enabled: {report['auto_accept_enabled']}",
        f"- privacy check passed: {report['privacy_check_passed']}",
        "",
    ])
    MD_REPORT_PATH.write_text("\n".join(md_lines), encoding="utf-8")

    SHORT_MD_PATH.write_text(
        "# MEDAI-STREAMLIT-LAUNCH-SMOKE-05 — Short Summary\n\n"
        f"Overall: **{'passed' if overall_passed else 'not_passed'}**\n\n"
        f"- streamlit present: {streamlit_present}\n"
        f"- helper import smoke passed: {helper_import_passed}\n"
        f"- app.main import smoke passed: {app_main_import.get('passed')}\n"
        f"- streamlit brief launch ran: {brief_launch.get('ran')}\n"
        f"- streamlit brief launch passed: {brief_launch.get('passed')}\n"
        f"- external API used: {report['external_api_used']}\n"
        f"- auto-accept enabled: {report['auto_accept_enabled']}\n"
        f"- privacy check passed: {report['privacy_check_passed']}\n",
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if overall_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
