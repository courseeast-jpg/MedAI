"""Focused tests for MEDAI-OPERATOR-CONSOLE-REDESIGN-13A."""
from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

from clinical_knowledge.privacy import check_public_report_payload

REPO_ROOT = Path(__file__).resolve().parents[1]
MAIN_PATH = REPO_ROOT / "app" / "main.py"
SOURCE = MAIN_PATH.read_text(encoding="utf-8")
TREE = ast.parse(SOURCE)


def _function_source(name: str) -> str:
    for node in TREE.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(SOURCE, node) or ""
    raise AssertionError(f"{name} not found")


def test_default_tabs_are_run_review_mkb_explorer_review_queue() -> None:
    from app.main import operator_console_redesign_static_model, operator_tabs

    assert operator_tabs(False) == ["Run & Review", "MKB Explorer", "Review Queue"]
    assert operator_console_redesign_static_model()["default_tabs"] == [
        "Run & Review",
        "MKB Explorer",
        "Review Queue",
    ]


def test_safety_banner_and_pills_present() -> None:
    safety_source = _function_source("render_operator_safety_panel")

    assert "Review required - not for diagnosis." in safety_source
    for expected in ("Local only", "Cloud APIs off", "Privacy check on", "Human review"):
        assert expected in safety_source
    assert "MedAI does not diagnose" in SOURCE


def test_category_and_specialty_selectors_present() -> None:
    run_source = _function_source("render_current_run_tab")

    assert '"Document category"' in run_source
    assert "render_specialty_selector(" in run_source
    assert '"Medical specialty / domain"' in SOURCE


def test_supported_file_type_copy_present() -> None:
    assert "SUPPORTED_FILE_TYPE_COPY" in SOURCE
    assert "PDF, TXT, PNG, JPG/JPEG, TIFF/TIF, BMP, DOCX" in SOURCE


def test_documents_waiting_count_visible() -> None:
    run_source = _function_source("render_current_run_tab")
    queue_source = _function_source("render_queue_panel")

    assert 'start_col.metric("Documents waiting"' in run_source
    assert "Documents waiting" in queue_source


def test_start_run_disabled_reason_appears_when_queue_empty() -> None:
    from app.main import queue_display_state, start_run_state_reason

    state = start_run_state_reason(queue_display_state(queued_count=0, selected_count=0))

    assert state["enabled"] is False
    assert state["reason"] == "No documents queued. Add supported files to start."


def test_completed_run_empty_queue_has_restart_reason_not_stale_selected_copy() -> None:
    from app.main import queue_display_state, start_run_state_reason

    state = start_run_state_reason(
        queue_display_state(queued_count=0, selected_count=1),
        active_run={"failed": False, "results": []},
    )

    assert state["enabled"] is False
    assert state["reason"] == "Run complete. Add files to start another run."
    assert "selected files are still being added" not in state["reason"]
    assert "adding to local queue" not in state["reason"]


def test_selected_files_not_queued_show_add_action_not_adding_state() -> None:
    from app.main import queue_display_state, start_run_state_reason

    queue_state = queue_display_state(queued_count=0, selected_count=2)
    state = start_run_state_reason(queue_state)

    assert queue_state["message"] == "Files selected. Add selected files to queue."
    assert state["reason"] == "Start disabled: selected files must be added to the queue first."
    assert "adding to local queue" not in queue_state["message"]
    assert "still being added" not in state["reason"]


def test_start_run_enabled_reason_appears_when_queued_count_positive() -> None:
    from app.main import queue_display_state, start_run_state_reason

    state = start_run_state_reason(queue_display_state(queued_count=2, selected_count=0))

    assert state["enabled"] is True
    assert state["reason"] == "Ready: 2 documents waiting."


def test_empty_queue_hides_stale_failed_current_result() -> None:
    from app.main import visible_current_run

    assert visible_current_run({"failed": True}, queued_count=0, selected_count=0) is None


def test_completed_run_does_not_show_stale_adding_to_local_queue_message() -> None:
    from app.main import current_run_status_message, queue_display_state

    message = current_run_status_message(
        queue_state=queue_display_state(queued_count=0, selected_count=1),
        active_run={"failed": False, "results": []},
    )

    assert message == "Run complete. Review results below."
    assert "adding to local queue" not in message


def test_stale_queue_copy_removed_from_source() -> None:
    assert "Files selected; adding to local queue..." not in SOURCE
    assert "selected files are still being added to the local queue" not in SOURCE


def test_mkb_explorer_displays_active_and_quarantined_review_bound() -> None:
    mkb_source = _function_source("render_mkb_tab")

    assert '"Active"' in mkb_source
    assert '"Quarantined / review-bound"' in mkb_source
    assert '"Superseded / rejected"' in mkb_source
    assert 'index=3 if base_counts["review_bound"] else 0' in mkb_source


def test_review_queue_displays_accept_reject_defer() -> None:
    review_source = _function_source("render_review_queue_tab")

    assert "Accept after source comparison" in SOURCE
    assert "reject_cfg" in review_source
    assert "defer_cfg" in review_source
    assert "human review action" in review_source


def test_no_auto_accept_copy_or_behavior_introduced() -> None:
    from app.main import operator_console_redesign_static_model

    assert operator_console_redesign_static_model()["auto_accept"] is False
    assert "automatic acceptance" in SOURCE


def test_external_api_used_remains_false_in_report() -> None:
    proc = subprocess.run(
        [sys.executable, "scripts/run_medai_operator_console_redesign_13a.py"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    payload = json.loads(
        (REPO_ROOT / "reports" / "medai_operator_console_redesign_13a_r1" / "medai_operator_console_redesign_13a_r1_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["external_api_used"] is False
    assert payload["auto_accept"] is False


def test_privacy_report_safe() -> None:
    subprocess.run(
        [sys.executable, "scripts/run_medai_operator_console_redesign_13a.py"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    report_dir = REPO_ROOT / "reports" / "medai_operator_console_redesign_13a_r1"
    for path in report_dir.iterdir():
        payload = json.loads(path.read_text(encoding="utf-8")) if path.suffix == ".json" else path.read_text(encoding="utf-8")
        result = check_public_report_payload(payload)
        assert result.passed, f"{path.name}: {result.leak_examples_redacted}"
