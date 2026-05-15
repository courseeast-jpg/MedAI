from __future__ import annotations

from pathlib import Path

from app import operator_control_panel as panel


PANEL_SOURCE = Path(panel.__file__)


def panel_source() -> str:
    return PANEL_SOURCE.read_text(encoding="utf-8")


def test_operator_control_panel_operator_labels_are_present() -> None:
    source = panel_source()

    for text in (
        "Operator Control Panel",
        "Run safe local checks and maintenance actions.",
        "Only approved local checks are available.",
        "Main checks",
        "Quick health check",
        "Fast check that MedAI is ready.",
        "Final MVP validation",
        "Confirms the core release still passes.",
        "Git safety check",
        "Confirms no unsafe files are staged.",
        "Advanced: terminology checks",
        "Terminology preflight",
        "Terminology inventory",
        "B07-TERM validation",
        "Advanced: routing and extraction checks",
        "ROUTE-FIX validation",
        "Focused routing tests",
        "Advanced: full test suite",
        "Full test suite",
        "Advanced: reports and recovery",
        "Last validation reports",
        "Release tags",
        "Verify release bundle",
        "Not run yet",
    ):
        assert text in source


def test_old_primary_operator_panel_labels_are_not_in_panel_source() -> None:
    source = panel_source()

    for text in (
        "MedAI Operator Control Panel",
        "Fixed allowlisted local maintenance and validation commands. No free-form shell input is available.",
        "Status: not run",
        "Run Quick Health Check",
        "Run Final MVP Validation",
        "Run Git Safety Check",
    ):
        assert text not in source


def test_operator_status_labels_are_plain_language() -> None:
    assert panel.operator_status_label(None) == "Not run yet"
    assert panel.operator_status_label("not_run") == "Not run yet"
    assert panel.operator_status_label("running") == "Running..."
    assert panel.operator_status_label("passed") == "Passed"
    assert panel.operator_status_label("success") == "Passed"
    assert panel.operator_status_label("failed") == "Needs attention"
    assert panel.operator_status_label("error") == "Needs attention"


def test_operator_panel_summary_defaults_are_safe() -> None:
    summary = panel.operator_panel_summary({})

    assert summary == {
        "last_check": "Not run",
        "system_status": "Waiting for check",
        "safety": "Local-only checks",
    }


def test_command_ids_and_argv_are_unchanged_for_core_actions() -> None:
    assert panel.get_command("quick_health_check").argv == (
        panel.sys.executable,
        "scripts/run_cka_final_mvp_release_validation.py",
    )
    assert panel.get_command("final_mvp_validation").argv == (
        panel.sys.executable,
        "scripts/run_cka_final_mvp_release_validation.py",
    )
    assert panel.get_command("git_safety_check").argv == ("__internal_git_safety_check__",)
    assert panel.get_command("full_test_suite").requires_confirmation is True


def test_grouping_uses_primary_and_advanced_sections_without_expanding_allowlist() -> None:
    groups = panel.command_groups()

    assert set(groups) == set(panel.PANEL_GROUP_ORDER)
    assert [command.command_id for command in groups["Main checks"]] == [
        "quick_health_check",
        "final_mvp_validation",
        "git_safety_check",
    ]
    assert len(panel.COMMAND_ALLOWLIST) == 12


def test_no_free_form_shell_added() -> None:
    source = panel_source()

    assert "shell=True" not in source
    assert "text_input" not in source
    assert "st.text_area" not in source
    assert "subprocess.run(" in source
