from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from app import operator_control_panel as panel
from clinical_knowledge.terminology.b07_term_opt_in import read_b07_term_flag_state


def test_allowlist_contains_required_buttons() -> None:
    labels = {command.label for command in panel.COMMAND_ALLOWLIST.values()}
    assert "Run Quick Health Check" in labels
    assert "Run Final MVP Validation" in labels
    assert "Run Full Test Suite" in labels
    assert "Run Terminology Source Preflight" in labels
    assert "Run Terminology Inventory" in labels
    assert "Run B07-TERM Validation" in labels
    assert "Run ROUTE-FIX Validation" in labels
    assert "Run Focused Routing Tests" in labels
    assert "Run Git Safety Check" in labels
    assert "Show Last Validation Reports" in labels
    assert "Show Release Tags" in labels
    assert "Verify Final Release Bundle" in labels


def test_unknown_command_id_rejected() -> None:
    with pytest.raises(ValueError):
        panel.run_operator_command("not_allowed")


def test_no_free_form_shell_execution() -> None:
    for command in panel.COMMAND_ALLOWLIST.values():
        assert isinstance(command.argv, tuple)
        assert command.argv
        assert "cmd.exe" not in command.argv
        assert "powershell" not in command.argv
    source = Path(panel.__file__).read_text(encoding="utf-8")
    assert "shell=True" not in source


def test_command_labels_map_to_fixed_commands() -> None:
    final_mvp = panel.get_command("final_mvp_validation")
    assert final_mvp.argv == (sys.executable, "scripts/run_cka_final_mvp_release_validation.py")
    inventory = panel.get_command("terminology_inventory")
    assert inventory.argv == (
        sys.executable,
        "scripts/run_medai_terminology_inventory.py",
        "--terminology-root",
        "terminology_data",
    )
    assert panel.get_command("full_test_suite").requires_confirmation is True


def test_missing_script_handled_safely(tmp_path: Path) -> None:
    result = panel.run_operator_command("final_mvp_validation", repo_root=tmp_path)
    assert result.status == "failed"
    assert result.exit_code is None
    assert "Script missing" in result.output_summary


def test_timeout_handled_safely(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    script = tmp_path / "scripts" / "run_cka_final_mvp_release_validation.py"
    script.parent.mkdir(parents=True)
    script.write_text("print('ok')\n", encoding="utf-8")

    def raise_timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=1, output="partial output", stderr="private error")

    monkeypatch.setattr(panel.subprocess, "run", raise_timeout)
    result = panel.run_operator_command("final_mvp_validation", repo_root=tmp_path)

    assert result.status == "failed"
    assert result.exit_code is None
    assert "Timed out" in result.stderr_summary
    assert "partial output" in result.output_summary


def test_redaction_removes_private_paths_and_ack_name() -> None:
    text = r"G:\\Codex\\repo\\terminology_data\\LICENSE_ACK_PRIVATE.json token=abcdefghijklmnopqrstuvwxyz"
    redacted = panel.redact_output(text)
    assert "G:\\" not in redacted
    assert "LICENSE_ACK_PRIVATE.json" not in redacted
    assert "abcdefghijklmnopqrstuvwxyz" not in redacted


def test_no_private_ack_contents_read_by_panel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "terminology_data").mkdir()
    (tmp_path / "terminology_data" / "LICENSE_ACK_PRIVATE.json").write_text("PRIVATE_ACK_SENTINEL", encoding="utf-8")

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args=args[0], returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(panel.subprocess, "run", fake_run)
    script = tmp_path / "scripts" / "run_medai_terminology_inventory.py"
    script.parent.mkdir(parents=True)
    script.write_text("print('ok')\n", encoding="utf-8")

    result = panel.run_operator_command("terminology_inventory", repo_root=tmp_path)
    assert result.status == "passed"
    assert "PRIVATE_ACK_SENTINEL" not in result.output_summary


def test_source_rows_are_not_printed_when_subprocess_output_is_safe(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    script = tmp_path / "scripts" / "run_medai_terminology_inventory.py"
    script.parent.mkdir(parents=True)
    script.write_text("print('ok')\n", encoding="utf-8")

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args=args[0], returncode=0, stdout='{"conclusion":"ready"}', stderr="")

    monkeypatch.setattr(panel.subprocess, "run", fake_run)
    result = panel.run_operator_command("terminology_inventory", repo_root=tmp_path)
    assert "RXNORM_ROW_SHOULD_NOT_PRINT" not in result.output_summary
    assert "LOINC_ROW_SHOULD_NOT_PRINT" not in result.output_summary


def test_b07_flags_default_safety_unchanged() -> None:
    flags = read_b07_term_flag_state(env={})
    assert flags.enabled is False
    assert flags.read_only is True
    assert flags.allow_writes is False


def test_ui_import_does_not_run_commands(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def fake_run(*args, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(args=args[0], returncode=0, stdout="", stderr="")

    monkeypatch.setattr(panel.subprocess, "run", fake_run)
    assert panel.command_groups()
    assert calls == []


def test_app_main_includes_operator_control_panel_tab() -> None:
    main_path = Path("app/main.py")
    text = main_path.read_text(encoding="utf-8")
    assert "MedAI Operator Control Panel" in text
    assert "render_operator_control_panel" in text
    assert "ExecutionPipeline(" in text


def test_public_report_privacy_clean_if_present() -> None:
    report_dir = Path("reports/medai_ui_ops_01")
    if not report_dir.exists():
        return
    from clinical_knowledge.privacy.report_privacy import check_public_report_payload

    for path in report_dir.glob("*"):
        if not path.is_file():
            continue
        result = check_public_report_payload(path.read_text(encoding="utf-8"))
        assert result.passed, result.leak_examples_redacted
