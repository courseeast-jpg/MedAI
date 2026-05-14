from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from clinical_knowledge.privacy.report_privacy import check_public_report_payload

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports" / "b07_term_opt_in_planning"
REQUIRED_FILES = [
    "B07_TERM_OPT_IN_INTEGRATION_PLAN.md",
    "B07_TERM_SAFETY_CONTRACT.md",
    "B07_TERM_ROLLBACK_PLAN.md",
    "B07_TERM_IMPLEMENTATION_PROMPT.md",
    "b07_term_opt_in_planning_report.json",
    "b07_term_opt_in_planning_report.md",
]


def _combined_text() -> str:
    return "\n".join((REPORT_DIR / name).read_text(encoding="utf-8") for name in REQUIRED_FILES)


def test_planning_files_and_prompt_exist() -> None:
    for name in REQUIRED_FILES:
        assert (REPORT_DIR / name).exists(), name
    assert "Start B07-TERM-01" in (REPORT_DIR / "B07_TERM_IMPLEMENTATION_PROMPT.md").read_text(encoding="utf-8")


def test_no_runtime_implementation_file_created() -> None:
    assert not (ROOT / "clinical_knowledge" / "terminology" / "b07_term_integration.py").exists()
    assert not (ROOT / "clinical_knowledge" / "terminology" / "b07_term_opt_in.py").exists()


def test_forbidden_behaviors_and_feature_flags_are_listed() -> None:
    text = _combined_text()
    for phrase in [
        "MEDAI_B07_TERMINOLOGY_OPT_IN=false",
        "MEDAI_TERMINOLOGY_LOOKUP_ENABLED=false",
        "MEDAI_TERMINOLOGY_HYPOTHESIS_ANNOTATION=false",
        "MEDAI_TERMINOLOGY_READ_ONLY=true",
        "MEDAI_TERMINOLOGY_ALLOW_WRITES=false",
        "No accepted clinical fact",
        "No hypothesis promotion",
        "No DDI status clearing",
        "No external API",
        "Unknown terms must remain unmapped",
        "Ambiguous terms must remain ambiguous",
    ]:
        assert phrase in text


def test_rollback_plan_exists_and_preserves_disabled_defaults() -> None:
    text = (REPORT_DIR / "B07_TERM_ROLLBACK_PLAN.md").read_text(encoding="utf-8")
    assert "Immediate Disable" in text
    assert "MEDAI_B07_TERMINOLOGY_OPT_IN=false" in text
    assert "MEDAI_TERMINOLOGY_ALLOW_WRITES=false" in text


def test_report_flags_are_safe_and_design_only() -> None:
    payload = json.loads((REPORT_DIR / "b07_term_opt_in_planning_report.json").read_text(encoding="utf-8"))
    assert payload["conclusion"] == "b07_term_opt_in_planning_ready"
    assert payload["design_only"] is True
    assert payload["b07_runtime_behavior_changed"] is False
    assert payload["b07_integration_implemented"] is False
    assert payload["accepted_clinical_facts_created"] is False
    assert payload["external_api_used"] is False
    assert payload["ocr_extractor_safety_gates_changed"] is False


def test_public_reports_are_privacy_clean() -> None:
    for name in REQUIRED_FILES:
        path = REPORT_DIR / name
        payload = json.loads(path.read_text(encoding="utf-8")) if path.suffix == ".json" else {name: path.read_text(encoding="utf-8")}
        check = check_public_report_payload(payload)
        assert check.passed is True, name
    rendered = _combined_text()
    assert "LICENSE_ACK_PRIVATE" not in rendered
    assert "RXNCONSO" not in rendered
    assert "Loinc.csv" not in rendered


def test_validation_script_runs() -> None:
    proc = subprocess.run(
        [sys.executable, "scripts/run_b07_term_opt_in_planning_validation.py"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "b07_term_opt_in_planning_ready" in proc.stdout


def test_no_private_or_runtime_files_staged() -> None:
    staged = subprocess.run(["git", "diff", "--cached", "--name-only"], cwd=ROOT, text=True, capture_output=True, check=False)
    assert staged.returncode == 0
    lowered = staged.stdout.lower().replace("\\", "/")
    assert "terminology_data" not in lowered
    assert "data/terminology" not in lowered
    assert "license_ack_private" not in lowered
    assert ".rrf" not in lowered
    assert ".csv" not in lowered
    assert ".sqlite" not in lowered
    assert ".db" not in lowered
