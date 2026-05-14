from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from clinical_knowledge.privacy.report_privacy import check_public_report_payload

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports" / "cka_term04_integration_readiness_blueprint"


def test_term04_report_files_exist() -> None:
    for name in [
        "CKA_TERM04_INTEGRATION_READINESS_BLUEPRINT.md",
        "CKA_TERM04_SAFETY_CONTRACT.md",
        "CKA_TERM04_FUTURE_BLOCK_PLAN.md",
        "cka_term04_integration_readiness_report.json",
        "cka_term04_integration_readiness_report.md",
    ]:
        assert (REPORT_DIR / name).exists(), name


def test_future_block_plan_exists_and_is_guarded() -> None:
    text = (REPORT_DIR / "CKA_TERM04_FUTURE_BLOCK_PLAN.md").read_text(encoding="utf-8")
    assert "TERM-05 Synthetic Read-Only Terminology Adapter" in text
    assert "TERM-06 Private-Store Read-Only Adapter Validation" in text
    assert "TERM-07 UI-Only Terminology Lookup Panel" in text
    assert "TERM-08 Hypothesis-Only Coding Annotation Pilot" in text
    assert "B07-TERM Opt-In Integration" in text
    assert "Blocked:" in text


def test_forbidden_behaviors_are_documented() -> None:
    combined = "\n".join(path.read_text(encoding="utf-8") for path in REPORT_DIR.glob("*.md"))
    for phrase in [
        "must not generate clinical advice",
        "must not clear or downgrade DDI status",
        "Unknown terms must return unmapped",
        "Ambiguous terms must return ambiguous/manual-review",
        "Promote terminology lookup results to accepted clinical facts",
        "Invent a code for unknown terms",
    ]:
        assert phrase in combined


def test_report_privacy_clean() -> None:
    for path in REPORT_DIR.glob("*"):
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8")) if path.suffix == ".json" else {path.name: path.read_text(encoding="utf-8")}
        check = check_public_report_payload(payload)
        assert check.passed, path.name
        rendered = json.dumps(payload)
        assert "terminology_data/" not in rendered.replace("\\", "/")
        assert "LICENSE_ACK_PRIVATE" not in rendered
        assert "RXNCONSO" not in rendered
        assert "Loinc.csv" not in rendered


def test_report_flags_no_runtime_or_external_behavior() -> None:
    payload = json.loads((REPORT_DIR / "cka_term04_integration_readiness_report.json").read_text(encoding="utf-8"))
    assert payload["external_api_used"] is False
    assert payload["clinical_decision_logic_changed"] is False
    assert payload["b07_integration_changed"] is False
    assert payload["b07_integration_enabled"] is False
    assert payload["ocr_extractor_safety_gates_changed"] is False
    assert payload["terminology_data_staged"] is False
    assert payload["local_db_index_committed"] is False


def test_validation_script_passes() -> None:
    proc = subprocess.run(
        [sys.executable, "scripts/run_cka_term04_integration_readiness_validation.py"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "cka_term04_integration_readiness_blueprint_ready" in proc.stdout
