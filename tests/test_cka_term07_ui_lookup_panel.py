from __future__ import annotations

import json
import py_compile
import subprocess
import sys
from pathlib import Path

import pytest

from app.clinical_knowledge_terminology_lookup_viewer import (
    TERM07_FEATURE_FLAG,
    get_lookup_store_status,
    local_only_mode_enabled,
    normalize_source_filter,
    render_lookup_result_text,
    run_local_lookup,
    terminology_lookup_panel_enabled,
)
from clinical_knowledge.privacy.report_privacy import check_public_report_payload
from clinical_knowledge.terminology.local_store import LocalTerminologyStore
from clinical_knowledge.terminology.models import TerminologyConcept, TerminologySystem
from clinical_knowledge.terminology.term02_controlled_import import TERM02_DB_RELATIVE

ROOT = Path(__file__).resolve().parents[1]


def _term07_store(repo_root: Path) -> Path:
    db_path = repo_root / TERM02_DB_RELATIVE
    db_path.parent.mkdir(parents=True, exist_ok=True)
    store = LocalTerminologyStore(str(db_path))
    rx = store.register_source(TerminologySystem.RXNORM, safe_source_id="term07_rx", license_confirmed=True)
    loinc = store.register_source(TerminologySystem.LOINC, safe_source_id="term07_loinc", license_confirmed=True)
    store.add_concepts(
        [
            TerminologyConcept(
                concept_id="term07_rx_exact",
                system=TerminologySystem.RXNORM,
                code="RXTERM07",
                display="term07 rx exact",
                synthetic=False,
            ),
            TerminologyConcept(
                concept_id="term07_rx_ambiguous",
                system=TerminologySystem.RXNORM,
                code="RXTERM07AMB",
                display="term07 shared concept",
                synthetic=False,
            ),
        ],
        rx,
    )
    store.add_concepts(
        [
            TerminologyConcept(
                concept_id="term07_loinc_exact",
                system=TerminologySystem.LOINC,
                code="LTERM07",
                display="term07 loinc exact",
                synthetic=False,
            ),
            TerminologyConcept(
                concept_id="term07_loinc_ambiguous",
                system=TerminologySystem.LOINC,
                code="LTERM07AMB",
                display="term07 shared concept",
                synthetic=False,
            ),
        ],
        loinc,
    )
    return db_path


def test_feature_flag_default_off_and_local_only_default_on() -> None:
    assert terminology_lookup_panel_enabled({}) is False
    assert terminology_lookup_panel_enabled({TERM07_FEATURE_FLAG: "1"}) is True
    assert local_only_mode_enabled({}) is True
    assert local_only_mode_enabled({"MEDAI_LOCAL_ONLY": "0"}) is False


def test_store_status_missing_store_does_not_expose_path(tmp_path: Path) -> None:
    status = get_lookup_store_status(repo_root=tmp_path, env={TERM07_FEATURE_FLAG: "1"})
    summary = status.safe_public_summary()
    assert summary["store_available"] is False
    assert summary["private_path_displayed"] is False
    assert str(tmp_path) not in json.dumps(summary)


def test_lookup_hidden_by_default_and_missing_store_safe(tmp_path: Path) -> None:
    disabled = run_local_lookup("term07 rx exact", repo_root=tmp_path, env={})
    assert disabled["reason_codes"] == ["feature_flag_disabled"]
    missing = run_local_lookup("term07 rx exact", repo_root=tmp_path, env={TERM07_FEATURE_FLAG: "1"})
    assert missing["status"] == "unmapped"
    assert missing["reason_codes"] == ["store_missing"]
    assert missing["external_api_used"] is False


def test_exact_unknown_ambiguous_and_source_filter_behaviors(tmp_path: Path) -> None:
    _term07_store(tmp_path)
    env = {TERM07_FEATURE_FLAG: "1", "MEDAI_LOCAL_ONLY": "1"}
    exact = run_local_lookup("term07 rx exact", source_filter="rxnorm", repo_root=tmp_path, env=env)
    assert exact["status"] == "exact"
    assert exact["match_count"] == 1
    assert exact["source_systems"] == ["rxnorm"]
    unknown = run_local_lookup("term07 unknown stays unmapped", repo_root=tmp_path, env=env)
    assert unknown["status"] == "unmapped"
    assert unknown["match_count"] == 0
    assert unknown["no_code_hallucinated"] is True
    ambiguous = run_local_lookup("term07 shared concept", repo_root=tmp_path, env=env)
    assert ambiguous["status"] == "ambiguous"
    assert "manual_review_required" in ambiguous["reason_codes"]
    isolated = run_local_lookup("term07 rx exact", source_filter="loinc", repo_root=tmp_path, env=env)
    assert isolated["status"] == "unmapped"


def test_local_only_required_and_source_filter_validation(tmp_path: Path) -> None:
    _term07_store(tmp_path)
    blocked = run_local_lookup(
        "term07 rx exact",
        repo_root=tmp_path,
        env={TERM07_FEATURE_FLAG: "1", "MEDAI_LOCAL_ONLY": "0"},
    )
    assert blocked["reason_codes"] == ["local_only_mode_required"]
    assert normalize_source_filter("all") == ()
    assert normalize_source_filter("rxnorm") == ("rxnorm",)
    with pytest.raises(ValueError, match="unsupported_source_filter"):
        normalize_source_filter("umls")


def test_result_text_and_public_payload_are_privacy_clean(tmp_path: Path) -> None:
    _term07_store(tmp_path)
    result = run_local_lookup(
        "term07 rx exact",
        source_filter="rxnorm",
        repo_root=tmp_path,
        env={TERM07_FEATURE_FLAG: "1", "MEDAI_LOCAL_ONLY": "1"},
    )
    rendered = render_lookup_result_text(result)
    assert "Status: exact" in rendered
    public_payload = {
        "status": result["status"],
        "match_count": result["match_count"],
        "source_systems": result["source_systems"],
        "reason_codes": result["reason_codes"],
        "external_api_used": result["external_api_used"],
        "read_only": result["read_only"],
    }
    assert str(tmp_path) not in json.dumps(public_payload)
    assert check_public_report_payload(public_payload).passed is True


def test_ui_hook_exists_without_default_tab_activation() -> None:
    py_compile.compile(str(ROOT / "app" / "main.py"), doraise=True)
    text = (ROOT / "app" / "main.py").read_text(encoding="utf-8")
    assert "TERMINOLOGY_LOOKUP_TAB" in text
    assert "terminology_lookup_panel_enabled" in text
    assert "render_terminology_lookup_panel" in text


def test_validation_script_runs_and_report_is_safe() -> None:
    proc = subprocess.run(
        [sys.executable, "scripts/run_cka_term07_ui_lookup_panel_validation.py"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    report_path = ROOT / "reports" / "cka_term07_ui_lookup_panel" / "cka_term07_ui_lookup_panel_report.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["conclusion"] == "cka_term07_ui_lookup_panel_ready"
    assert payload["feature_flag_default_enabled"] is False
    assert payload["external_api_used"] is False
    assert payload["b07_integrated"] is False
    assert payload["ddi_status_cleared"] is False
    assert payload["hypothesis_promoted"] is False
    assert payload["privacy_report_clean"] is True
    rendered = json.dumps(payload)
    assert "terminology_data/" not in rendered.replace("\\", "/")
    assert "LICENSE_ACK_PRIVATE" not in rendered
    assert "RXNCONSO" not in rendered
    assert "Loinc.csv" not in rendered


def test_no_private_or_runtime_files_staged() -> None:
    staged = subprocess.run(["git", "diff", "--cached", "--name-only"], cwd=ROOT, text=True, capture_output=True, check=False)
    assert staged.returncode == 0
    lowered = staged.stdout.lower().replace("\\", "/")
    assert "terminology_data" not in lowered
    assert "data/terminology" not in lowered
    assert "license_ack_private" not in lowered
    assert ".rrf" not in lowered
    assert ".sqlite" not in lowered
    assert ".db" not in lowered
