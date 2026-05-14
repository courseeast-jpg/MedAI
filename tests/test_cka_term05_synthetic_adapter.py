from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from clinical_knowledge.privacy.report_privacy import check_public_report_payload
from clinical_knowledge.terminology.term05_read_only_adapter import (
    build_synthetic_read_only_adapter,
    run_term05_synthetic_adapter_validation,
)

ROOT = Path(__file__).resolve().parents[1]


def test_exact_lookup_for_synthetic_rxnorm() -> None:
    adapter = build_synthetic_read_only_adapter()
    result = adapter.lookup("aspirin", source_filter=["rxnorm"])
    assert result.status == "exact"
    assert result.confidence_label == "high"
    assert len(result.matches) == 1
    assert result.matches[0].system == "rxnorm"
    assert result.read_only is True


def test_exact_lookup_for_synthetic_loinc() -> None:
    adapter = build_synthetic_read_only_adapter()
    result = adapter.lookup("glucose synthetic lab", source_filter=["loinc"])
    assert result.status == "exact"
    assert len(result.matches) == 1
    assert result.matches[0].system == "loinc"


def test_code_lookup_works() -> None:
    adapter = build_synthetic_read_only_adapter()
    result = adapter.lookup_code("LOINC001", source_filter=["loinc"])
    assert result.status == "exact"
    assert len(result.matches) == 1
    assert result.matches[0].code == "LOINC001"


def test_source_filter_isolation() -> None:
    adapter = build_synthetic_read_only_adapter()
    result = adapter.lookup("glucose synthetic lab", source_filter=["rxnorm"])
    assert result.status == "unmapped"
    assert result.matches == ()
    assert "no_code_hallucinated" in result.reason_codes


def test_unknown_returns_unmapped_without_hallucination() -> None:
    adapter = build_synthetic_read_only_adapter()
    result = adapter.lookup("term05 unknown does not exist")
    assert result.status == "unmapped"
    assert result.matches == ()
    assert result.no_code_hallucinated is True


def test_ambiguous_returns_manual_review() -> None:
    adapter = build_synthetic_read_only_adapter()
    result = adapter.lookup("aspirin")
    assert result.status == "ambiguous"
    assert len(result.matches) == 2
    assert "manual_review_required" in result.reason_codes
    assert result.confidence_label == "manual_review"


def test_determinism_and_normalization() -> None:
    adapter = build_synthetic_read_only_adapter()
    a = adapter.lookup("aspirin", source_filter=["rxnorm"])
    b = adapter.lookup("  ASPIRIN  ", source_filter=["rxnorm"])
    c = adapter.lookup("aspirin", source_filter=["rxnorm"])
    assert b.normalized_query == "aspirin"
    assert a.status == b.status == c.status
    assert [m.safe_public_summary() for m in a.matches] == [m.safe_public_summary() for m in b.matches]
    assert a.safe_public_summary() == c.safe_public_summary()


def test_adapter_never_writes_or_generates_clinical_advice() -> None:
    result = build_synthetic_read_only_adapter().lookup("aspirin", source_filter=["rxnorm"])
    assert result.external_api_used is False
    assert result.clinical_advice_generated is False
    assert result.dosing_advice_generated is False
    assert result.mkb_write_performed is False
    assert result.b07_integrated is False
    assert result.ddi_status_cleared is False
    assert result.hypothesis_promoted is False


def test_validation_summary_passes() -> None:
    summary = run_term05_synthetic_adapter_validation()
    assert summary.conclusion == "cka_term05_synthetic_adapter_ready"
    assert summary.cases_failed == 0
    assert summary.private_store_accessed is False
    assert summary.terminology_data_accessed is False
    assert summary.data_terminology_accessed is False


def test_public_report_privacy_clean() -> None:
    payload = run_term05_synthetic_adapter_validation().safe_public_summary()
    rendered = json.dumps(payload)
    assert "terminology_data/" not in rendered.replace("\\", "/")
    assert "data/terminology" not in rendered.replace("\\", "/")
    assert "LICENSE_ACK_PRIVATE" not in rendered
    assert "RXNCONSO" not in rendered
    assert "Loinc.csv" not in rendered
    check = check_public_report_payload(payload)
    assert check.passed is True


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


def test_validation_script_runs() -> None:
    proc = subprocess.run(
        [sys.executable, "scripts/run_cka_term05_synthetic_adapter_validation.py"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "cka_term05_synthetic_adapter_ready" in proc.stdout
