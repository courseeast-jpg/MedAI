from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from clinical_knowledge.privacy.report_privacy import check_public_report_payload
from clinical_knowledge.terminology.term05_read_only_adapter import build_synthetic_read_only_adapter
from clinical_knowledge.terminology.term08_hypothesis_annotation_pilot import (
    TERM08_FEATURE_FLAG,
    annotate_candidate_term,
    summarize_annotation_for_public_report,
    term08_annotation_enabled,
)

ROOT = Path(__file__).resolve().parents[1]


def test_feature_flag_off_preserves_current_behavior() -> None:
    adapter = build_synthetic_read_only_adapter()
    assert term08_annotation_enabled({}) is False
    assert annotate_candidate_term("aspirin", adapter=adapter, env={}) is None


def test_exact_lookup_creates_hypothesis_only_annotation() -> None:
    adapter = build_synthetic_read_only_adapter()
    annotation = annotate_candidate_term(
        "aspirin",
        adapter=adapter,
        source_filter=["rxnorm"],
        env={TERM08_FEATURE_FLAG: "1", "MEDAI_LOCAL_ONLY": "1"},
    )
    assert annotation is not None
    assert annotation.terminology_status == "exact"
    assert annotation.source_system == "rxnorm"
    assert len(annotation.candidate_codes) == 1
    assert annotation.annotation_tier == "hypothesis"
    assert annotation.requires_review is True
    assert annotation.read_only_lookup is True
    assert annotation.writes_active_fact is False


def test_unknown_lookup_unmapped_no_invented_code() -> None:
    adapter = build_synthetic_read_only_adapter()
    annotation = annotate_candidate_term(
        "term08 unknown remains unmapped",
        adapter=adapter,
        env={TERM08_FEATURE_FLAG: "1"},
    )
    assert annotation is not None
    assert annotation.terminology_status == "unmapped"
    assert annotation.candidate_codes == ()
    assert annotation.no_code_hallucinated is True
    assert "unmapped_no_code_hallucinated" in annotation.reason_codes


def test_ambiguous_lookup_requires_manual_review() -> None:
    adapter = build_synthetic_read_only_adapter()
    annotation = annotate_candidate_term("aspirin", adapter=adapter, env={TERM08_FEATURE_FLAG: "1"})
    assert annotation is not None
    assert annotation.terminology_status == "ambiguous"
    assert annotation.source_system is None
    assert annotation.requires_review is True
    assert "ambiguity_requires_manual_review" in annotation.reason_codes


def test_annotation_never_changes_clinical_safety_boundaries() -> None:
    annotation = annotate_candidate_term(
        "aspirin",
        adapter=build_synthetic_read_only_adapter(),
        source_filter=["rxnorm"],
        env={TERM08_FEATURE_FLAG: "1"},
    )
    assert annotation is not None
    assert annotation.writes_active_fact is False
    assert annotation.promotes_hypothesis is False
    assert annotation.clears_ddi_status is False
    assert annotation.dosing_advice_generated is False
    assert annotation.prescribing_advice_generated is False
    assert annotation.clinical_recommendation_generated is False
    assert annotation.external_api_used is False
    assert annotation.b07_integrated is False


def test_public_summary_and_privacy_clean() -> None:
    annotation = annotate_candidate_term(
        "aspirin",
        adapter=build_synthetic_read_only_adapter(),
        source_filter=["rxnorm"],
        env={TERM08_FEATURE_FLAG: "1"},
    )
    payload = summarize_annotation_for_public_report(annotation)
    assert payload["annotation_tier"] == "hypothesis"
    assert payload["requires_review"] is True
    assert payload["writes_active_fact"] is False
    assert check_public_report_payload(payload).passed is True


def test_validation_script_runs_and_report_is_safe() -> None:
    proc = subprocess.run(
        [sys.executable, "scripts/run_cka_term08_hypothesis_annotation_validation.py"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    report_path = ROOT / "reports" / "cka_term08_hypothesis_annotation_pilot" / "cka_term08_hypothesis_annotation_report.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["conclusion"] == "cka_term08_hypothesis_annotation_ready"
    assert payload["feature_flag_default_enabled"] is False
    assert payload["cases_failed"] == 0
    assert payload["writes_active_fact"] is False
    assert payload["clears_ddi_status"] is False
    assert payload["promotes_hypothesis"] is False
    assert payload["external_api_used"] is False
    assert payload["clinical_recommendations_generated"] is False
    assert payload["dosing_advice_generated"] is False
    assert payload["privacy_report_clean"] is True
    rendered = json.dumps(payload)
    assert "terminology_data/" not in rendered.replace("\\", "/")
    assert "data/terminology" not in rendered.replace("\\", "/")
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
    assert ".csv" not in lowered
    assert ".sqlite" not in lowered
    assert ".db" not in lowered
