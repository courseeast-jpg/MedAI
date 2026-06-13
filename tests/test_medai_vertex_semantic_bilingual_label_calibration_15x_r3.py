"""No-live tests for 15X-R3 bilingual label calibration (deterministic aliases)."""
from __future__ import annotations

import inspect
import json

from execution.vertex_semantic_package_contract import (
    evidence_text_is_source_verbatim,
    label_matches_candidate,
    normalize_candidate_label,
)
from execution.vertex_semantic_calibration_batch import (
    build_calibration_fixtures,
    validate_calibration_response,
)

CANONICAL = "pH (bilingual)"
ALIASES = ["pH", "рН"]


def _cyrillic_fixture():
    return next(f for f in build_calibration_fixtures() if f.fixture_id == "cal_cyrillic")


def _resp(label, section="Urinalysis", evidence="Анализ мочи pH 6.0"):
    fx = _cyrillic_fixture()
    return {
        "package_family": fx.package_family,
        "review_required": True,
        "auto_accept": False,
        "semantic_findings": [{
            "label": label, "value": "6.0", "source_section": section,
            "evidence_text": evidence, "uncertainty": "x", "unknown_value": False, "source_faithful": True,
        }],
    }


def test_exact_label_and_section_passes() -> None:
    v = validate_calibration_response(_resp(CANONICAL), _cyrillic_fixture())
    assert v["verbatim_evidence_anchor_pass"] is True and v["hallucinated_field_count"] == 0


def test_declared_alias_same_section_passes() -> None:
    v = validate_calibration_response(_resp("pH"), _cyrillic_fixture())
    assert v["verbatim_evidence_anchor_pass"] is True and v["hallucinated_field_count"] == 0


def test_declared_cyrillic_alias_passes() -> None:
    assert label_matches_candidate("рН", CANONICAL, ALIASES) is True


def test_unicode_whitespace_normalized_declared_alias_passes_only_if_listed() -> None:
    assert label_matches_candidate("  pH  ", CANONICAL, ALIASES) is True
    # a non-listed normalization variant still fails
    assert label_matches_candidate("p H", CANONICAL, ALIASES) is False


def test_undeclared_alias_fails() -> None:
    assert label_matches_candidate("ph level", CANONICAL, ALIASES) is False
    assert label_matches_candidate("acidity", CANONICAL, ALIASES) is False
    v = validate_calibration_response(_resp("acidity"), _cyrillic_fixture())
    assert v["verbatim_evidence_anchor_pass"] is False and v["hallucinated_field_count"] >= 1


def test_unrelated_label_fails() -> None:
    assert label_matches_candidate("Glucose", CANONICAL, ALIASES) is False


def test_correct_label_wrong_section_fails() -> None:
    v = validate_calibration_response(_resp(CANONICAL, section="Other"), _cyrillic_fixture())
    assert v["verbatim_evidence_anchor_pass"] is False and v["hallucinated_field_count"] >= 1


def test_correct_alias_wrong_section_fails() -> None:
    v = validate_calibration_response(_resp("pH", section="Other"), _cyrillic_fixture())
    assert v["verbatim_evidence_anchor_pass"] is False and v["hallucinated_field_count"] >= 1


def test_exact_verbatim_evidence_still_required() -> None:
    # alias label is fine but paraphrased evidence must still fail
    v = validate_calibration_response(_resp("pH", evidence="urine pH six"), _cyrillic_fixture())
    assert v["verbatim_evidence_anchor_pass"] is False and v["hallucinated_field_count"] >= 1


def test_paraphrased_evidence_still_fails_for_canonical_label() -> None:
    v = validate_calibration_response(_resp(CANONICAL, evidence="approximately pH six"), _cyrillic_fixture())
    assert v["verbatim_evidence_anchor_pass"] is False


def test_cal_uncertainty_protection_preserved() -> None:
    fx = next(f for f in build_calibration_fixtures() if f.fixture_id == "cal_uncertainty")
    src = fx.source_visible_body
    assert evidence_text_is_source_verbatim("possible mild finding uncertain", src) is False
    assert evidence_text_is_source_verbatim("possible mild finding, uncertain", src) is True


def test_cal_cyrillic_declares_natural_aliases() -> None:
    fx = _cyrillic_fixture()
    aliases = fx.candidate_facts[0].get("accepted_label_aliases", [])
    assert "pH" in aliases
    assert "рН" in aliases


def test_simulated_r2_failure_now_resolves() -> None:
    # R2 failure: Vertex returned label "pH" -> finding_0_label_section_not_candidate.
    # With the declared alias it now passes the candidate match (evidence verbatim).
    v = validate_calibration_response(_resp("pH"), _cyrillic_fixture())
    assert "finding_0_label_section_not_candidate" not in v["errors"]
    assert v["verbatim_evidence_anchor_pass"] is True


def test_all_fixtures_still_consistent_under_faithful_responses() -> None:
    from execution.vertex_semantic_calibration_batch import build_fake_calibration_response
    for fx in build_calibration_fixtures():
        v = validate_calibration_response(build_fake_calibration_response(fx), fx)
        assert v["schema_validation_pass"] is True, fx.fixture_id
        assert v["verbatim_evidence_anchor_pass"] is True, fx.fixture_id
        assert v["hallucinated_field_count"] == 0, fx.fixture_id


def test_no_fuzzy_embedding_or_llm_label_judge_in_source() -> None:
    import execution.vertex_semantic_package_contract as mod
    src = inspect.getsource(mod.label_matches_candidate) + inspect.getsource(mod.normalize_candidate_label)
    for banned in ("embedding", "cosine", "similarity", "llm", "judge", "model.generate", "fuzz", "ratio("):
        assert banned not in src.lower()


def test_script_report_passes_and_recommends_r4() -> None:
    import scripts.run_medai_vertex_semantic_bilingual_label_calibration_15x_r3 as mod
    s = mod.build_report()["summary"]
    assert s["bilingual_label_calibration_passed"] is True
    assert s["cal_cyrillic_natural_label_supported"] is True
    assert s["cal_cyrillic_undeclared_drift_rejected"] is True
    assert s["evidence_anchor_strictness_preserved"] is True
    assert s["cal_uncertainty_protection_preserved"] is True
    assert s["fuzzy_or_embedding_or_llm_label_judge_used"] is False
    assert s["wrong_section_rejected_count"] >= 1


def test_safety_invariants_and_no_leak() -> None:
    import scripts.run_medai_vertex_semantic_bilingual_label_calibration_15x_r3 as mod
    s = mod.build_report()["summary"]
    assert s["active_written_count"] == 0
    assert s["active_mkb_record_created_count"] == 0
    assert s["auto_accept_true_count"] == 0
    assert s["live_call_made"] is False
    assert s["external_api_used"] is False
    blob = json.dumps(mod.build_report(), default=str, ensure_ascii=True)
    for token in ("ya29.", "AIza", "Bearer ", "Authorization", "C:\\", "/home/", "DOB", "MRN", "Accession"):
        assert token not in blob


def test_no_live_gate_in_source() -> None:
    import scripts.run_medai_vertex_semantic_bilingual_label_calibration_15x_r3 as mod
    src = inspect.getsource(mod)
    assert "MEDAI_VERTEX_CALIBRATION_BATCH_SYNTHETIC_LIVE_ALLOWED" not in src
    for marker in ("requests.", "urllib.request", "httpx.", "generate_content", "acquire_google_cloud_access_token"):
        assert marker not in src
