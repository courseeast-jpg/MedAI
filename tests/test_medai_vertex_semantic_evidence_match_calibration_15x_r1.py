"""No-live tests for 15X-R1 Vertex evidence-match calibration (verbatim only)."""
from __future__ import annotations

import inspect
import json

from execution.vertex_semantic_package_contract import (
    PROMPT_REQUIRED_PHRASES,
    build_vertex_semantic_prompt,
    evidence_text_is_source_verbatim,
    normalize_evidence_text,
)
from execution.vertex_semantic_calibration_batch import build_calibration_fixtures
from app.ai_package_run_review_preview import build_run_review_package_previews

SOURCE = "SYNTHETIC note: 'possible mild finding, uncertain' in section Results; value 5 mg/dL within range."


def test_exact_evidence_match_passes() -> None:
    assert evidence_text_is_source_verbatim("possible mild finding, uncertain", SOURCE) is True


def test_whitespace_normalized_verbatim_substring_passes() -> None:
    assert evidence_text_is_source_verbatim("possible    mild  finding,   uncertain", SOURCE) is True
    assert normalize_evidence_text("a   b\tc\n") == "a b c"


def test_unicode_normalized_verbatim_substring_passes() -> None:
    # smart quotes normalize to straight quotes; underlying words unchanged
    assert evidence_text_is_source_verbatim("‘possible mild finding, uncertain’", SOURCE) is True


def test_plain_substring_of_source_passes() -> None:
    assert evidence_text_is_source_verbatim("value 5 mg/dL within range", SOURCE) is True


def test_paraphrase_evidence_fails() -> None:
    # the recorded 15X failure shape: comma dropped / reworded
    assert evidence_text_is_source_verbatim("possible mild finding uncertain", SOURCE) is False
    assert evidence_text_is_source_verbatim("a slight finding that is unclear", SOURCE) is False


def test_inferred_evidence_fails() -> None:
    assert evidence_text_is_source_verbatim("patient is clinically stable", SOURCE) is False


def test_wrong_section_fails_when_section_constrained() -> None:
    assert evidence_text_is_source_verbatim("possible mild finding, uncertain", SOURCE, section_text="header metadata only") is False
    # right section still passes
    assert evidence_text_is_source_verbatim("value 5 mg/dL within range", SOURCE, section_text="value 5 mg/dL within range") is True


def test_null_and_empty_evidence_fail() -> None:
    assert evidence_text_is_source_verbatim(None, SOURCE) is False
    assert evidence_text_is_source_verbatim("", SOURCE) is False


def test_recorded_15x_cal_uncertainty_failure_preserved() -> None:
    fixture = next(f for f in build_calibration_fixtures() if f.fixture_id == "cal_uncertainty")
    src = fixture.source_visible_body
    # paraphrase that caused finding_0_not_source_anchored stays a failure
    assert evidence_text_is_source_verbatim("possible mild finding uncertain", src) is False
    # a true verbatim span would pass (rule is verbatim, not blanket reject)
    assert evidence_text_is_source_verbatim("possible mild finding, uncertain", src) is True


def test_prompt_contract_hardened_for_verbatim_evidence() -> None:
    assert "evidence_verbatim" in PROMPT_REQUIRED_PHRASES
    phrase = PROMPT_REQUIRED_PHRASES["evidence_verbatim"].lower()
    assert "verbatim" in phrase
    assert "do not paraphrase" in phrase
    assert "set evidence_text to null" in phrase
    # the hardened phrase appears in every generated prompt
    for preview in build_run_review_package_previews():
        assert PROMPT_REQUIRED_PHRASES["evidence_verbatim"] in build_vertex_semantic_prompt(preview)


def test_no_embeddings_or_llm_judge_in_matcher_source() -> None:
    import execution.vertex_semantic_package_contract as mod

    src = inspect.getsource(mod.evidence_text_is_source_verbatim) + inspect.getsource(mod.normalize_evidence_text)
    for banned in ("embedding", "cosine", "similarity", "llm", "judge", "model.generate", "openai", "genai"):
        assert banned not in src.lower()


def test_calibration_script_runs_and_recommends_rerun() -> None:
    import scripts.run_medai_vertex_semantic_evidence_match_calibration_15x_r1 as mod

    report = mod.build_report()
    s = report["summary"]
    assert s["evidence_match_calibration_passed"] is True
    assert s["recorded_15x_failure_preserved"] is True
    assert s["prompt_contract_hardened"] is True
    assert s["evidence_text_verbatim_required"] is True
    assert s["embedding_or_llm_judge_used"] is False
    assert s["paraphrase_rejected_count"] >= 1
    assert s["inferred_evidence_rejected_count"] >= 1
    assert s["wrong_section_rejected_count"] >= 1
    assert all(c["correct"] for c in report["cases"])


def test_no_active_write_auto_accept_or_live_call() -> None:
    import scripts.run_medai_vertex_semantic_evidence_match_calibration_15x_r1 as mod

    s = mod.build_report()["summary"]
    assert s["active_written_count"] == 0
    assert s["active_mkb_record_created_count"] == 0
    assert s["auto_accept_true_count"] == 0
    assert s["live_call_made"] is False
    assert s["external_api_used"] is False
    assert s["review_required"] is True


def test_no_credentials_or_private_markers() -> None:
    import scripts.run_medai_vertex_semantic_evidence_match_calibration_15x_r1 as mod

    blob = json.dumps(mod.build_report(), default=str, ensure_ascii=True)
    for token in ("ya29.", "AIza", "Bearer ", "Authorization", "C:\\", "/home/", "DOB", "MRN", "Accession"):
        assert token not in blob


def test_no_live_gate_set_or_required_in_source() -> None:
    import scripts.run_medai_vertex_semantic_evidence_match_calibration_15x_r1 as mod

    src = inspect.getsource(mod)
    assert "MEDAI_VERTEX_CALIBRATION_BATCH_SYNTHETIC_LIVE_ALLOWED" not in src
    for marker in ("requests.", "urllib.request", "httpx.", "generate_content", "acquire_google_cloud_access_token"):
        assert marker not in src
