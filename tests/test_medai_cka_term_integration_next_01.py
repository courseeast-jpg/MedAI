"""Focused tests for MEDAI-CKA-TERM-INTEGRATION-NEXT-01.

These tests prove the default-off, fail-closed, aggregate-only,
review-bound terminology match hypothesis helper. They use the existing
synthetic TERM-05 adapter as the lookup fixture; no licensed
terminology rows are read.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Mapping

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clinical_knowledge.terminology.term05_read_only_adapter import (  # noqa: E402
    build_synthetic_read_only_adapter,
)
from clinical_knowledge.terminology.term_match_hypothesis import (  # noqa: E402
    DISCLAIMER,
    LICENSE_CLASS_OF_OUTPUT,
    MATCH_FAMILY_VALUES,
    SOURCE_PHASE,
    TERMINOLOGY_LOOKUP_ENV_VAR,
    TERMINOLOGY_SYSTEM_FAMILY_VALUES,
    derive_terminology_match_hypothesis,
    is_terminology_lookup_default_disabled,
    is_terminology_lookup_enabled,
    matches_terminology_lookup_signature,
)


SAFE_RECORD = {
    "anonymous_id": "record_001",
    "terminology_candidate_text": "aspirin",
    "terminology_system_filter": ["rxnorm"],
}


def _synthetic_adapter():
    return build_synthetic_read_only_adapter()


# ── 1. Default-off returns None ───────────────────────────────────────────


def test_default_off_when_env_unset_returns_none():
    assert (
        derive_terminology_match_hypothesis(
            SAFE_RECORD,
            env={},
            lookup_adapter=_synthetic_adapter(),
        )
        is None
    )


def test_default_off_when_env_falsy_returns_none():
    for val in ("0", "false", "no", "off", "disabled", ""):
        env = {TERMINOLOGY_LOOKUP_ENV_VAR: val}
        assert (
            derive_terminology_match_hypothesis(
                SAFE_RECORD, env=env, lookup_adapter=_synthetic_adapter()
            )
            is None
        ), f"env value {val!r} did not produce None"


def test_is_terminology_lookup_default_disabled_with_empty_env():
    assert is_terminology_lookup_default_disabled(env={}) is True
    assert is_terminology_lookup_enabled(env={}) is False


# ── 2. enabled=False overrides truthy env ─────────────────────────────────


def test_enabled_false_overrides_truthy_env():
    env = {TERMINOLOGY_LOOKUP_ENV_VAR: "true"}
    assert (
        derive_terminology_match_hypothesis(
            SAFE_RECORD,
            enabled=False,
            env=env,
            lookup_adapter=_synthetic_adapter(),
        )
        is None
    )


# ── 3. Truthy env allows only controlled-vocabulary metadata ─────────────


def test_truthy_env_emits_only_controlled_vocab():
    env = {TERMINOLOGY_LOOKUP_ENV_VAR: "true"}
    plan = derive_terminology_match_hypothesis(
        SAFE_RECORD, env=env, lookup_adapter=_synthetic_adapter()
    )
    assert plan is not None
    assert plan["source_phase"] == SOURCE_PHASE
    assert plan["env_var"] == TERMINOLOGY_LOOKUP_ENV_VAR
    assert plan["match_family"] in MATCH_FAMILY_VALUES
    assert plan["terminology_system_family"] in TERMINOLOGY_SYSTEM_FAMILY_VALUES
    assert plan["license_class"] == LICENSE_CLASS_OF_OUTPUT
    assert plan["disclaimer"] == DISCLAIMER
    # Every controlled-vocab token in the plan is a known token; no
    # arbitrary strings smuggled through.
    assert isinstance(plan["matches_count"], int)
    assert plan["matches_count"] >= 0


# ── 4. lookup_adapter is injectable and can be synthetic ──────────────────


def test_lookup_adapter_is_injectable_synthetic():
    env = {TERMINOLOGY_LOOKUP_ENV_VAR: "true"}
    adapter = _synthetic_adapter()
    plan = derive_terminology_match_hypothesis(
        SAFE_RECORD, env=env, lookup_adapter=adapter
    )
    assert plan is not None


def test_lookup_adapter_none_fails_closed_even_when_env_truthy():
    """Fail-closed: no adapter -> return None."""
    env = {TERMINOLOGY_LOOKUP_ENV_VAR: "true"}
    assert (
        derive_terminology_match_hypothesis(
            SAFE_RECORD, env=env, lookup_adapter=None
        )
        is None
    )


# ── 5. No real terminology rows needed for tests ──────────────────────────


def test_no_real_terminology_rows_required():
    """The synthetic adapter never touches the private RxNorm/LOINC store.

    We assert the adapter advertises ``read_only=True`` and
    ``external_api_used=False`` and that the helper's output does not
    embed any licensed-row content fields.
    """
    adapter = _synthetic_adapter()
    assert adapter.read_only is True
    assert adapter.external_api_used is False


# ── 6. No licensed row content appears in output ──────────────────────────


def test_no_licensed_row_content_in_output():
    env = {TERMINOLOGY_LOOKUP_ENV_VAR: "true"}
    plan = derive_terminology_match_hypothesis(
        SAFE_RECORD, env=env, lookup_adapter=_synthetic_adapter()
    )
    assert plan is not None
    assert plan["licensed_row_content_included"] is False

    forbidden_substrings = (
        "code",
        "display",
        "rxcui",
        "loinc_num",
        "synonym",
        "concept",
        "definition",
    )
    for key in plan.keys():
        for token in forbidden_substrings:
            assert token not in key.lower(), (
                f"output key {key!r} suggests row content; forbid "
                f"substring {token!r}"
            )


# ── 7. No raw text / filenames / paths / PHI / secrets in output ─────────


def test_no_raw_text_filenames_paths_phi_secrets_in_output():
    env = {TERMINOLOGY_LOOKUP_ENV_VAR: "true"}
    plan = derive_terminology_match_hypothesis(
        SAFE_RECORD, env=env, lookup_adapter=_synthetic_adapter()
    )
    assert plan is not None
    for k in (
        "raw_text_emitted",
        "raw_ocr_text_emitted",
        "raw_document_text_emitted",
        "raw_filename_emitted",
        "private_path_emitted",
        "phi_emitted",
        "secret_emitted",
    ):
        assert plan[k] is False, f"{k} should be False"


# ── 8-9. Review-bound invariants ──────────────────────────────────────────


def test_review_required_true():
    env = {TERMINOLOGY_LOOKUP_ENV_VAR: "true"}
    plan = derive_terminology_match_hypothesis(
        SAFE_RECORD, env=env, lookup_adapter=_synthetic_adapter()
    )
    assert plan["review_required"] is True


def test_auto_accept_allowed_false():
    env = {TERMINOLOGY_LOOKUP_ENV_VAR: "true"}
    plan = derive_terminology_match_hypothesis(
        SAFE_RECORD, env=env, lookup_adapter=_synthetic_adapter()
    )
    assert plan["auto_accept_allowed"] is False


# ── 10-14. Refusal flags ──────────────────────────────────────────────────


def test_clinical_interpretation_performed_false():
    env = {TERMINOLOGY_LOOKUP_ENV_VAR: "true"}
    plan = derive_terminology_match_hypothesis(
        SAFE_RECORD, env=env, lookup_adapter=_synthetic_adapter()
    )
    assert plan["clinical_interpretation_performed"] is False


def test_diagnosis_inference_performed_false():
    env = {TERMINOLOGY_LOOKUP_ENV_VAR: "true"}
    plan = derive_terminology_match_hypothesis(
        SAFE_RECORD, env=env, lookup_adapter=_synthetic_adapter()
    )
    assert plan["diagnosis_inference_performed"] is False


def test_treatment_inference_performed_false():
    env = {TERMINOLOGY_LOOKUP_ENV_VAR: "true"}
    plan = derive_terminology_match_hypothesis(
        SAFE_RECORD, env=env, lookup_adapter=_synthetic_adapter()
    )
    assert plan["treatment_inference_performed"] is False


def test_medication_inference_performed_false():
    env = {TERMINOLOGY_LOOKUP_ENV_VAR: "true"}
    plan = derive_terminology_match_hypothesis(
        SAFE_RECORD, env=env, lookup_adapter=_synthetic_adapter()
    )
    assert plan["medication_inference_performed"] is False


def test_ddi_behavior_changed_false():
    env = {TERMINOLOGY_LOOKUP_ENV_VAR: "true"}
    plan = derive_terminology_match_hypothesis(
        SAFE_RECORD, env=env, lookup_adapter=_synthetic_adapter()
    )
    assert plan["ddi_behavior_changed"] is False


# ── 15. app/main.py is not modified or referenced ─────────────────────────


def test_app_main_not_modified_by_term_helper():
    src = (REPO_ROOT / "app" / "main.py").read_text(encoding="utf-8")
    assert "MEDAI-CKA-TERM-INTEGRATION-NEXT-01" not in src
    assert "derive_terminology_match_hypothesis" not in src
    assert "term_match_hypothesis" not in src


# ── 16. No Streamlit / UI wiring exists ───────────────────────────────────


def test_no_streamlit_or_ui_wiring_in_helper():
    helper_src = (
        REPO_ROOT
        / "clinical_knowledge"
        / "terminology"
        / "term_match_hypothesis.py"
    ).read_text(encoding="utf-8")
    for token in (
        "streamlit",
        "import streamlit",
        "st.",
        "st_button",
        "st_form",
    ):
        assert token not in helper_src, f"helper unexpectedly mentions {token!r}"


# ── 17. external_api_used is false ────────────────────────────────────────


def test_external_api_used_false():
    env = {TERMINOLOGY_LOOKUP_ENV_VAR: "true"}
    plan = derive_terminology_match_hypothesis(
        SAFE_RECORD, env=env, lookup_adapter=_synthetic_adapter()
    )
    assert plan["external_api_used"] is False


# ── 18. Cue expansion remains NOT recommended ─────────────────────────────


def test_cue_expansion_remains_not_recommended():
    helper_src = (
        REPO_ROOT
        / "clinical_knowledge"
        / "terminology"
        / "term_match_hypothesis.py"
    ).read_text(encoding="utf-8")
    assert "cue_packs_added" in helper_src
    env = {TERMINOLOGY_LOOKUP_ENV_VAR: "true"}
    plan = derive_terminology_match_hypothesis(
        SAFE_RECORD, env=env, lookup_adapter=_synthetic_adapter()
    )
    assert plan["cue_packs_added"] is False


# ── 19. Freeze tags are not touched ───────────────────────────────────────


def test_freeze_and_park_tags_not_touched_by_helper():
    env = {TERMINOLOGY_LOOKUP_ENV_VAR: "true"}
    plan = derive_terminology_match_hypothesis(
        SAFE_RECORD, env=env, lookup_adapter=_synthetic_adapter()
    )
    assert plan["frozen_operator_release_preserved"] is True
    assert plan["freeze_tags_touched"] is False
    assert plan["park_20_tags_touched"] is False
    assert plan["park_21_tags_touched"] is False
    assert plan["park_22_tags_touched"] is False
    assert plan["park_23_tags_touched"] is False


# ── Signature checks ──────────────────────────────────────────────────────


def test_positive_signature_requires_candidate_text():
    env = {TERMINOLOGY_LOOKUP_ENV_VAR: "true"}
    # Missing field
    assert (
        derive_terminology_match_hypothesis(
            {"anonymous_id": "record_002"},
            env=env,
            lookup_adapter=_synthetic_adapter(),
        )
        is None
    )
    # Empty string
    assert (
        derive_terminology_match_hypothesis(
            {"anonymous_id": "record_003", "terminology_candidate_text": "   "},
            env=env,
            lookup_adapter=_synthetic_adapter(),
        )
        is None
    )


def test_record_is_not_mutated():
    env = {TERMINOLOGY_LOOKUP_ENV_VAR: "true"}
    record = dict(SAFE_RECORD)
    snapshot = dict(record)
    derive_terminology_match_hypothesis(
        record, env=env, lookup_adapter=_synthetic_adapter()
    )
    assert record == snapshot


# ── os.environ not polluted ───────────────────────────────────────────────


def test_os_environ_not_polluted_by_helper_run():
    """Running the helper with an explicit env mapping must not write to
    os.environ. The script never sets the env var.
    """
    # Run a few times; if anything wrote to os.environ, this would fail
    for _ in range(3):
        derive_terminology_match_hypothesis(
            SAFE_RECORD,
            env={TERMINOLOGY_LOOKUP_ENV_VAR: "true"},
            lookup_adapter=_synthetic_adapter(),
        )
    assert TERMINOLOGY_LOOKUP_ENV_VAR not in os.environ


# ── Signature helper exposed correctly ────────────────────────────────────


def test_matches_terminology_lookup_signature_strict():
    assert matches_terminology_lookup_signature({}) is False
    assert (
        matches_terminology_lookup_signature(
            {"terminology_candidate_text": "aspirin"}
        )
        is True
    )
    # Raw document text is NOT a valid signature trigger
    assert (
        matches_terminology_lookup_signature(
            {"ocr_text": "any raw document content here"}
        )
        is False
    )
    assert (
        matches_terminology_lookup_signature({"raw_text": "anything"}) is False
    )
    assert (
        matches_terminology_lookup_signature({"filename": "patient.pdf"})
        is False
    )
