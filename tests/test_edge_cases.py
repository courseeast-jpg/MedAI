"""
MedAI — Edge-case test suite (Phase 7).

Covers the 18 edge cases required by the hybrid extraction design.

Some edge cases depend on the HybridExtractor (under active development).
Those tests are guarded with ``pytest.importorskip`` and will be skipped
cleanly until the module lands, so the rest of the deduplication /
OCR / conflict test matrix always runs.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

import pytest


# ══════════════════════════════════════════════════════════════════════════
# EDGE CASES 1, 8 — DEDUPLICATION ENGINE (always testable)
# ══════════════════════════════════════════════════════════════════════════

from mkb.deduplication_engine import DeduplicationEngine, DedupResult


@pytest.fixture
def dedup() -> DeduplicationEngine:
    return DeduplicationEngine()


# ── Exact duplicates ────────────────────────────────────────────────────────

def test_exact_duplicate_same_day(dedup):
    f1 = {
        "entity_type": "diagnosis", "entity_name": "hypertension",
        "value": None, "date": date(2024, 3, 1),
    }
    f2 = {
        "entity_type": "diagnosis", "entity_name": "hypertension",
        "value": None, "date": date(2024, 3, 5),  # within 7 days
    }
    assert dedup.find_exact_match(f2, [f1]) is f1


def test_exact_duplicate_outside_window(dedup):
    f1 = {
        "entity_type": "diagnosis", "entity_name": "hypertension",
        "date": date(2024, 3, 1),
    }
    f2 = {
        "entity_type": "diagnosis", "entity_name": "hypertension",
        "date": date(2024, 4, 1),
    }
    assert dedup.find_exact_match(f2, [f1]) is None


# ── EDGE CASE 8: Semantic dedup ─────────────────────────────────────────────

def test_semantic_deduplication_htn_alias(dedup):
    """EDGE CASE 8 — Semantic matching (HTN = hypertension)."""
    f1 = {"entity_type": "diagnosis", "entity_name": "hypertension"}
    f2 = {"entity_type": "diagnosis", "entity_name": "HTN"}
    match = dedup.find_semantic_match(f2, [f1])
    assert match is f1


def test_semantic_dedup_dm_alias(dedup):
    f1 = {"entity_type": "diagnosis", "entity_name": "diabetes mellitus"}
    f2 = {"entity_type": "diagnosis", "entity_name": "DM"}
    assert dedup.find_semantic_match(f2, [f1]) is f1


def test_semantic_dedup_mi_alias(dedup):
    f1 = {"entity_type": "diagnosis", "entity_name": "myocardial infarction"}
    f2 = {"entity_type": "diagnosis", "entity_name": "heart attack"}
    assert dedup.find_semantic_match(f2, [f1]) is f1


def test_semantic_dedup_differs_on_type(dedup):
    f1 = {"entity_type": "medication", "entity_name": "aspirin"}
    f2 = {"entity_type": "diagnosis", "entity_name": "aspirin"}
    assert dedup.find_semantic_match(f2, [f1]) is None


# ── EDGE CASE 8: Time-series ────────────────────────────────────────────────

def test_timeseries_detection_hba1c(dedup):
    """EDGE CASE 8 — Value changes over time."""
    f1 = {
        "entity_type": "test_result", "entity_name": "HbA1c",
        "value": 8.2, "date": date(2024, 1, 1),
    }
    f2 = {
        "entity_type": "test_result", "entity_name": "HbA1c",
        "value": 7.5, "date": date(2024, 4, 1),
    }
    assert dedup.find_timeseries_match(f2, [f1]) is f1


def test_timeseries_detection_weight(dedup):
    history = [
        {"entity_type": "measurement", "entity_name": "weight",
         "value": 90, "date": date(2024, 1, 1)},
        {"entity_type": "measurement", "entity_name": "weight",
         "value": 85, "date": date(2024, 3, 1)},
    ]
    new = {"entity_type": "measurement", "entity_name": "weight",
           "value": 82, "date": date(2024, 5, 1)}
    match = dedup.find_timeseries_match(new, history)
    assert match is not None


# ── EDGE CASE 1: Conflict detection ─────────────────────────────────────────

def test_conflict_detection_implausible_weight(dedup):
    """EDGE CASE 1 — Implausible weight change."""
    f1 = {"entity_type": "measurement", "entity_name": "weight",
          "value": 85, "unit": "kg", "date": date(2024, 3, 15)}
    f2 = {"entity_type": "measurement", "entity_name": "weight",
          "value": 78, "unit": "kg", "date": date(2024, 3, 16)}
    conflict = dedup.detect_conflict(f2, [f1])
    assert conflict is not None
    assert conflict.conflict_type in {"implausible_change", "value_mismatch"}


def test_conflict_detection_same_day_bp(dedup):
    f1 = {"entity_type": "measurement", "entity_name": "blood pressure",
          "value": 120, "unit": "mmHg", "date": date(2024, 3, 15)}
    f2 = {"entity_type": "measurement", "entity_name": "blood pressure",
          "value": 170, "unit": "mmHg", "date": date(2024, 3, 15)}
    conflict = dedup.detect_conflict(f2, [f1])
    assert conflict is not None


def test_conflict_detection_type_mismatch(dedup):
    f1 = {"entity_type": "diagnosis", "entity_name": "type 1 diabetes"}
    f2 = {"entity_type": "diagnosis", "entity_name": "type 2 diabetes"}
    conflict = dedup.detect_conflict(f2, [f1])
    assert conflict is not None
    assert conflict.conflict_type == "type_mismatch"


def test_conflict_detection_drug_interaction(dedup):
    f1 = {"entity_type": "medication", "entity_name": "warfarin"}
    f2 = {"entity_type": "medication", "entity_name": "rivaroxaban"}
    conflict = dedup.detect_conflict(f2, [f1])
    assert conflict is not None
    assert conflict.conflict_type == "drug_interaction"
    assert conflict.severity == "critical"


def test_conflict_detection_temporal_impossibility(dedup):
    diagnosis = {
        "entity_type": "diagnosis", "entity_name": "pneumonia",
        "date": date(2024, 6, 1),
    }
    treatment = {
        "entity_type": "medication", "entity_name": "amoxicillin",
        "date": date(2024, 5, 1),
    }
    conflict = dedup.detect_conflict(treatment, [diagnosis])
    assert conflict is not None
    assert conflict.conflict_type == "temporal"


# ── Implausible change unit tests ───────────────────────────────────────────

def test_is_implausible_change_hba1c(dedup):
    f1 = {"entity_name": "HbA1c", "value": 8.5, "date": date(2024, 1, 1)}
    f2 = {"entity_name": "HbA1c", "value": 5.0, "date": date(2024, 1, 30)}
    assert dedup.is_implausible_change(f1, f2) is True


def test_is_plausible_change_hba1c(dedup):
    f1 = {"entity_name": "HbA1c", "value": 8.0, "date": date(2024, 1, 1)}
    f2 = {"entity_name": "HbA1c", "value": 7.3, "date": date(2024, 4, 1)}
    assert dedup.is_implausible_change(f1, f2) is False


# ── Full orchestration ─────────────────────────────────────────────────────

def test_deduplicate_returns_insert_for_empty(dedup):
    new = {"entity_type": "diagnosis", "entity_name": "asthma",
           "date": date(2024, 4, 1)}
    result = dedup.deduplicate(new, [])
    assert isinstance(result, DedupResult)
    assert result.action == "insert"
    assert result.strategy == "none"


def test_deduplicate_exact_returns_merge(dedup):
    existing = {"entity_type": "diagnosis", "entity_name": "asthma",
                "date": date(2024, 4, 1)}
    new = {"entity_type": "diagnosis", "entity_name": "asthma",
           "date": date(2024, 4, 3)}
    result = dedup.deduplicate(new, [existing])
    assert result.action == "merge"
    assert result.strategy == "exact"


def test_deduplicate_conflict_returns_quarantine(dedup):
    existing = {"entity_type": "medication", "entity_name": "warfarin"}
    new = {"entity_type": "medication", "entity_name": "rivaroxaban"}
    result = dedup.deduplicate(new, [existing])
    assert result.action == "quarantine"
    assert result.strategy == "conflict"


# ══════════════════════════════════════════════════════════════════════════
# PHASE 5 — CONFLICT RESOLVER
# ══════════════════════════════════════════════════════════════════════════

from mkb.conflict_resolver import ConflictResolver


@pytest.fixture
def resolver(tmp_path) -> ConflictResolver:
    return ConflictResolver(db_path=tmp_path / "conflicts.db")


def test_quarantine_stores_conflict(resolver):
    f1 = {"id": "a", "entity_name": "weight", "value": 85}
    f2 = {"id": "b", "entity_name": "weight", "value": 78}
    cid = resolver.quarantine_conflict(f1, f2, "value_mismatch", severity="high",
                                       reason="same-day disparity")
    assert cid
    pending = resolver.list_pending()
    assert len(pending) == 1
    assert pending[0]["conflict_type"] == "value_mismatch"
    assert pending[0]["severity"] == "high"


def test_quarantine_severity_validation(resolver):
    with pytest.raises(ValueError):
        resolver.quarantine_conflict({}, {}, "t", severity="bogus")


def test_resolve_conflict_choose_fact1(resolver):
    cid = resolver.quarantine_conflict(
        {"id": "a", "entity_name": "x"},
        {"id": "b", "entity_name": "y"},
        "value_mismatch",
    )
    resolver.resolve_conflict(cid, {"choice": "fact1", "notes": "chose a"})
    conflict = resolver.get_conflict(cid)
    assert conflict["status"] == "resolved"
    assert conflict["resolution"]["choice"] == "fact1"


def test_resolve_conflict_invalid_choice(resolver):
    cid = resolver.quarantine_conflict(
        {"id": "a"}, {"id": "b"}, "value_mismatch"
    )
    with pytest.raises(ValueError):
        resolver.resolve_conflict(cid, {"choice": "nope"})


def test_list_pending_ordered_by_severity(resolver):
    resolver.quarantine_conflict({"id": "1"}, {"id": "2"}, "t", severity="low")
    resolver.quarantine_conflict({"id": "3"}, {"id": "4"}, "t", severity="critical")
    resolver.quarantine_conflict({"id": "5"}, {"id": "6"}, "t", severity="medium")
    pending = resolver.list_pending()
    assert [c["severity"] for c in pending] == ["critical", "medium", "low"]


def test_count_pending(resolver):
    assert resolver.count_pending() == 0
    resolver.quarantine_conflict({}, {}, "t")
    assert resolver.count_pending() == 1


# ══════════════════════════════════════════════════════════════════════════
# EDGE CASE 12 — OCR VALIDATOR
# ══════════════════════════════════════════════════════════════════════════

from extraction.ocr_validator import OCRValidator


@pytest.fixture
def ocr() -> OCRValidator:
    return OCRValidator()


def test_ocr_error_detection_number_letter_confusion(ocr):
    """EDGE CASE 12 — OCR quality."""
    text = "Patient weight 50Omg prescribed metf0rmin l00mg"
    result = ocr.validate_ocr_quality(text, None)
    assert result["errors_detected"] is True
    assert result["confidence"] < 0.8


def test_ocr_clean_text_high_confidence(ocr):
    text = (
        "Patient diagnosed with hypertension. Prescribed metformin 500mg daily. "
        "Blood pressure measured at 140/90. Follow-up in 3 months."
    )
    result = ocr.validate_ocr_quality(text, None)
    assert result["errors_detected"] is False
    assert result["confidence"] >= 0.9


def test_ocr_missing_spaces(ocr):
    text = "hypertensiondiabetesmellitusasthmabronchitispneumonia"
    result = ocr.validate_ocr_quality(text, None)
    assert result["errors_detected"] is True
    assert result["checks"]["missing_spaces"]["failed"] is True


def test_ocr_special_chars(ocr):
    text = "@#$% ^&*( !@# $%^ &*( @#$ @#$%^&*()!@#$%^&*()"
    result = ocr.validate_ocr_quality(text, None)
    assert result["errors_detected"] is True
    assert result["checks"]["special_characters"]["failed"] is True


def test_ocr_empty_text(ocr):
    result = ocr.validate_ocr_quality("", None)
    assert result["confidence"] == 0.0
    assert result["errors_detected"] is True


# ══════════════════════════════════════════════════════════════════════════
# HYBRID EXTRACTOR EDGE CASES (2, 3, 4A, 4B, 6)
# Guarded — run only when extraction.hybrid_extractor lands.
# ══════════════════════════════════════════════════════════════════════════

def _hybrid_extractor_or_skip():
    mod = pytest.importorskip(
        "extraction.hybrid_extractor",
        reason="HybridExtractor not yet implemented",
    )
    return mod.HybridExtractor()


def test_negation_detection():
    """EDGE CASE 3 — negation detection."""
    extractor = _hybrid_extractor_or_skip()

    text = "Patient denies chest pain"
    is_neg, neg_type = extractor.detect_negation("chest pain", text)
    assert is_neg is True
    assert neg_type == "denied"

    text = "Pneumonia ruled out"
    is_neg, neg_type = extractor.detect_negation("pneumonia", text)
    assert is_neg is True
    assert neg_type == "ruled_out"

    text = "No history of diabetes"
    is_neg, neg_type = extractor.detect_negation("diabetes", text)
    assert is_neg is True
    assert neg_type == "absent"


def test_family_history_detection():
    """EDGE CASE 4A — family-history vs patient."""
    extractor = _hybrid_extractor_or_skip()

    subject, conf = extractor.identify_subject(
        "heart attack", "Father had heart attack at age 55"
    )
    assert subject == "father"
    assert conf > 0.8

    subject, _ = extractor.identify_subject(
        "breast cancer", "Mother's breast cancer diagnosis"
    )
    assert subject == "mother"

    subject, _ = extractor.identify_subject(
        "hypertension", "Patient has hypertension"
    )
    assert subject == "patient"


def test_certainty_assessment():
    """EDGE CASE 4B — certainty levels."""
    extractor = _hybrid_extractor_or_skip()

    cert, _ = extractor.assess_certainty("diabetes", "Diagnosed with type 2 diabetes")
    assert cert == "confirmed"

    cert, _ = extractor.assess_certainty("pneumonia", "Suspect pneumonia based on symptoms")
    assert cert == "suspected"

    cert, _ = extractor.assess_certainty("appendicitis", "Consider appendicitis vs gastritis")
    assert cert == "differential"


def test_unit_normalization():
    """EDGE CASE 6 — unit conversion."""
    extractor = _hybrid_extractor_or_skip()

    norm_val, norm_unit = extractor.normalize_measurement(185, "lbs", "weight")
    assert norm_unit == "kg"
    assert 83 < norm_val < 85

    norm_val, norm_unit = extractor.normalize_measurement(55, "mmol/mol", "HbA1c")
    assert norm_unit == "%"
    assert 7.0 < norm_val < 7.5


def test_temporal_extraction():
    """EDGE CASE 2 — temporal handling."""
    extractor = _hybrid_extractor_or_skip()
    doc_date = date(2024, 4, 1)

    event_date, conf, _ = extractor.extract_temporal_info("diagnosed on 2024-03-15", doc_date)
    assert event_date == date(2024, 3, 15)
    assert conf == 1.0

    event_date, conf, _ = extractor.extract_temporal_info("hospitalized 2 weeks ago", doc_date)
    assert event_date == date(2024, 3, 18)
    assert conf > 0.7
