from __future__ import annotations

from execution.confidence_scorer import score_extraction_result
from execution.consensus import consensus_merge
from execution.connectors.phi3_connector import Phi3Connector
from execution.validation import validate_extraction_result
from extractors.spacy_extractor import SpacyExtractor


def test_phi3_with_zero_entities_remains_review_bound():
    result = score_extraction_result({
        "extractor": "phi3",
        "actual_extractor": "phi3",
        "entities": [],
        "raw_text": "unreadable fragment",
        "confidence": 0.0,
        "latency_ms": 1,
        "notes": [],
    })

    decision = validate_extraction_result(result, extractor_route="phi3")

    assert decision.status == "rejected"
    assert result["confidence_breakdown"]["base_extractor_weight"] == 0.6
    assert result["confidence_breakdown"]["calibrated_extractor_weight"] == 0.6
    assert result["confidence_breakdown"]["calibration_reason"] == "none"


def test_phi3_with_one_entity_remains_review_bound():
    result = score_extraction_result({
        "extractor": "phi3",
        "actual_extractor": "phi3",
        "entities": [{"type": "test_result", "text": "RBC"}],
        "raw_text": "RBC present",
        "confidence": 0.0,
        "latency_ms": 1,
        "notes": [],
    })

    decision = validate_extraction_result(result, extractor_route="phi3")

    assert decision.status == "needs_review"
    assert result["confidence"] < 0.65
    assert result["confidence_breakdown"]["calibrated_extractor_weight"] == 0.6


def test_phi3_with_valid_supplemental_entities_gets_calibrated_confidence():
    result = Phi3Connector().extract("UA BLOOD positive RBC 10-20 /hpf calcium oxalate crystals present")

    assert result["supplemental_rules_applied"] is True
    assert result["supplemental_entity_count"] >= 1
    assert result["confidence"] >= 0.65
    assert result["confidence_breakdown"]["base_extractor_weight"] == 0.6
    assert result["confidence_breakdown"]["calibrated_extractor_weight"] == 1.0
    assert result["confidence_breakdown"]["calibration_reason"] == "phi3_supplemental_entities"


def test_phi3_normalizes_noisy_lab_text_before_supplemental_rules():
    result = Phi3Connector().extract("UR0KULTURE |||| NEGAT1V ____ VERDHE")
    entity_texts = {entity["text"] for entity in result["entities"]}

    assert result["normalization_applied"] is True
    assert "Urine Culture" in result["normalized_text_preview"]
    assert "Negative" in result["normalized_text_preview"]
    assert "Yellow" in result["normalized_text_preview"]
    assert "Urine Culture" in entity_texts


def test_consensus_recomputes_stale_phi3_confidence_when_calibration_applies():
    merged = consensus_merge([
        {
            "extractor": "phi3",
            "actual_extractor": "phi3",
            "entities": [
                {"type": "test_result", "text": "Blood"},
                {"type": "test_result", "text": "RBC"},
                {"type": "test_result", "text": "Calcium Oxalate Crystals"},
            ],
            "confidence": 0.45,
            "latency_ms": 1,
            "raw_text": "Blood RBC Calcium Oxalate Crystals",
            "notes": [],
            "supplemental_rules_applied": True,
            "supplemental_entity_count": 3,
            "final_entity_count_after_supplement": 3,
        }
    ], extractor_route="phi3")

    assert merged["confidence"] > 0.45
    assert merged["confidence"] >= 0.65
    assert merged["confidence_breakdown"]["calibrated_extractor_weight"] == 1.0


def test_spacy_confidence_behavior_is_unchanged():
    result = SpacyExtractor().extract("Patient has diabetes. Takes metformin. No hypertension.")

    assert result["confidence_breakdown"]["extractor_weight"] == 0.8
    assert result["confidence_breakdown"]["base_extractor_weight"] == 0.8
    assert result["confidence_breakdown"]["calibrated_extractor_weight"] == 0.8
    assert result["confidence_breakdown"]["calibration_reason"] == "none"
