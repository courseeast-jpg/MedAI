from __future__ import annotations

from pathlib import Path

from scripts.run_phase12_real_world_validation import (
    build_phase12_summary,
    is_external_quota_error,
    summarize_document,
)


def test_build_phase12_summary_aggregates_documents():
    documents = [
        {
            "document": "short_01.pdf",
            "status": "processed",
            "error": None,
            "outcome": "written",
            "validation_status": "accepted",
            "extractor": "spacy",
            "extractor_actual": "spacy",
            "confidence": 0.7,
            "entity_count": 2,
            "written_count": 2,
            "queued_count": 0,
            "blocked_count": 0,
            "review_reasons": [],
            "notes": [],
        },
        {
            "document": "short_02.pdf",
            "status": "processed",
            "error": None,
            "outcome": "queued_for_review",
            "validation_status": "needs_review",
            "extractor": "gemini",
            "extractor_actual": "rules_based",
            "confidence": 0.45,
            "entity_count": 1,
            "written_count": 0,
            "queued_count": 1,
            "blocked_count": 0,
            "review_reasons": ["pending_validation_review"],
            "notes": [],
        },
        {
            "document": "short_03.pdf",
            "status": "external_quota_blocked",
            "error": "429 quota exceeded for metric generate_content",
            "outcome": "external_quota_blocked",
            "validation_status": "skipped_external_quota",
            "extractor": None,
            "extractor_actual": None,
            "confidence": 0.0,
            "entity_count": 0,
            "written_count": 0,
            "queued_count": 0,
            "blocked_count": 0,
            "review_reasons": [],
            "notes": [],
        },
        {
            "document": "short_03.pdf",
            "status": "error",
            "error": "boom",
            "outcome": "error",
            "validation_status": "error",
            "extractor": None,
            "extractor_actual": None,
            "confidence": 0.0,
            "entity_count": 0,
            "written_count": 0,
            "queued_count": 0,
            "blocked_count": 0,
            "review_reasons": [],
            "notes": [],
        },
    ]

    summary = build_phase12_summary(
        dataset_dir=Path("test_data/final_batch_50"),
        requested_limit=10,
        documents=documents,
        runtime_counts={"total": 3, "active": 2, "hypothesis": 0, "quarantined": 1},
        component_state={"governance_active": True},
    )

    assert summary["documents_processed"] == 2
    assert summary["written"] == 1
    assert summary["queued_for_review"] == 1
    assert summary["external_quota_blocked"] == 1
    assert summary["hard_failures"] == 1
    assert summary["documents_failed"] == 1
    assert summary["run_passed"] is False
    assert summary["aggregate"]["outcomes"] == {"queued_for_review": 1, "written": 1}
    assert summary["aggregate"]["validation_statuses"] == {"accepted": 1, "needs_review": 1}
    assert summary["aggregate"]["extractors"] == {"rules_based": 1, "spacy": 1}
    assert summary["aggregate"]["avg_confidence"] == 0.575
    assert summary["aggregate"]["total_entities"] == 3
    assert any("at least 10 documents" in item for item in summary["recommendations"])
    assert any("processing errors" in item for item in summary["recommendations"])
    assert any("quota exhaustion" in item for item in summary["recommendations"])


def test_quota_safe_mode_classifies_quota_exhaustion_as_external_block():
    document = summarize_document(
        Path("quota_blocked.pdf"),
        None,
        error=RuntimeError(
            "Gemini route fallback occurred despite configured key: rules_based; "
            "root_cause=429 You exceeded your current quota, please retry in 36 seconds."
        ),
        quota_safe=True,
    )

    assert is_external_quota_error(document["error"]) is True
    assert document["status"] == "external_quota_blocked"
    assert document["outcome"] == "external_quota_blocked"
    assert document["validation_status"] == "skipped_external_quota"
    assert "429" in document["error"]
    assert "route_actual_mismatch" not in document["review_reasons"]


def test_quota_safe_mode_does_not_hide_normal_hard_failures():
    document = summarize_document(
        Path("hard_failure.pdf"),
        None,
        error=RuntimeError("corrupt PDF header"),
        quota_safe=True,
    )

    assert is_external_quota_error(document["error"]) is False
    assert document["status"] == "error"
    assert document["outcome"] == "error"
    assert document["validation_status"] == "error"
