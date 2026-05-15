from __future__ import annotations

from pathlib import Path

from app.review_package_viewer import (
    bucket_operator_copy,
    display_bucket_name,
    get_bucket_summary,
    load_review_package,
    review_status_summary,
)


APP_VIEWER = Path(__file__).resolve().parents[1] / "app" / "review_package_viewer.py"


def viewer_source() -> str:
    return APP_VIEWER.read_text(encoding="utf-8")


def test_review_package_operator_labels_are_present() -> None:
    source = viewer_source()

    for text in (
        "Review Package",
        "Review items and system safety findings.",
        "Review status: {summary['review_status']}",
        "scan-quality items need attention when convenient",
        "No production changes recommended.",
        "Review summary",
        "Issue type",
        "Items",
        "Needs action?",
        "System change?",
        "Scan quality review",
        "No text found",
        "Unknown document type",
        "Possible combined PDF",
        "Unsupported format",
        "Completed review paths",
        "Review details",
        "Example safe file IDs",
        "These are internal safe IDs, not patient names or real filenames.",
        "Advanced: full audit report",
        "Build / audit details",
    ):
        assert text in source


def test_old_primary_labels_are_not_in_review_viewer() -> None:
    source = viewer_source()

    for text in (
        "Review Package \u2014 Phase 74 Auto-Improvement",
        "Phase74 conclusion",
        "Prod. change allowed?",
        "Operator required?",
        "Bucket Details",
        "View full safe review package (Markdown)",
    ):
        assert text not in source


def test_bucket_label_mapping_preserves_raw_bucket_ids() -> None:
    package = load_review_package()
    buckets = get_bucket_summary(package)
    by_id = {bucket["bucket_id"]: bucket for bucket in buckets}

    assert display_bucket_name(by_id["ocr_quality_review"]) == "Scan quality review"
    assert display_bucket_name(by_id["empty_extraction_review"]) == "No text found"
    assert display_bucket_name(by_id["unknown_document_class_review"]) == "Unknown document type"
    assert display_bucket_name(by_id["possible_multi_document_pdf_review"]) == "Possible combined PDF"
    assert display_bucket_name(by_id["unsupported_or_deferred_format_review"]) == "Unsupported format"
    assert display_bucket_name(by_id["completed_manual_boundary_branches"]) == "Completed review paths"
    assert by_id["ocr_quality_review"]["bucket_name"] == "OCR Quality Review"


def test_operator_copy_for_core_buckets_is_plain_language() -> None:
    package = load_review_package()
    buckets = {bucket["bucket_id"]: bucket for bucket in get_bucket_summary(package)}

    scan = bucket_operator_copy(buckets["ocr_quality_review"])
    assert scan["why"] == "Some files may have weak scan quality. MedAI may miss or misread values."
    assert scan["knows"] == "The scan quality was below the safe threshold."
    assert scan["unknown"] == "Whether the missed text is clinically important."
    assert scan["next"] == "Review these files later or upload clearer copies. Do not lower the quality threshold."

    empty = bucket_operator_copy(buckets["empty_extraction_review"])
    assert empty["why"] == "MedAI could not find useful text to process."
    assert empty["next"] == "Check the source file and upload a clearer or correct copy if needed."


def test_review_status_summary_uses_existing_counts() -> None:
    package = load_review_package()
    buckets = get_bucket_summary(package)
    summary = review_status_summary(buckets)

    assert summary["review_status"] == "No blocking review required"
    assert summary["scan_quality_attention_count"] == 12
    assert summary["total_review_items"] == 1489
    assert summary["review_categories"] == 6
    assert summary["production_change_recommended"] is False


def test_review_package_helpers_do_not_expose_private_values() -> None:
    package = load_review_package()
    text = str(get_bucket_summary(package)) + str(review_status_summary(get_bucket_summary(package)))

    assert "Patient Jane Doe" not in text
    assert "full_corpus_input" not in text
    assert "local_filename_mapping_PRIVATE" not in text
    assert "original_relative_path" not in text
    assert "ocr_text" not in text
