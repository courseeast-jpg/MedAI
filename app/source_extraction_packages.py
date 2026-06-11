"""Source-facing extraction package model for review-bound MKB records.

The package layer groups already-persisted extracted facts for operator review.
It does not parse source text, change MKB safety semantics, or auto-accept.
Public package dictionaries never include raw OCR text, filenames, paths, or
private source names; source identity is represented by generated safe IDs.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Iterable

from app.mkb_explorer_model import display_content_public_safe, safe_record_id
from app.operator_review_actions import (
    ACCEPT_DISCLAIMER,
    DEFER_DISCLAIMER,
    REJECT_DISCLAIMER,
    accept_after_source_comparison,
    defer_extracted_fact,
    reject_extracted_fact,
    render_action_affordances_plan,
)
from app.schemas import MKBRecord
from app.specialty_selection import specialty_label, validate_specialty_key


@dataclass
class SourceObservation:
    record_id: str
    record_id_full: str
    fact_type: str
    label: str
    value: str = ""
    flag: str = ""
    unit: str = ""
    reference_interval: str = ""
    source_section: str = "Unsectioned"
    review_status: str = "review-bound"
    display_content: str = ""


@dataclass
class SourcePackageSection:
    section_id: str
    heading: str
    observation_family: str
    observations: list[SourceObservation] = field(default_factory=list)
    narrative_source_section: str = ""
    narrative_preview_available: bool = False
    narrative_label: str = "Source text excerpt; not MedAI interpretation"


@dataclass
class SourceExtractionPackage:
    package_id: str
    safe_source_document_id: str
    selected_document_category: str
    selected_medical_specialty_domain: str
    selected_medical_specialty_label: str
    detected_document_family_type: str
    source_modality: str
    package_status: str
    record_ids: list[str] = field(default_factory=list)
    sections: list[SourcePackageSection] = field(default_factory=list)
    actions: dict[str, Any] = field(default_factory=dict)
    auto_accept_allowed: bool = False
    review_required: bool = True


def build_source_extraction_packages(
    sql_store: Any,
    *,
    tier_filter: str = "review_bound",
    limit: int = 200,
) -> dict[str, Any]:
    """Build source-facing packages from persisted MKB records."""
    records = _load_records(sql_store, tier_filter=tier_filter, limit=limit)
    packages_by_key: dict[str, SourceExtractionPackage] = {}
    ungrouped_records_count = 0
    for record in records:
        structured = dict(record.structured or {})
        if not _is_package_candidate(record, structured):
            ungrouped_records_count += 1
            continue
        source_key = _source_group_key(record, structured)
        package = packages_by_key.get(source_key)
        if package is None:
            package = SourceExtractionPackage(
                package_id=f"pkg_{_stable_hash(source_key)[:12]}",
                safe_source_document_id=f"source_{_stable_hash(source_key)[:12]}",
                selected_document_category=str(structured.get("document_category") or "Unspecified"),
                selected_medical_specialty_domain=validate_specialty_key(record.specialty),
                selected_medical_specialty_label=specialty_label(record.specialty),
                detected_document_family_type=str(
                    structured.get("document_family")
                    or structured.get("document_type")
                    or structured.get("source_document_type")
                    or "Unknown"
                ),
                source_modality=_source_modality_label(str(structured.get("source_modality") or record.source_type or "")),
                package_status=_package_status_for_records([record]),
                actions=package_action_plan([]),
            )
            packages_by_key[source_key] = package
        package.record_ids.append(record.id)
        package.package_status = _package_status_for_records([*(_records_for_ids(sql_store, package.record_ids) or []), record])
        _append_observation(package, record, structured)

    packages = list(packages_by_key.values())
    for package in packages:
        package.actions = package_action_plan(package.record_ids)

    sections_created = sum(len(package.sections) for package in packages)
    observations_grouped = sum(len(section.observations) for package in packages for section in package.sections)
    return {
        "available": sql_store is not None,
        "packages": [package_to_public_dict(package) for package in packages],
        "packages_created": len(packages),
        "sections_created": sections_created,
        "observations_grouped": observations_grouped,
        "ungrouped_records_count": ungrouped_records_count,
        "auto_accept": False,
        "external_api_used": False,
        "atomic_review_fallback_preserved": True,
    }


def package_to_public_dict(package: SourceExtractionPackage) -> dict[str, Any]:
    return {
        "package_id": package.package_id,
        "safe_source_document_id": package.safe_source_document_id,
        "selected_document_category": package.selected_document_category,
        "selected_medical_specialty_domain": package.selected_medical_specialty_domain,
        "selected_medical_specialty_label": package.selected_medical_specialty_label,
        "detected_document_family_type": package.detected_document_family_type,
        "source_modality": package.source_modality,
        "package_status": package.package_status,
        "record_ids": list(package.record_ids),
        "record_count": len(package.record_ids),
        "sections": [
            {
                "section_id": section.section_id,
                "heading": section.heading,
                "observation_family": section.observation_family,
                "narrative_source_section": section.narrative_source_section,
                "narrative_preview_available": section.narrative_preview_available,
                "narrative_label": section.narrative_label,
                "observations": [
                    {
                        "record_id": obs.record_id,
                        "record_id_full": obs.record_id_full,
                        "fact_type": obs.fact_type,
                        "label": obs.label,
                        "value": obs.value,
                        "flag": obs.flag,
                        "unit": obs.unit,
                        "reference_interval": obs.reference_interval,
                        "source_section": obs.source_section,
                        "review_status": obs.review_status,
                        "display_content": obs.display_content,
                    }
                    for obs in section.observations
                ],
            }
            for section in package.sections
        ],
        "actions": package.actions,
        "auto_accept_allowed": False,
        "review_required": True,
    }


def package_action_plan(record_ids: Iterable[str]) -> dict[str, Any]:
    record_ids = [str(record_id) for record_id in record_ids if str(record_id or "")]
    return {
        "record_count": len(record_ids),
        "actions": [
            {
                "key": "accept_package_after_source_comparison",
                "label": "Accept package after source comparison",
                "enabled": bool(record_ids),
                "disclaimer": ACCEPT_DISCLAIMER,
            },
            {
                "key": "reject_package",
                "label": "Reject package",
                "enabled": bool(record_ids),
                "disclaimer": REJECT_DISCLAIMER,
            },
            {
                "key": "defer_package",
                "label": "Defer package",
                "enabled": bool(record_ids),
                "disclaimer": DEFER_DISCLAIMER,
            },
        ],
        "auto_accept_allowed": False,
        "review_required": True,
    }


def accept_package_after_source_comparison(sql_store: Any, record_ids: Iterable[str], *, session_id: str = "") -> dict[str, Any]:
    return _apply_package_action(
        "accept_package_after_source_comparison",
        sql_store,
        record_ids,
        lambda record_id: accept_after_source_comparison(sql_store, record_id, session_id=session_id),
    )


def reject_package(sql_store: Any, record_ids: Iterable[str], *, session_id: str = "") -> dict[str, Any]:
    return _apply_package_action(
        "reject_package",
        sql_store,
        record_ids,
        lambda record_id: reject_extracted_fact(sql_store, record_id, session_id=session_id),
    )


def defer_package(sql_store: Any, record_ids: Iterable[str], *, session_id: str = "") -> dict[str, Any]:
    return _apply_package_action(
        "defer_package",
        sql_store,
        record_ids,
        lambda record_id: defer_extracted_fact(sql_store, record_id, session_id=session_id),
    )


def _apply_package_action(action: str, sql_store: Any, record_ids: Iterable[str], fn: Any) -> dict[str, Any]:
    del sql_store
    ids = [str(record_id) for record_id in record_ids if str(record_id or "")]
    results = [fn(record_id).to_public_dict() for record_id in ids]
    return {
        "success": bool(results) and all(result.get("success") for result in results),
        "action": action,
        "record_count": len(ids),
        "success_count": sum(1 for result in results if result.get("success")),
        "failure_count": sum(1 for result in results if not result.get("success")),
        "auto_accept_allowed": False,
        "review_required_before_action": True,
        "safe_message": "Package action applied to review-bound records only. Human source comparison remains required.",
        "record_results": results,
    }


def _load_records(sql_store: Any, *, tier_filter: str, limit: int) -> list[MKBRecord]:
    if sql_store is None:
        return []
    clauses: list[str] = []
    params: list[Any] = []
    if tier_filter == "review_bound":
        clauses.append("requires_review=1")
    elif tier_filter and tier_filter != "all":
        clauses.append("tier=?")
        params.append(tier_filter)
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    with sql_store._get_conn() as conn:
        rows = conn.execute(f"SELECT * FROM records{where} ORDER BY first_recorded DESC LIMIT ?", [*params, int(limit)]).fetchall()
    return [sql_store._row_to_record(row) for row in rows]


def _records_for_ids(sql_store: Any, record_ids: Iterable[str]) -> list[MKBRecord]:
    records: list[MKBRecord] = []
    if sql_store is None:
        return records
    for record_id in record_ids:
        record = sql_store.get_record(record_id)
        if record is not None:
            records.append(record)
    return records


def _is_package_candidate(record: MKBRecord, structured: dict[str, Any]) -> bool:
    if not (record.requires_review or record.tier == "quarantined"):
        return False
    if record.fact_type not in {"test_result", "observation", "note"}:
        return False
    return bool(structured.get("source_visible_observation") or structured.get("parser_name") or structured.get("section_heading"))


def _source_group_key(record: MKBRecord, structured: dict[str, Any]) -> str:
    return "|".join(
        [
            str(record.session_id or "sessionless"),
            str(structured.get("source_document_id") or record.source_name or "source"),
            str(structured.get("source_modality") or record.source_type or "unknown"),
        ]
    )


def _append_observation(package: SourceExtractionPackage, record: MKBRecord, structured: dict[str, Any]) -> None:
    heading = str(structured.get("section_heading") or structured.get("source_section") or structured.get("candidate_kind") or record.fact_type or "Unsectioned")
    family = str(structured.get("candidate_kind") or record.fact_type or "observation")
    section_id = f"section_{_stable_hash(package.package_id + '|' + heading + '|' + family)[:10]}"
    section = next((item for item in package.sections if item.section_id == section_id), None)
    if section is None:
        narrative_available = record.fact_type == "note" or family == "narrative_source_section"
        section = SourcePackageSection(
            section_id=section_id,
            heading=heading,
            observation_family=family,
            narrative_source_section=heading if narrative_available else "",
            narrative_preview_available=narrative_available,
        )
        package.sections.append(section)
    section.observations.append(_observation_from_record(record, structured, heading))


def _observation_from_record(record: MKBRecord, structured: dict[str, Any], heading: str) -> SourceObservation:
    review_status = str(structured.get("operator_review_status") or record.status or "review-bound")
    if record.requires_review or record.tier == "quarantined":
        review_status = "review-bound" if review_status in {"pending_validation_review", "queued_for_review", "active"} else review_status
    label = str(
        structured.get("test_name")
        or structured.get("field_label")
        or structured.get("section_heading")
        or structured.get("name")
        or display_content_public_safe(record)
    )
    return SourceObservation(
        record_id=safe_record_id(record.id),
        record_id_full=record.id,
        fact_type=record.fact_type,
        label=label,
        value=str(structured.get("value") or structured.get("field_value") or ""),
        flag=str(structured.get("flag") or ""),
        unit=str(structured.get("unit") or ""),
        reference_interval=str(structured.get("reference_range") or structured.get("normal_range") or ""),
        source_section=heading,
        review_status=review_status,
        display_content=display_content_public_safe(record),
    )


def _package_status_for_records(records: list[MKBRecord]) -> str:
    if not records:
        return "review-bound"
    if all(record.tier == "active" and not record.requires_review for record in records):
        return "accepted"
    if all(record.tier == "superseded" for record in records):
        return "rejected"
    if any(record.status == "deferred_by_operator" for record in records):
        return "deferred"
    return "review-bound"


def _source_modality_label(value: str) -> str:
    value = (value or "").strip()
    if value == "image_ocr":
        return "local OCR"
    if value:
        return value.replace("_", " ")
    return "unknown"


def _stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="ignore")).hexdigest()


__all__ = [
    "SourceExtractionPackage",
    "SourcePackageSection",
    "SourceObservation",
    "build_source_extraction_packages",
    "package_action_plan",
    "accept_package_after_source_comparison",
    "reject_package",
    "defer_package",
]
