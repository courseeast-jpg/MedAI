"""Source-facing extraction package model for review-bound MKB records.

The package layer groups already-persisted extracted facts for operator review.
It does not parse source text, change MKB safety semantics, or auto-accept.
Public package dictionaries never include raw OCR text, filenames, paths, or
private source names; source identity is represented by generated safe IDs.
"""
from __future__ import annotations

import hashlib
import re
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
    row_kind: str = "observation"
    normalization_status: str = "unchanged"


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
    malformed_value_pair_count = 0
    normalized_value_pair_count = 0
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
                detected_document_family_type=_display_document_family(structured),
                source_modality=_source_modality_label(str(structured.get("source_modality") or record.source_type or "")),
                package_status=_package_status_for_records([record]),
                actions=package_action_plan([]),
            )
            packages_by_key[source_key] = package
        package.record_ids.append(record.id)
        package.package_status = _package_status_for_records([*(_records_for_ids(sql_store, package.record_ids) or []), record])
        normalization_status = _append_observation(package, record, structured)
        if normalization_status == "normalized":
            normalized_value_pair_count += 1
        elif normalization_status == "malformed_note":
            malformed_value_pair_count += 1

    packages = list(packages_by_key.values())
    for package in packages:
        package.actions = package_action_plan(package.record_ids)

    sections_created = sum(len(package.sections) for package in packages)
    observations_grouped = sum(len(section.observations) for package in packages for section in package.sections)
    unknown_type_package_count = sum(
        1 for package in packages if package.detected_document_family_type.strip().lower() in {"", "unknown"}
    )
    return {
        "available": sql_store is not None,
        "packages": [package_to_public_dict(package) for package in packages],
        "packages_created": len(packages),
        "sections_created": sections_created,
        "observations_grouped": observations_grouped,
        "ungrouped_records_count": ungrouped_records_count,
        "unknown_type_package_count": unknown_type_package_count,
        "malformed_value_pair_count": malformed_value_pair_count,
        "normalized_value_pair_count": normalized_value_pair_count,
        "package_action_color_semantics_present": True,
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
                        "row_kind": obs.row_kind,
                        "normalization_status": obs.normalization_status,
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


def source_package_from_ai_draft(draft: Any) -> dict[str, Any]:
    """Convert an AI-shaped package draft into the package-first UI shape.

    The bridge is intentionally in-memory. It does not write active MKB records
    and all generated observation IDs are synthetic package-local IDs.
    """
    source_id = str(getattr(draft, "safe_source_document_id", "") or "source_fake_001")
    package_id = f"ai_pkg_{_stable_hash(source_id + str(getattr(draft, 'document_type', '')))[:12]}"
    sections: list[dict[str, Any]] = []
    record_ids: list[str] = []
    for section_index, section in enumerate(list(getattr(draft, "sections", []) or []), start=1):
        heading = str(getattr(section, "heading", "") or f"Section {section_index}")
        observations: list[dict[str, Any]] = []
        for obs_index, obs in enumerate(list(getattr(section, "observations", []) or []), start=1):
            record_id = f"{package_id}_obs_{section_index:02d}_{obs_index:02d}"
            record_ids.append(record_id)
            normalized = normalize_package_observation_fields(
                label=getattr(obs, "label", ""),
                value=getattr(obs, "value", ""),
                reference_interval=getattr(obs, "reference_interval", ""),
                flag=getattr(obs, "flag", ""),
                unit=getattr(obs, "unit", ""),
            )
            observations.append(
                {
                    "record_id": safe_record_id(record_id),
                    "record_id_full": record_id,
                    "fact_type": "test_result" if normalized["row_kind"] == "observation" else "note",
                    "label": normalized["label"],
                    "value": normalized["value"],
                    "flag": normalized["flag"],
                    "unit": normalized["unit"],
                    "reference_interval": normalized["reference_interval"],
                    "source_section": heading,
                    "review_status": "review-bound",
                    "display_content": normalized["label"],
                    "row_kind": normalized["row_kind"],
                    "normalization_status": normalized["normalization_status"],
                }
            )
        sections.append(
            {
                "section_id": f"ai_section_{_stable_hash(package_id + heading)[:10]}",
                "heading": heading,
                "observation_family": "ai_assisted_source_extraction",
                "narrative_source_section": heading if getattr(section, "narrative_preview", "") else "",
                "narrative_preview_available": bool(getattr(section, "narrative_preview", "")),
                "narrative_label": str(
                    getattr(section, "narrative_label", "")
                    or "source text only - not MedAI interpretation"
                ),
                "observations": observations,
            }
        )
    return {
        "package_id": package_id,
        "safe_source_document_id": source_id,
        "selected_document_category": str(getattr(draft, "selected_document_category", "") or "AI-assisted extraction"),
        "selected_medical_specialty_domain": validate_specialty_key(
            str(getattr(draft, "selected_specialty_domain", "") or "general")
        ),
        "selected_medical_specialty_label": specialty_label(
            str(getattr(draft, "selected_specialty_domain", "") or "general")
        ),
        "detected_document_family_type": str(getattr(draft, "document_type", "") or "Unknown"),
        "source_modality": _source_modality_label(str(getattr(draft, "source_modality", "") or "local_text")),
        "package_status": "review-bound",
        "record_ids": record_ids,
        "record_count": len(record_ids),
        "sections": sections,
        "actions": package_action_plan([]),
        "auto_accept_allowed": False,
        "review_required": True,
        "ai_assisted_draft": True,
        "active_written_count": 0,
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
                "visual_semantic": "non_red_primary",
            },
            {
                "key": "reject_package",
                "label": "Reject package",
                "enabled": bool(record_ids),
                "disclaimer": REJECT_DISCLAIMER,
                "visual_semantic": "red_destructive",
            },
            {
                "key": "defer_package",
                "label": "Defer package",
                "enabled": bool(record_ids),
                "disclaimer": DEFER_DISCLAIMER,
                "visual_semantic": "neutral_secondary",
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
    return bool(
        structured.get("source_visible_observation")
        or structured.get("parser_name")
        or structured.get("section_heading")
        or structured.get("document_category")
        or structured.get("source_modality")
        or record.session_id
    )


def _source_group_key(record: MKBRecord, structured: dict[str, Any]) -> str:
    return "|".join(
        [
            str(record.session_id or "sessionless"),
            str(structured.get("source_document_id") or record.source_name or "source"),
            str(structured.get("source_modality") or record.source_type or "unknown"),
        ]
    )


def _append_observation(package: SourceExtractionPackage, record: MKBRecord, structured: dict[str, Any]) -> str:
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
    observation = _observation_from_record(record, structured, heading)
    section.observations.append(observation)
    return observation.normalization_status


def _observation_from_record(record: MKBRecord, structured: dict[str, Any], heading: str) -> SourceObservation:
    review_status = str(structured.get("operator_review_status") or record.status or "review-bound")
    if record.requires_review or record.tier == "quarantined":
        review_status = "review-bound" if review_status in {"pending_validation_review", "queued_for_review", "active"} else review_status
    label = _clean_cell(
        structured.get("test_name")
        or structured.get("field_label")
        or structured.get("section_heading")
        or structured.get("name")
        or display_content_public_safe(record)
    )
    normalized = normalize_package_observation_fields(
        label=label,
        value=structured.get("value") or structured.get("field_value") or "",
        reference_interval=structured.get("reference_range") or structured.get("normal_range") or "",
        flag=structured.get("flag") or "",
        unit=structured.get("unit") or "",
    )
    return SourceObservation(
        record_id=safe_record_id(record.id),
        record_id_full=record.id,
        fact_type=record.fact_type,
        label=normalized["label"],
        value=normalized["value"],
        flag=normalized["flag"],
        unit=normalized["unit"],
        reference_interval=normalized["reference_interval"],
        source_section=heading,
        review_status=review_status,
        display_content=display_content_public_safe(record),
        row_kind=normalized["row_kind"],
        normalization_status=normalized["normalization_status"],
    )


def normalize_package_observation_fields(
    *,
    label: Any,
    value: Any,
    reference_interval: Any = "",
    flag: Any = "",
    unit: Any = "",
) -> dict[str, str]:
    """Normalize source-visible observation cells without inferring content."""
    label_text = _clean_cell(label)
    value_text = _clean_cell(value)
    original_label_text = label_text
    original_value_text = value_text
    reference_text = _clean_cell(reference_interval)
    flag_text = _clean_cell(flag)
    unit_text = _clean_cell(unit)
    status = "unchanged"
    row_kind = "observation"

    for prefix in (label_text, f"{label_text}:"):
        if label_text and value_text.lower().startswith(prefix.lower()):
            value_text = _clean_cell(value_text[len(prefix) :])
            value_text = value_text.lstrip(":").strip()
            status = "normalized"

    normal_range_match = re.search(
        r"(?i)\bNormal range:\s*(?P<range>.+?)(?:\s+Normal value:\s*(?P<normal>.+))?$",
        value_text,
    )
    if normal_range_match:
        extracted_range = _clean_cell(normal_range_match.group("range") or "")
        extracted_normal = _clean_cell(normal_range_match.group("normal") or "")
        if extracted_normal and " normal value:" not in extracted_range.lower():
            value_text = extracted_normal
            reference_text = reference_text or extracted_range
            status = "normalized"
        elif extracted_range and not reference_text:
            reference_text = extracted_range
            value_text = ""
            status = "normalized"

    normal_value_match = re.search(r"(?i)\bNormal value:\s*(?P<normal>.+)$", value_text)
    if normal_value_match and not normal_range_match:
        value_text = _clean_cell(normal_value_match.group("normal") or "")
        status = "normalized"

    if _looks_malformed_pair(original_label_text, original_value_text):
        row_kind = "source_visible_note"
        status = "malformed_note"
        reference_text = reference_text or ""
        flag_text = ""
        unit_text = ""

    return {
        "label": label_text,
        "value": value_text,
        "flag": flag_text,
        "unit": unit_text,
        "reference_interval": reference_text,
        "row_kind": row_kind,
        "normalization_status": status,
    }


def _looks_malformed_pair(label: str, value: str) -> bool:
    if not value:
        return False
    lower = value.lower()
    field_markers = sum(1 for marker in ("normal value:", "normal range:", "reference range:", "value:") if marker in lower)
    return bool(label and label.lower() in {"normal value", "normal range", "reference range"} and field_markers >= 1)


def _clean_cell(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _display_document_family(structured: dict[str, Any]) -> str:
    selected = str(structured.get("document_category") or "").strip()
    detected = str(
        structured.get("document_family")
        or structured.get("document_type")
        or structured.get("source_document_type")
        or ""
    ).strip()
    if selected.lower() == "urinalysis" and detected.lower() in {"", "unknown", "treatment plan"}:
        return "Urinalysis"
    return detected or selected or "Unknown"


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
    "source_package_from_ai_draft",
    "normalize_package_observation_fields",
    "package_action_plan",
    "accept_package_after_source_comparison",
    "reject_package",
    "defer_package",
]
