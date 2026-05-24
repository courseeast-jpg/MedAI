"""Streamlit-free render-plan helper for the Run & Review extracted-info card.

MEDAI-CORPUS-EXTRACTION-TO-MKB-MINIMUM-01.

This module builds a public-safe dict the Streamlit layer can render. It
imports nothing from Streamlit so it is testable in any environment.

Contract:

* The helper accepts a TestFileResult-shaped dict.
* The helper never reads from disk, never opens the DB, never calls
  external services.
* The helper never emits raw OCR text, raw source lines, raw filenames,
  private paths, or PHI.
* When no facts are present, the helper returns a render plan whose
  ``rows`` list is empty and whose ``message`` field instructs the
  operator that classification succeeded only.
"""
from __future__ import annotations

from typing import Any

from execution.extracted_medical_facts import (
    CONSERVATIVE_CONFIDENCE,
    PARSER_NAME,
    PARSER_VERSION,
    facts_for_ui,
)
from app.operator_review_actions import (
    ACCEPT_DISCLAIMER,
    DEFER_DISCLAIMER,
    REJECT_DISCLAIMER,
    render_action_affordances_plan,
)

#: Section heading the operator sees in the Run & Review card.
SECTION_HEADING = "Extracted information preview"

#: Disclaimer line, replacing the absolute prose
#: "The lab values have not been checked or accepted."
DISCLAIMER_LINE = (
    "Structured factual observations may be present below. They are not "
    "clinically interpreted and require source comparison before use."
)

#: Empty-state line shown when no facts were extracted.
EMPTY_STATE_LINE = (
    "No structured medical facts extracted. Classification succeeded only."
)


def _facts_payload_from_item(item: dict[str, Any]) -> dict[str, Any]:
    """Pull the facts payload from a TestFileResult-shaped dict.

    The new fields are added by the pipeline / test_launcher hand-off:

    * ``extracted_medical_facts_preview_safe`` (list[dict])
    * ``extracted_medical_fact_count`` (int)
    * ``extraction_to_mkb_candidate_count`` (int)
    * ``extraction_to_mkb_written_count`` (int)
    * ``extraction_to_mkb_review_count`` (int)
    """
    payload: dict[str, Any] = {
        "extracted_medical_facts_preview_safe": item.get(
            "extracted_medical_facts_preview_safe"
        )
        or [],
        "extracted_medical_fact_count": int(
            item.get("extracted_medical_fact_count") or 0
        ),
        "extraction_to_mkb_candidate_count": int(
            item.get("extraction_to_mkb_candidate_count") or 0
        ),
        "extraction_to_mkb_written_count": int(
            item.get("extraction_to_mkb_written_count") or 0
        ),
        "extraction_to_mkb_review_count": int(
            item.get("extraction_to_mkb_review_count") or 0
        ),
    }
    return payload


def build_extracted_information_preview_plan(item: dict[str, Any]) -> dict[str, Any]:
    """Build a render-plan dict for the Run & Review card.

    Returns a dict with:

    * ``section_heading`` (str)
    * ``disclaimer_line`` (str)
    * ``columns`` (list[str])
    * ``rows`` (list[dict])
    * ``row_count`` (int)
    * ``counts`` (dict[str, int]): ``structured_facts_extracted``,
      ``written_to_mkb``, ``needs_review``
    * ``message`` (str) — operator-visible explanation
    * ``parser_name`` / ``parser_version`` (str)
    * ``auto_accept_allowed`` (bool, always False)
    * ``review_required`` (bool, always True)
    """
    payload = _facts_payload_from_item(item)
    ui = facts_for_ui(payload)
    counts = dict(ui.get("counts") or {})
    counts.setdefault(
        "structured_facts_extracted", payload["extracted_medical_fact_count"]
    )
    counts.setdefault(
        "written_to_mkb", payload["extraction_to_mkb_written_count"]
    )
    counts.setdefault(
        "needs_review", payload["extraction_to_mkb_review_count"]
    )
    rows = list(ui.get("rows") or [])
    message = EMPTY_STATE_LINE if not rows else DISCLAIMER_LINE
    # MEDAI-CORPUS-EXTRACTION-TO-MKB-MINIMUM-03: attach per-row action
    # affordance plans so the Run & Review card can render the operator
    # accept / reject / defer panel below the preview table.
    row_actions: list[dict[str, Any]] = []
    fact_ids = list(item.get("extracted_medical_fact_record_ids") or [])
    record_state_lookup = item.get("extracted_medical_fact_record_states") or {}
    for index, row in enumerate(rows):
        record_id = ""
        state: dict[str, Any] = {}
        if index < len(fact_ids):
            record_id = str(fact_ids[index])
        if isinstance(record_state_lookup, dict) and record_id in record_state_lookup:
            state = dict(record_state_lookup.get(record_id) or {})
        plan_state = {
            "record_id": record_id,
            "fact_type": str(row.get("type") or state.get("fact_type") or "test_result"),
            "tier": str(state.get("tier") or "quarantined"),
            "status": str(state.get("status") or "active"),
            "requires_review": bool(state.get("requires_review", True)),
            "ddi_status": str(state.get("ddi_status") or ""),
        }
        row_actions.append(render_action_affordances_plan(plan_state))
    return {
        "section_heading": SECTION_HEADING,
        "disclaimer_line": message,
        "columns": list(ui.get("columns") or []),
        "rows": rows,
        "row_count": len(rows),
        "counts": counts,
        "message": message,
        "parser_name": str(ui.get("parser_name") or PARSER_NAME),
        "parser_version": str(ui.get("parser_version") or PARSER_VERSION),
        "auto_accept_allowed": False,
        "review_required": True,
        "conservative_confidence_floor": CONSERVATIVE_CONFIDENCE,
        "row_actions": row_actions,
        "operator_review_disclaimers": {
            "accept": ACCEPT_DISCLAIMER,
            "reject": REJECT_DISCLAIMER,
            "defer": DEFER_DISCLAIMER,
        },
    }


__all__ = [
    "SECTION_HEADING",
    "DISCLAIMER_LINE",
    "EMPTY_STATE_LINE",
    "build_extracted_information_preview_plan",
]
