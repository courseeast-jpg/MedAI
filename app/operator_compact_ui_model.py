"""Streamlit-free compact operator layout requirements.

MEDAI-OPERATOR-ONE-SCREEN-UI-POLISH-11E.
"""
from __future__ import annotations

from dataclasses import dataclass


COMPACT_HEADER_ITEMS: tuple[str, ...] = (
    "Local safe mode",
    "Human review",
    "Local only",
    "Cloud APIs off",
    "Privacy check on",
)

RUN_REVIEW_FIRST_VIEWPORT_CONTROLS: tuple[str, ...] = (
    "Document category",
    "Medical specialty / domain",
    "Upload files",
    "Start run",
    "Documents waiting",
    "Current run status",
)

MKB_EXPLORER_FIRST_VIEWPORT_CONTROLS: tuple[str, ...] = (
    "Total",
    "Active",
    "Quarantined / review-bound",
    "Superseded / rejected",
    "Specialty/domain filter",
    "Tier/status filter",
    "Fact type filter",
)

REVIEW_QUEUE_FIRST_VIEWPORT_CONTROLS: tuple[str, ...] = (
    "Needs review count",
    "Source comparison disclaimer",
    "Accept",
    "Reject",
    "Defer",
)

ADVANCED_HIDDEN_BY_DEFAULT: tuple[str, ...] = (
    "Build / audit details",
    "Operator Control Panel",
    "Validation Batch Audit",
    "Validation History",
    "Safety & Governance",
    "Terminology Admin",
    "terminology phase status",
    "implementation phase status",
    "debug/report internals",
)


@dataclass(frozen=True)
class CompactOperatorUiModel:
    compact_header_items: tuple[str, ...]
    run_review_first_viewport_controls: tuple[str, ...]
    mkb_explorer_first_viewport_controls: tuple[str, ...]
    review_queue_first_viewport_controls: tuple[str, ...]
    advanced_hidden_by_default: tuple[str, ...]


def build_compact_operator_ui_model() -> CompactOperatorUiModel:
    return CompactOperatorUiModel(
        compact_header_items=COMPACT_HEADER_ITEMS,
        run_review_first_viewport_controls=RUN_REVIEW_FIRST_VIEWPORT_CONTROLS,
        mkb_explorer_first_viewport_controls=MKB_EXPLORER_FIRST_VIEWPORT_CONTROLS,
        review_queue_first_viewport_controls=REVIEW_QUEUE_FIRST_VIEWPORT_CONTROLS,
        advanced_hidden_by_default=ADVANCED_HIDDEN_BY_DEFAULT,
    )


__all__ = [
    "ADVANCED_HIDDEN_BY_DEFAULT",
    "COMPACT_HEADER_ITEMS",
    "CompactOperatorUiModel",
    "MKB_EXPLORER_FIRST_VIEWPORT_CONTROLS",
    "REVIEW_QUEUE_FIRST_VIEWPORT_CONTROLS",
    "RUN_REVIEW_FIRST_VIEWPORT_CONTROLS",
    "build_compact_operator_ui_model",
]
