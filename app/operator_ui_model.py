"""Streamlit-free operator-first UI requirements for MedAI.

MEDAI-OPERATOR-FIRST-UI-RESTORE-11C.

This module contains product-facing navigation and control requirements so
tests and validation scripts can verify the default MedAI workflow without
launching Streamlit.
"""
from __future__ import annotations

from dataclasses import dataclass


RUN_REVIEW_TAB = "Run & Review"
MKB_EXPLORER_TAB = "MKB Explorer"
REVIEW_QUEUE_TAB = "Review Queue"

DEFAULT_PRIMARY_TABS: tuple[str, ...] = (
    RUN_REVIEW_TAB,
    MKB_EXPLORER_TAB,
    REVIEW_QUEUE_TAB,
)

ADVANCED_TABS: tuple[str, ...] = (
    "Operator Control Panel",
    "Validation Batch Audit",
    "Validation History",
    "Safety & Governance",
    "Terminology Admin",
)

DEFAULT_HIDDEN_ENGINEERING_SECTIONS: tuple[str, ...] = (
    "Build / audit details",
    "phase implementation status",
    "terminology phase status",
    "validation internals",
    "raw debug/report details",
)

RUN_REVIEW_REQUIRED_CONTROLS: tuple[str, ...] = (
    "Document category",
    "Medical specialty / domain",
    "Upload files",
    "Start run",
    "Current run status",
)

MKB_EXPLORER_REQUIRED_CONTROLS: tuple[str, ...] = (
    "Total records count",
    "Active count",
    "Quarantined / review-bound count",
    "Superseded / rejected count",
    "Specialty/domain filter",
    "Tier/status filter",
    "Fact type filter",
)

REVIEW_QUEUE_REQUIRED_CONTROLS: tuple[str, ...] = (
    "Records needing review",
    "Accept",
    "Reject",
    "Defer",
    "Source comparison disclaimer",
)

SOURCE_COMPARISON_DISCLAIMER = (
    "Accept only after comparing with source. This does not clinically interpret the result."
)


@dataclass(frozen=True)
class OperatorUiModel:
    default_primary_tabs: tuple[str, ...]
    advanced_tabs: tuple[str, ...]
    visible_tabs: tuple[str, ...]
    default_hidden_engineering_sections: tuple[str, ...]
    run_review_required_controls: tuple[str, ...]
    mkb_explorer_required_controls: tuple[str, ...]
    review_queue_required_controls: tuple[str, ...]


def build_operator_ui_model(*, show_advanced_tools: bool = False) -> OperatorUiModel:
    visible_tabs = DEFAULT_PRIMARY_TABS + (ADVANCED_TABS if show_advanced_tools else ())
    return OperatorUiModel(
        default_primary_tabs=DEFAULT_PRIMARY_TABS,
        advanced_tabs=ADVANCED_TABS,
        visible_tabs=visible_tabs,
        default_hidden_engineering_sections=DEFAULT_HIDDEN_ENGINEERING_SECTIONS,
        run_review_required_controls=RUN_REVIEW_REQUIRED_CONTROLS,
        mkb_explorer_required_controls=MKB_EXPLORER_REQUIRED_CONTROLS,
        review_queue_required_controls=REVIEW_QUEUE_REQUIRED_CONTROLS,
    )


def operator_tabs(show_advanced_tools: bool = False) -> list[str]:
    return list(build_operator_ui_model(show_advanced_tools=show_advanced_tools).visible_tabs)


def advanced_only_tabs() -> list[str]:
    return list(ADVANCED_TABS)


__all__ = [
    "ADVANCED_TABS",
    "DEFAULT_HIDDEN_ENGINEERING_SECTIONS",
    "DEFAULT_PRIMARY_TABS",
    "MKB_EXPLORER_REQUIRED_CONTROLS",
    "MKB_EXPLORER_TAB",
    "OperatorUiModel",
    "REVIEW_QUEUE_REQUIRED_CONTROLS",
    "REVIEW_QUEUE_TAB",
    "RUN_REVIEW_REQUIRED_CONTROLS",
    "RUN_REVIEW_TAB",
    "SOURCE_COMPARISON_DISCLAIMER",
    "advanced_only_tabs",
    "build_operator_ui_model",
    "operator_tabs",
]
