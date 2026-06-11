"""Public-safe medical specialty/domain selection helpers.

MEDAI-UI-CAPABILITY-RESTORE-11B.

Specialty selection is routing and organization metadata only. It does not
diagnose, interpret, classify, score, or change extraction behavior.
"""
from __future__ import annotations

from dataclasses import dataclass


DEFAULT_SPECIALTY_KEY = "general"


@dataclass(frozen=True)
class SpecialtyOption:
    key: str
    label: str


SPECIALTY_OPTIONS: tuple[SpecialtyOption, ...] = (
    SpecialtyOption("general", "General medicine"),
    SpecialtyOption("dermatology", "Dermatology"),
    SpecialtyOption("gastroenterology", "Gastroenterology"),
    SpecialtyOption("cardiology", "Cardiology"),
    SpecialtyOption("neurology", "Neurology"),
    SpecialtyOption("endocrinology", "Endocrinology"),
    SpecialtyOption("hematology", "Hematology"),
    SpecialtyOption("nephrology", "Nephrology"),
    SpecialtyOption("pulmonology", "Pulmonology"),
    SpecialtyOption("rheumatology", "Rheumatology"),
    SpecialtyOption("infectious_disease", "Infectious disease"),
    SpecialtyOption("oncology", "Oncology"),
    SpecialtyOption("pediatrics", "Pediatrics"),
    SpecialtyOption("obstetrics_gynecology", "Obstetrics / Gynecology"),
    SpecialtyOption("psychiatry", "Psychiatry"),
    SpecialtyOption("urology", "Urology"),
    SpecialtyOption("ophthalmology", "Ophthalmology"),
    SpecialtyOption("otolaryngology", "Otolaryngology"),
    SpecialtyOption("orthopedics", "Orthopedics"),
    SpecialtyOption("other_review", "Other / needs review"),
)

_LABEL_BY_KEY = {option.key: option.label for option in SPECIALTY_OPTIONS}


def validate_specialty_key(value: str | None) -> str:
    """Return a stable specialty key, falling back safely to general."""
    candidate = str(value or "").strip().lower()
    return candidate if candidate in _LABEL_BY_KEY else DEFAULT_SPECIALTY_KEY


def specialty_label(value: str | None) -> str:
    """Return the public-safe display label for a stable specialty key."""
    return _LABEL_BY_KEY[validate_specialty_key(value)]


def specialty_options_for_ui() -> list[dict[str, str]]:
    """Return options suitable for UI select controls."""
    return [{"key": option.key, "label": option.label} for option in SPECIALTY_OPTIONS]


def specialty_keys_for_ui(*, include_all: bool = False) -> list[str]:
    keys = [option.key for option in SPECIALTY_OPTIONS]
    return ["all", *keys] if include_all else keys


def specialty_labels_for_ui(*, include_all: bool = False) -> list[str]:
    labels = [option.label for option in SPECIALTY_OPTIONS]
    return ["All specialties", *labels] if include_all else labels


def specialty_key_from_label(label: str | None, *, include_all: bool = False) -> str:
    if include_all and str(label or "") == "All specialties":
        return "all"
    for option in SPECIALTY_OPTIONS:
        if option.label == label:
            return option.key
    return DEFAULT_SPECIALTY_KEY


__all__ = [
    "DEFAULT_SPECIALTY_KEY",
    "SPECIALTY_OPTIONS",
    "SpecialtyOption",
    "specialty_key_from_label",
    "specialty_keys_for_ui",
    "specialty_label",
    "specialty_labels_for_ui",
    "specialty_options_for_ui",
    "validate_specialty_key",
]
