from __future__ import annotations

from dataclasses import asdict, dataclass


CYRILLIC_RANGE = range(0x0400, 0x0500)


@dataclass(frozen=True)
class LanguageSupportResult:
    detected_language: str
    language_confidence: float
    script_detected: str
    cyrillic_detected: bool
    requires_ocr: bool
    language_route_note: str
    translation_status: str
    language_support_status: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def detect_language_support(*, text: str) -> LanguageSupportResult:
    normalized_text = str(text or "")
    latin_letters = sum(1 for char in normalized_text if char.isascii() and char.isalpha())
    cyrillic_letters = sum(1 for char in normalized_text if _is_cyrillic(char))
    alphabetic_total = latin_letters + cyrillic_letters
    nonspace_total = sum(1 for char in normalized_text if not char.isspace())

    cyrillic_ratio = cyrillic_letters / alphabetic_total if alphabetic_total else 0.0
    latin_ratio = latin_letters / alphabetic_total if alphabetic_total else 0.0
    cyrillic_detected = cyrillic_letters > 0
    requires_ocr = bool(nonspace_total and alphabetic_total == 0)

    if cyrillic_letters >= 3 and cyrillic_ratio >= 0.7:
        detected_language = "russian"
        language_confidence = round(max(cyrillic_ratio, 0.85), 3)
        script_detected = "cyrillic"
        translation_status = "pending_translation"
        language_route_note = "metadata_only:russian_cyrillic_detected"
        language_support_status = "supported_metadata_only"
    elif latin_letters >= 3 and latin_ratio >= 0.7:
        detected_language = "english"
        language_confidence = round(max(latin_ratio, 0.85), 3)
        script_detected = "latin"
        translation_status = "not_required"
        language_route_note = "metadata_only:english_latin_detected"
        language_support_status = "supported_metadata_only"
    elif cyrillic_letters >= 2 and latin_letters >= 2:
        detected_language = "mixed"
        language_confidence = round(max(min(cyrillic_ratio, latin_ratio), 0.6), 3)
        script_detected = "mixed"
        translation_status = "pending_translation" if cyrillic_detected else "not_required"
        language_route_note = "metadata_only:mixed_language_detected"
        language_support_status = "supported_metadata_only"
    else:
        detected_language = "unknown"
        language_confidence = 0.0 if alphabetic_total == 0 else round(max(cyrillic_ratio, latin_ratio), 3)
        script_detected = "cyrillic" if cyrillic_detected else ("latin" if latin_letters else "unknown")
        translation_status = "skipped_no_translator" if cyrillic_detected else "not_required"
        language_route_note = "metadata_only:language_signal_insufficient"
        language_support_status = "unknown_metadata_only"

    return LanguageSupportResult(
        detected_language=detected_language,
        language_confidence=language_confidence,
        script_detected=script_detected,
        cyrillic_detected=cyrillic_detected,
        requires_ocr=requires_ocr,
        language_route_note=language_route_note,
        translation_status=translation_status,
        language_support_status=language_support_status,
    )


def _is_cyrillic(char: str) -> bool:
    if not char:
        return False
    return ord(char) in CYRILLIC_RANGE
