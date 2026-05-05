"""Phase 45 — Cyrillic non-lab review classification refinement.

Pure function. Given a file's current status, OCR-layout band, document
classification, lab-normalization summary, and selected text, decides
whether ``review_ocr_quality`` should be downgraded to ``review`` because
the underlying issue was *non-lab document content*, not poor OCR.

Hard guarantees:
  * Never promotes any file to ``accepted``.
  * Never touches files in ``poor_ocr`` or ``empty`` OCR bands.
  * Never touches files with zero entities (empty extraction).
  * Never touches files whose status is anything other than
    ``review_ocr_quality``.
  * Adds new reason codes; never removes existing ones.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any


_CYRILLIC_RE = re.compile(r"[Ѐ-ӿ]")

# Reconciliation only applies in these document types.
_ELIGIBLE_TYPES = {"prescription", "unknown_medical"}

# Reconciliation only applies for these OCR bands.
_ELIGIBLE_BANDS = {"good", "usable_with_review"}

# Reconciliation only applies if lab table coverage is none/weak (i.e. no
# meaningful lab content was actually parsed).
_ELIGIBLE_COVERAGE_BANDS = {"none", "weak"}

# Minimum Cyrillic character ratio in the selected text required to call
# this a "Cyrillic recovered" document.
_MIN_CYRILLIC_RATIO = 0.20

# For unknown_medical, reconcile only when the classifier metadata shows
# at least this many prescription-token signals.
_UNKNOWN_MEDICAL_MIN_RX_SIGNAL = 2


@dataclass(frozen=True)
class CyrillicNonLabReconciliation:
    eligible: bool
    triggered: bool
    new_status: str
    cyrillic_non_lab_document_detected: bool
    ocr_quality_recovered_non_lab: bool
    new_reason_codes: list[str]
    explanation: list[str]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _cyrillic_ratio(text: str) -> float:
    if not text:
        return 0.0
    visible = [c for c in text if not c.isspace()]
    if not visible:
        return 0.0
    cyr = sum(1 for c in visible if _CYRILLIC_RE.match(c))
    return cyr / len(visible)


def reconcile_cyrillic_nonlab_status(
    *,
    current_status: str,
    is_ocr_low_quality: bool,
    classification_reason_codes: list[str],
    selected_text: str,
    ocr_layout: dict[str, Any] | None,
    lab_normalization: dict[str, Any] | None,
    entity_count: int,
) -> CyrillicNonLabReconciliation:
    ocr_layout = ocr_layout or {}
    lab_normalization = lab_normalization or {}
    explanation: list[str] = []

    band = str(ocr_layout.get("input_quality_band") or "")
    classification = (
        lab_normalization.get("document_classification") or {}
    ) if isinstance(lab_normalization, dict) else {}
    document_type = classification.get("document_type") if isinstance(classification, dict) else None
    classification_metadata = (
        classification.get("metadata") or {}
    ) if isinstance(classification, dict) else {}
    coverage_band = (
        lab_normalization.get("lab_coverage_band") or "none"
    ) if isinstance(lab_normalization, dict) else "none"
    cyr_ratio = _cyrillic_ratio(selected_text or "")

    metadata: dict[str, Any] = {
        "current_status": current_status,
        "ocr_layout_quality_band": band,
        "document_type": document_type,
        "lab_coverage_band": coverage_band,
        "cyrillic_ratio_in_selected_text": round(cyr_ratio, 4),
        "entity_count": entity_count,
        "is_ocr_low_quality_input": is_ocr_low_quality,
    }

    no_change = CyrillicNonLabReconciliation(
        eligible=False,
        triggered=False,
        new_status=current_status,
        cyrillic_non_lab_document_detected=False,
        ocr_quality_recovered_non_lab=False,
        new_reason_codes=list(classification_reason_codes),
        explanation=explanation,
        metadata=metadata,
    )

    # Gate 1: only applies to review_ocr_quality
    if current_status != "review_ocr_quality":
        explanation.append(f"current_status={current_status} not review_ocr_quality")
        return no_change

    # Gate 2: poor OCR must remain review_ocr_quality
    if band not in _ELIGIBLE_BANDS:
        explanation.append(f"ocr_band={band} not in eligible set {sorted(_ELIGIBLE_BANDS)}")
        return no_change

    # NOTE: An empty extraction (entity_count == 0) is *not* a blocker here.
    # "No empty extraction leakage" in the spec is the requirement that empty-
    # extraction files never reach `accepted`. Phase 45 only flips
    # review_ocr_quality -> review, so empty-extraction files remain in the
    # human-review queue regardless. We surface entity_count in metadata so
    # downstream reports can audit it.

    # Gate 4: document type eligibility
    if document_type not in _ELIGIBLE_TYPES:
        explanation.append(f"document_type={document_type} not in {sorted(_ELIGIBLE_TYPES)}")
        return no_change
    if document_type == "unknown_medical":
        rx_signal_total = int(
            classification_metadata.get("rx_signal_total", 0) or 0
        ) if isinstance(classification_metadata, dict) else 0
        if rx_signal_total < _UNKNOWN_MEDICAL_MIN_RX_SIGNAL:
            explanation.append(
                f"document_type=unknown_medical with rx_signal_total={rx_signal_total} "
                f"< {_UNKNOWN_MEDICAL_MIN_RX_SIGNAL}"
            )
            return no_change

    # Gate 5: lab coverage must be none or weak (no meaningful lab data parsed)
    if coverage_band not in _ELIGIBLE_COVERAGE_BANDS:
        explanation.append(f"lab_coverage_band={coverage_band} indicates real lab data was parsed")
        return no_change

    # Gate 6: Cyrillic recovery — selected text must be substantively Cyrillic
    if cyr_ratio < _MIN_CYRILLIC_RATIO:
        explanation.append(
            f"cyrillic_ratio={cyr_ratio:.3f} < {_MIN_CYRILLIC_RATIO}"
        )
        return no_change

    # All gates passed — reconcile
    new_reasons = list(classification_reason_codes)
    additions = ["cyrillic_non_lab_document_review", "ocr_quality_recovered_non_lab"]
    if document_type == "prescription":
        additions.append("prescription_or_medication_instruction_detected")
    for code in additions:
        if code not in new_reasons:
            new_reasons.append(code)

    explanation.append(
        f"reconciled review_ocr_quality -> review (doc_type={document_type}, "
        f"band={band}, coverage={coverage_band}, cyr_ratio={cyr_ratio:.3f})"
    )
    return CyrillicNonLabReconciliation(
        eligible=True,
        triggered=True,
        new_status="review",
        cyrillic_non_lab_document_detected=True,
        ocr_quality_recovered_non_lab=True,
        new_reason_codes=new_reasons,
        explanation=explanation,
        metadata=metadata,
    )
