"""
MedAI — Document Classifier (Phase 5 of hybrid-extraction design).

Classifies incoming documents into a document type, assigns a trust level,
and picks the default ingestion tier (``active`` vs ``hypothesis``). The
pipeline uses this to decide whether extracted facts bypass the hypothesis
tier and land directly in the active MKB.

Tier assignment policy
----------------------
* ``active``      — clinical documents (trust 1) and peer-reviewed guidelines
                    (trust 2). Facts written here are immediately usable in
                    synthesis.
* ``hypothesis``  — AI-derived, user-uploaded web content, food guides,
                    unverified PDFs. Facts land in the hypothesis tier and
                    require promotion before use.
* ``quarantined`` — explicitly untrusted sources or OCR confidence below
                    ``ocr_min_confidence`` (default 0.60). These are held for
                    user review before any write.

Classification is rules-based; the code is intentionally small and
extensible. Add new patterns to ``_CLASSIFIER_RULES`` below.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from loguru import logger


# ── Document-type rules ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class _Rule:
    document_type:  str
    trust_level:    int                 # 1..5 per app.config
    default_tier:   str                 # active | hypothesis | quarantined
    keyword_weight: float = 1.0
    keywords:       tuple[str, ...] = ()
    filename_hints: tuple[str, ...] = ()
    min_matches:    int = 1


# Keyword buckets are lowercased when matched. Patterns are plain substrings
# unless wrapped in /.../ which is treated as a regex.
_CLASSIFIER_RULES: list[_Rule] = [
    # ── Tier ACTIVE ──────────────────────────────────────────────────────────
    _Rule(
        document_type="clinical_note",
        trust_level=1,
        default_tier="active",
        keywords=(
            "chief complaint", "history of present illness", "assessment and plan",
            "physical examination", "review of systems", "discharge summary",
            "operative report", "progress note", "consultation note",
            "medical record number", "attending physician",
        ),
        filename_hints=("clinic", "clinical", "hospital", "discharge", "consult",
                        "op_report", "progress_note"),
        min_matches=2,
    ),
    _Rule(
        document_type="lab_report",
        trust_level=1,
        default_tier="active",
        keywords=(
            "laboratory report", "lab results", "reference range", "specimen",
            "cbc", "comprehensive metabolic panel", "hba1c", "tsh",
            "/\\b(?:wbc|rbc|hgb|plt|alt|ast|bun|cr)\\b:?\\s*\\d/",
            "collection date", "accession number",
        ),
        filename_hints=("lab", "labs", "bloodwork", "results"),
        min_matches=2,
    ),
    _Rule(
        document_type="imaging",
        trust_level=1,
        default_tier="active",
        keywords=(
            "impression:", "findings:", "mri", "ct scan", "ultrasound",
            "radiologist", "contrast", "no acute findings",
        ),
        filename_hints=("mri", "ct", "xray", "imaging", "radiology", "ultrasound"),
        min_matches=2,
    ),
    _Rule(
        document_type="prescription",
        trust_level=1,
        default_tier="active",
        keywords=(
            "rx:", "prescription", "take ", " once daily", " twice daily",
            " tid", " bid", " qd", " qid", " prn",
            "/\\b\\d+\\s?mg\\b/", "/\\bsig\\b/",
        ),
        filename_hints=("rx", "prescription"),
        min_matches=2,
    ),
    _Rule(
        document_type="peer_reviewed_guideline",
        trust_level=2,
        default_tier="active",
        keywords=(
            "clinical practice guideline", "systematic review", "meta-analysis",
            "cochrane", "published in", "doi:", "pubmed",
        ),
        filename_hints=("guideline", "pubmed", "meta_analysis", "systematic_review"),
        min_matches=2,
    ),

    # ── Tier HYPOTHESIS ──────────────────────────────────────────────────────
    _Rule(
        document_type="food_guide",
        trust_level=3,
        default_tier="hypothesis",
        keywords=(
            "fodmap", "ibs score", "diverticulitis score", "oxalate",
            "crystalluria", "безопасно", "разрешено", "нежелательно",
            "исключить", "safety category", "food rating",
        ),
        filename_hints=("food", "diet", "nutrition", "fodmap"),
        min_matches=1,
    ),
    _Rule(
        document_type="patient_education",
        trust_level=4,
        default_tier="hypothesis",
        keywords=(
            "what is", "how to manage", "living with", "frequently asked",
            "patient information", "educational material",
        ),
        filename_hints=("patient", "education", "info", "faq"),
        min_matches=2,
    ),
    _Rule(
        document_type="web_article",
        trust_level=4,
        default_tier="hypothesis",
        keywords=(
            "http://", "https://", "www.", "read more", "subscribe",
            "copyright ©", "all rights reserved",
        ),
        filename_hints=("web", "html", "article"),
        min_matches=1,
    ),
    _Rule(
        document_type="ai_response",
        trust_level=3,
        default_tier="hypothesis",
        keywords=("as an ai", "language model", "chatgpt", "claude", "gemini"),
        filename_hints=("ai", "chat", "response"),
        min_matches=1,
    ),
]


# ── Result ───────────────────────────────────────────────────────────────────

@dataclass
class ClassificationResult:
    document_type:       str
    trust_level:         int
    default_tier:        str                # active | hypothesis | quarantined
    confidence:          float               # 0..1 based on rule strength
    matched_keywords:    list[str] = field(default_factory=list)
    ocr_confidence:      Optional[float] = None
    ocr_errors_detected: bool = False
    reasons:             list[str] = field(default_factory=list)


# ── Classifier ───────────────────────────────────────────────────────────────

class DocumentClassifier:
    """Classify a document and decide its default tier."""

    def __init__(
        self,
        rules: Iterable[_Rule] = _CLASSIFIER_RULES,
        ocr_min_confidence: float = 0.60,
        ocr_validator=None,
    ):
        self._rules = list(rules)
        self._ocr_min = ocr_min_confidence
        self._ocr = ocr_validator

    # ── public API ───────────────────────────────────────────────────────────

    def classify(
        self,
        text: str,
        source_path: Optional[str | Path] = None,
        source_type: str = "document",
        explicit_trust: Optional[int] = None,
    ) -> ClassificationResult:
        """Classify ``text`` and return the ingestion policy.

        :param source_path:    path or URL to include in filename hint matching.
        :param source_type:    ``document|ai_response|manual|web|guideline``.
        :param explicit_trust: optional override (e.g. when the caller knows
                               the document is from a trusted clinical source).
        """
        text = text or ""
        lower = text.lower()
        filename = (str(source_path) if source_path else "").lower()

        best: Optional[tuple[_Rule, list[str], int]] = None
        for rule in self._rules:
            matched = self._match_keywords(lower, rule.keywords)
            hits = len(matched)
            if hits >= rule.min_matches or self._matches_filename(filename, rule.filename_hints):
                if best is None or hits > best[2]:
                    best = (rule, matched, hits)

        reasons: list[str] = []
        if best is None:
            document_type = "generic"
            trust_level = 4
            tier = "hypothesis"
            confidence = 0.40
            matched: list[str] = []
            reasons.append("no rule matched — defaulting to hypothesis tier")
        else:
            rule, matched, hits = best
            document_type = rule.document_type
            trust_level = rule.trust_level
            tier = rule.default_tier
            confidence = min(1.0, 0.50 + 0.10 * hits)
            reasons.append(f"matched rule '{rule.document_type}' with {hits} keyword hit(s)")

        if source_type == "ai_response":
            trust_level = max(trust_level, 3)
            tier = "hypothesis"
            reasons.append("source_type=ai_response forces hypothesis tier")
        elif source_type == "web":
            trust_level = max(trust_level, 4)
            tier = "hypothesis"
            reasons.append("source_type=web forces hypothesis tier")
        elif source_type == "manual":
            trust_level = min(trust_level, 2)
            tier = "active"
            reasons.append("source_type=manual → trust=2, active")

        if explicit_trust is not None:
            trust_level = int(explicit_trust)
            if trust_level <= 2:
                tier = "active"
            else:
                tier = "hypothesis" if tier == "active" else tier
            reasons.append(f"explicit_trust={trust_level} override")

        # OCR gate.
        ocr_confidence: Optional[float] = None
        ocr_errors = False
        if self._ocr is not None:
            report = self._ocr.validate_ocr_quality(text)
            ocr_confidence = report.get("confidence")
            ocr_errors = bool(report.get("errors_detected"))
            if ocr_confidence is not None and ocr_confidence < self._ocr_min:
                tier = "quarantined"
                reasons.append(
                    f"OCR confidence {ocr_confidence:.2f} below threshold "
                    f"{self._ocr_min:.2f} → quarantined"
                )

        result = ClassificationResult(
            document_type=document_type,
            trust_level=trust_level,
            default_tier=tier,
            confidence=confidence,
            matched_keywords=matched,
            ocr_confidence=ocr_confidence,
            ocr_errors_detected=ocr_errors,
            reasons=reasons,
        )
        logger.debug(
            f"Classified '{source_path or '<text>'}': type={document_type}, "
            f"trust={trust_level}, tier={tier}, conf={confidence:.2f}"
        )
        return result

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _match_keywords(text: str, keywords: tuple[str, ...]) -> list[str]:
        hits: list[str] = []
        for kw in keywords:
            if kw.startswith("/") and kw.endswith("/"):
                try:
                    if re.search(kw[1:-1], text, re.I):
                        hits.append(kw)
                except re.error:
                    continue
            elif kw in text:
                hits.append(kw)
        return hits

    @staticmethod
    def _matches_filename(filename: str, hints: tuple[str, ...]) -> bool:
        if not filename or not hints:
            return False
        return any(h in filename for h in hints)
