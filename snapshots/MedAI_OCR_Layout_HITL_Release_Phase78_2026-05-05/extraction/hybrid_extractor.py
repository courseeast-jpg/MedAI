"""
MedAI — Hybrid Extractor (Phase 4 of hybrid-extraction design).

Combines the LLM-based ``Extractor`` with rules-based edge-case handlers that
operate on the extracted text. The handlers are independently callable so
pipeline stages (and tests) can invoke them directly:

    detect_negation(entity, text)       -> (is_negated, negation_type)
    identify_subject(entity, text)      -> (subject, confidence)
    assess_certainty(entity, text)      -> (certainty_level, confidence)
    extract_temporal_info(text, doc_date)
                                        -> (event_date, confidence, date_range)
    normalize_measurement(value, unit, entity)
                                        -> (normalized_value, normalized_unit)
    validate_ocr_quality(text, image_path)
                                        -> dict (delegates to OCRValidator)

``extract(text, ...)`` calls the underlying extractor and then decorates the
returned entities with the edge-case metadata.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Optional

from loguru import logger

from extraction.extractor import Extractor
from extraction.ocr_validator import OCRValidator


# ── Pattern libraries ─────────────────────────────────────────────────────

# EC 3 — Negation cues. Patterns look for a cue word preceding (or following)
# the entity mention within a short window. Each pattern is compiled at class
# construction.
_NEGATION_CUES: list[tuple[str, str]] = [
    # explicit denial
    (r"\bdenies?\b",                          "denied"),
    (r"\bdenied\b",                           "denied"),
    (r"\bpatient\s+denies?\b",                "denied"),
    # ruled out
    (r"\bruled\s+out\b",                      "ruled_out"),
    (r"\br/o\b",                              "ruled_out"),
    (r"\bnegative\s+for\b",                   "ruled_out"),
    (r"\bexcluded\b",                         "ruled_out"),
    # absent / no history
    (r"\bno\s+history\s+of\b",                "absent"),
    (r"\bno\s+evidence\s+of\b",               "absent"),
    (r"\bno\s+signs?\s+of\b",                 "absent"),
    (r"\bwithout\b",                          "absent"),
    (r"\babsent\b",                           "absent"),
    (r"^\s*no\s+",                            "absent"),
    (r"\bno\b",                               "absent"),
    (r"\bnot\s+(?:present|observed|noted)\b", "absent"),
]

# EC 4A — Family history cues. Each cue maps a surface form to a relation.
_SUBJECT_CUES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bmother(?:'s|s)?\b", re.I),                                "mother"),
    (re.compile(r"\bfather(?:'s|s)?\b", re.I),                                "father"),
    (re.compile(r"\bdad(?:'s|s)?\b",    re.I),                                "father"),
    (re.compile(r"\bmom(?:'s|s)?\b",    re.I),                                "mother"),
    (re.compile(r"\b(?:brother|sister|sibling)(?:'s|s)?\b", re.I),            "sibling"),
    (re.compile(r"\b(?:son|daughter|child)(?:'s|s)?\b", re.I),                "child"),
    (re.compile(r"\bmaternal\s+(?:grand(?:mother|father|parent))\b", re.I),   "maternal_grandparent"),
    (re.compile(r"\bpaternal\s+(?:grand(?:mother|father|parent))\b", re.I),   "paternal_grandparent"),
    (re.compile(r"\b(?:grandmother|grandfather|grandparent)(?:'s|s)?\b", re.I), "grandparent"),
    (re.compile(r"\baunt(?:'s|s)?\b",   re.I),                                "aunt"),
    (re.compile(r"\buncle(?:'s|s)?\b",  re.I),                                "uncle"),
    (re.compile(r"\bfamily\s+history\s+of\b", re.I),                          "family"),
]

# EC 4B — Certainty cues.
_CERTAINTY_CUES: dict[str, list[re.Pattern[str]]] = {
    "confirmed": [
        re.compile(r"\bdiagnosed\s+with\b",         re.I),
        re.compile(r"\bconfirmed\b",                re.I),
        re.compile(r"\bestablished\b",              re.I),
        re.compile(r"\bbiopsy[-\s]?proven\b",       re.I),
        re.compile(r"\bpathology[-\s]?confirmed\b", re.I),
        re.compile(r"\bknown\s+history\s+of\b",     re.I),
    ],
    "suspected": [
        re.compile(r"\bsuspect(?:ed|s)?\b",   re.I),
        re.compile(r"\bpossible\b",           re.I),
        re.compile(r"\blikely\b",             re.I),
        re.compile(r"\bprobable\b",           re.I),
        re.compile(r"\bconsistent\s+with\b",  re.I),
        re.compile(r"\brule[-\s]?out\b",      re.I),
    ],
    "ruled_out": [
        re.compile(r"\bruled\s+out\b",        re.I),
        re.compile(r"\bexcluded\b",           re.I),
        re.compile(r"\bnegative\s+for\b",     re.I),
    ],
    "differential": [
        re.compile(r"\bdifferential\b",                  re.I),
        re.compile(r"\bconsider\s+\w+\s+vs\.?\s+\w+\b",  re.I),
        re.compile(r"\b\w+\s+vs\.?\s+\w+\b",             re.I),
        re.compile(r"\bworkup\s+for\b",                  re.I),
    ],
    "historical": [
        re.compile(r"\bhistory\s+of\b",      re.I),
        re.compile(r"\bpast\s+medical\b",    re.I),
        re.compile(r"\bprior\b",             re.I),
        re.compile(r"\bresolved\b",          re.I),
    ],
}

# EC 2 — Temporal patterns.
_EXPLICIT_DATE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b"),              # 2024-03-15
    re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b"),              # 03/15/2024 (assume m/d/y)
    re.compile(r"\b(\d{1,2})\.(\d{1,2})\.(\d{4})\b"),            # 15.03.2024 (d.m.y)
]

_RELATIVE_UNITS = {
    "day": 1, "days": 1,
    "week": 7, "weeks": 7,
    "month": 30, "months": 30,
    "year": 365, "years": 365,
}

_RELATIVE_PATTERN = re.compile(
    r"(\d+)\s+(day|days|week|weeks|month|months|year|years)\s+ago", re.I
)

_NAMED_RELATIVE = {
    "today":      0,
    "yesterday":  1,
    "last week":  7,
    "last month": 30,
    "last year":  365,
}

# EC 6 — Unit conversion table.
# Each key is a canonical entity name; value is a dict of
# (input_unit) -> (canonical_unit, conversion_fn).
_UNIT_CONVERSIONS: dict[str, dict[str, tuple[str, Any]]] = {
    "weight": {
        "lb":  ("kg", lambda v: v * 0.45359237),
        "lbs": ("kg", lambda v: v * 0.45359237),
        "pound":  ("kg", lambda v: v * 0.45359237),
        "pounds": ("kg", lambda v: v * 0.45359237),
        "kg":  ("kg", lambda v: v),
        "g":   ("kg", lambda v: v / 1000.0),
        "oz":  ("kg", lambda v: v * 0.0283495),
    },
    "temperature": {
        "f":  ("°C", lambda v: (v - 32) * 5.0 / 9.0),
        "°f": ("°C", lambda v: (v - 32) * 5.0 / 9.0),
        "c":  ("°C", lambda v: v),
        "°c": ("°C", lambda v: v),
    },
    "hba1c": {
        "mmol/mol": ("%", lambda v: (v / 10.929) + 2.15),
        "%":        ("%", lambda v: v),
    },
    "glucose": {
        "mg/dl":  ("mmol/L", lambda v: v / 18.0182),
        "mmol/l": ("mmol/L", lambda v: v),
    },
    "cholesterol": {
        "mg/dl":  ("mmol/L", lambda v: v / 38.67),
        "mmol/l": ("mmol/L", lambda v: v),
    },
    "creatinine": {
        "mg/dl":   ("µmol/L", lambda v: v * 88.4),
        "µmol/l":  ("µmol/L", lambda v: v),
        "umol/l":  ("µmol/L", lambda v: v),
    },
    "height": {
        "in":     ("cm", lambda v: v * 2.54),
        "inches": ("cm", lambda v: v * 2.54),
        "cm":     ("cm", lambda v: v),
        "m":      ("cm", lambda v: v * 100.0),
    },
}


# ── Data container ────────────────────────────────────────────────────────

@dataclass
class EntityAnnotation:
    """Edge-case metadata attached to an extracted entity."""
    entity:              str
    is_negated:          bool = False
    negation_type:       Optional[str] = None
    subject:             str = "patient"
    subject_confidence:  float = 1.0
    certainty:           str = "confirmed"
    certainty_confidence: float = 1.0
    event_date:          Optional[date] = None
    temporal_confidence: float = 1.0
    date_range:          Optional[tuple[date, date]] = None


# ── HybridExtractor ───────────────────────────────────────────────────────

class HybridExtractor:
    """Extractor + edge-case handlers."""

    _WINDOW_CHARS = 80   # how many characters around an entity we inspect

    def __init__(self, extractor: Optional[Extractor] = None,
                 ocr_validator: Optional[OCRValidator] = None):
        self._extractor = extractor   # may be None for tests that only use handlers
        self._ocr = ocr_validator or OCRValidator()
        self._negation_compiled = [(re.compile(p, re.I), t) for p, t in _NEGATION_CUES]

    # ── EC 3: negation ─────────────────────────────────────────────────────

    # Post-entity cues — e.g. "pneumonia ruled out", "fever absent".
    _POST_NEGATION_CUES: list[tuple[str, str]] = [
        (r"\bruled\s+out\b",      "ruled_out"),
        (r"\bexcluded\b",         "ruled_out"),
        (r"\bnegative\b",         "ruled_out"),
        (r"\babsent\b",           "absent"),
        (r"\bnot\s+present\b",    "absent"),
        (r"\bnot\s+observed\b",   "absent"),
        (r"\bresolved\b",         "absent"),
    ]

    def detect_negation(self, entity: str, text: str) -> tuple[bool, Optional[str]]:
        """Return (is_negated, negation_type) for ``entity`` inside ``text``."""
        if not entity or not text:
            return False, None

        for span in self._spans_of(entity, text):
            window = self._window_before(text, span, self._WINDOW_CHARS)
            hit = self._scan_negation(window)
            if hit:
                return True, hit
            # Also examine the phrase that starts at the beginning of the sentence.
            sentence = self._sentence_containing(text, span)
            hit = self._scan_negation(sentence)
            if hit and self._negation_applies_to(sentence, entity):
                return True, hit
            # Post-entity cues.
            after = text[span[1]:span[1] + self._WINDOW_CHARS]
            for pat, label in self._POST_NEGATION_CUES:
                if re.search(pat, after, re.I):
                    return True, label
        return False, None

    def _scan_negation(self, segment: str) -> Optional[str]:
        best: Optional[tuple[int, str]] = None
        for pat, label in self._negation_compiled:
            for m in pat.finditer(segment):
                pos = m.start()
                if best is None or pos > best[0]:
                    best = (pos, label)
        return best[1] if best else None

    def _negation_applies_to(self, sentence: str, entity: str) -> bool:
        # Guard against "no fever, has chest pain" — negation stops at a comma
        # or conjunction that precedes the entity.
        idx = sentence.lower().find(entity.lower())
        if idx == -1:
            return False
        before = sentence[:idx].lower()
        last_cue = max(
            (before.rfind(cue) for cue in ("no ", "denies", "denied", "without",
                                           "ruled out", "negative for", "absent", "excluded")),
            default=-1,
        )
        if last_cue == -1:
            return False
        boundary = max(before.rfind(","), before.rfind(";"),
                       before.rfind(" but "), before.rfind(" however "))
        return last_cue > boundary

    # ── EC 4A: subject attribution ─────────────────────────────────────────

    def identify_subject(self, entity: str, text: str) -> tuple[str, float]:
        """Return (subject, confidence) for ``entity`` inside ``text``."""
        if not entity or not text:
            return "patient", 1.0

        for span in self._spans_of(entity, text):
            window = self._window_before(text, span, self._WINDOW_CHARS)
            sentence = self._sentence_containing(text, span)
            # Family cues in the same sentence first (higher confidence), then
            # the preceding window.
            hit = self._best_subject_match(sentence) or self._best_subject_match(window)
            if hit is None:
                continue
            subject, distance = hit
            if subject == "family":
                return "family_member", 0.80
            confidence = 0.95 if distance < 40 else 0.85
            return subject, confidence
        return "patient", 0.90

    def _best_subject_match(self, segment: str) -> Optional[tuple[str, int]]:
        best: Optional[tuple[str, int]] = None
        for pat, relation in _SUBJECT_CUES:
            m = pat.search(segment)
            if m:
                distance = len(segment) - m.end()
                if best is None or distance < best[1]:
                    best = (relation, distance)
        return best

    # ── EC 4B: certainty ───────────────────────────────────────────────────

    def assess_certainty(self, entity: str, text: str) -> tuple[str, float]:
        """Return (certainty_level, confidence) for ``entity`` inside ``text``."""
        if not entity or not text:
            return "confirmed", 0.5

        sentence = None
        for span in self._spans_of(entity, text):
            sentence = self._sentence_containing(text, span)
            break
        target = sentence or text

        # Priority order — a sentence can match multiple buckets; pick the
        # highest-confidence hit.
        order = ["ruled_out", "suspected", "differential", "historical", "confirmed"]
        for level in order:
            for pat in _CERTAINTY_CUES[level]:
                if pat.search(target):
                    return level, 0.90
        return "confirmed", 0.60

    # ── EC 2: temporal extraction ──────────────────────────────────────────

    def extract_temporal_info(
        self,
        text: str,
        doc_date: date,
    ) -> tuple[Optional[date], float, Optional[tuple[date, date]]]:
        """Extract an event date from ``text``.

        Returns (event_date, confidence, date_range).
        * explicit ISO/US/EU dates -> confidence 1.0, no range
        * "N weeks ago" / "yesterday" / "last month" -> confidence ~0.75,
          with a +/-1 unit range
        """
        if not text:
            return None, 0.0, None

        # Explicit date formats (most specific first).
        for pat in _EXPLICIT_DATE_PATTERNS:
            m = pat.search(text)
            if not m:
                continue
            try:
                a, b, c = m.groups()
                if len(a) == 4:                                # YYYY-MM-DD
                    y, mth, d = int(a), int(b), int(c)
                elif pat.pattern.startswith(r"\b(\d{1,2})\.(\d{1,2})"):
                    d, mth, y = int(a), int(b), int(c)          # d.m.y (European)
                else:
                    mth, d, y = int(a), int(b), int(c)          # m/d/y (US)
                event = date(y, mth, d)
                return event, 1.0, None
            except (ValueError, TypeError):
                continue

        # Numeric relative phrases.
        m = _RELATIVE_PATTERN.search(text)
        if m:
            n = int(m.group(1))
            unit = m.group(2).lower()
            days = n * _RELATIVE_UNITS[unit]
            event = doc_date - timedelta(days=days)
            span = max(1, _RELATIVE_UNITS[unit] // 2)
            return event, 0.75, (event - timedelta(days=span),
                                  event + timedelta(days=span))

        # Named relative phrases.
        lower = text.lower()
        for phrase, days in _NAMED_RELATIVE.items():
            if phrase in lower:
                event = doc_date - timedelta(days=days)
                span = max(1, days // 2 or 1)
                return event, 0.80, (event - timedelta(days=span),
                                      event + timedelta(days=span))

        return None, 0.0, None

    # ── EC 6: unit normalization ───────────────────────────────────────────

    def normalize_measurement(
        self,
        value: float,
        unit: Optional[str],
        entity: str,
    ) -> tuple[float, str]:
        """Normalize a measurement to the canonical unit for ``entity``.

        If no conversion table exists, returns the input unchanged.
        """
        if value is None:
            raise ValueError("value must not be None")

        canon_entity = (entity or "").strip().lower()
        unit_key = (unit or "").strip().lower()

        table = _UNIT_CONVERSIONS.get(canon_entity)
        if not table:
            # fallback: some entities alias (e.g. "HbA1c" -> "hba1c") — already
            # lowercased above; no other aliasing needed here.
            return float(value), unit or ""

        if unit_key not in table:
            # Unknown unit — return raw value but signal canonical unit as target.
            canonical = next(iter(table.values()))[0]
            return float(value), canonical

        target_unit, fn = table[unit_key]
        try:
            return float(fn(float(value))), target_unit
        except Exception as exc:   # pragma: no cover
            logger.warning(f"normalize_measurement failed ({entity} {value} {unit}): {exc}")
            return float(value), unit or target_unit

    # ── EC 12: OCR quality (delegate) ──────────────────────────────────────

    def validate_ocr_quality(self, text: str, image_path: Optional[str] = None) -> dict:
        return self._ocr.validate_ocr_quality(text, image_path)

    # ── Orchestration ──────────────────────────────────────────────────────

    def extract(self, text: str, specialty: str = "general",
                doc_date: Optional[date] = None) -> dict[str, Any]:
        """Run the underlying extractor and decorate entities with metadata.

        Returns a dict with keys:
            entities:         list of annotated entity dicts
            raw_extraction:   the ExtractionOutput-like object
            ocr:              ocr validator report
        """
        doc_date = doc_date or datetime.utcnow().date()

        ocr_report = self._ocr.validate_ocr_quality(text)

        if self._extractor is None:
            raw = None
            names: list[tuple[str, str]] = []
        else:
            raw = self._extractor.extract(text, specialty=specialty)
            names = self._collect_names(raw)

        annotated = []
        for name, entity_type in names:
            ann = self.annotate(name, text, doc_date)
            annotated.append({
                "entity_name": name,
                "entity_type": entity_type,
                "is_negated": ann.is_negated,
                "negation_type": ann.negation_type,
                "subject": ann.subject,
                "subject_confidence": ann.subject_confidence,
                "certainty": ann.certainty,
                "certainty_confidence": ann.certainty_confidence,
                "event_date": ann.event_date.isoformat() if ann.event_date else None,
                "temporal_confidence": ann.temporal_confidence,
                "ocr_confidence": ocr_report.get("confidence", 1.0),
            })

        return {
            "entities": annotated,
            "raw_extraction": raw,
            "ocr": ocr_report,
        }

    def annotate(self, entity: str, text: str, doc_date: date) -> EntityAnnotation:
        neg, neg_type = self.detect_negation(entity, text)
        subject, subj_conf = self.identify_subject(entity, text)
        certainty, cert_conf = self.assess_certainty(entity, text)
        event_date, temp_conf, dr = self.extract_temporal_info(text, doc_date)
        return EntityAnnotation(
            entity=entity,
            is_negated=neg,
            negation_type=neg_type,
            subject=subject,
            subject_confidence=subj_conf,
            certainty=certainty,
            certainty_confidence=cert_conf,
            event_date=event_date,
            temporal_confidence=temp_conf,
            date_range=dr,
        )

    # ── internals ──────────────────────────────────────────────────────────

    @staticmethod
    def _collect_names(raw) -> list[tuple[str, str]]:
        names: list[tuple[str, str]] = []
        if raw is None:
            return names
        for d in getattr(raw, "diagnoses", []):
            names.append((getattr(d, "name", str(d)), "diagnosis"))
        for m in getattr(raw, "medications", []):
            names.append((getattr(m, "name", str(m)), "medication"))
        for t in getattr(raw, "test_results", []):
            names.append((getattr(t, "test_name", str(t)), "test_result"))
        for s in getattr(raw, "symptoms", []):
            names.append((getattr(s, "description", str(s)), "symptom"))
        return names

    @staticmethod
    def _spans_of(needle: str, haystack: str) -> list[tuple[int, int]]:
        if not needle:
            return []
        spans = []
        pattern = re.compile(r"\b" + re.escape(needle) + r"\b", re.I)
        for m in pattern.finditer(haystack):
            spans.append((m.start(), m.end()))
        if not spans:
            # fall back to case-insensitive substring if word boundary fails
            low = haystack.lower()
            idx = 0
            target = needle.lower()
            while True:
                pos = low.find(target, idx)
                if pos == -1:
                    break
                spans.append((pos, pos + len(target)))
                idx = pos + len(target)
        return spans

    @staticmethod
    def _window_before(text: str, span: tuple[int, int], size: int) -> str:
        start = max(0, span[0] - size)
        return text[start:span[0]]

    @staticmethod
    def _sentence_containing(text: str, span: tuple[int, int]) -> str:
        # Sentence boundary = . ! ? or newline. Use the span's midpoint.
        mid = (span[0] + span[1]) // 2
        left = max(
            (text.rfind(ch, 0, mid) for ch in (".", "!", "?", "\n")),
            default=-1,
        )
        right_candidates = [text.find(ch, mid) for ch in (".", "!", "?", "\n")]
        right_candidates = [c for c in right_candidates if c != -1]
        right = min(right_candidates) if right_candidates else len(text)
        return text[left + 1:right].strip()
