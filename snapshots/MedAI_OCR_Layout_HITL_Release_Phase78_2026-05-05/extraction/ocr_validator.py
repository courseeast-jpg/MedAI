"""
MedAI — OCR Validator (Phase 6, EDGE CASE 12).

Detect common OCR artefacts that degrade downstream extraction:

    1. Number / letter confusion        ("50Omg", "l00", "S0")
    2. Medical term misspellings         (edit-distance fuzzy matching)
    3. Excessive special characters      (>10% of total chars)
    4. Missing spaces                    (words >20 chars, concatenations)
    5. Inconsistent line spacing         (merged lines or huge gaps)

The validator does not touch images directly; it works on the OCR text and
optionally accepts the source image path for logging.
"""
from __future__ import annotations

import re
from difflib import get_close_matches
from pathlib import Path
from typing import Any, Optional


# ── Small reference dictionary of common medical terms ──────────────────────
# Intentionally modest — the goal is to catch obvious misspellings, not to
# replace a proper medical spell-checker. Additional terms can be merged in
# at construction time.
_MEDICAL_TERMS: set[str] = {
    "hypertension", "diabetes", "hypothyroidism", "hyperthyroidism",
    "pneumonia", "bronchitis", "asthma", "copd", "emphysema",
    "myocardial", "infarction", "angina", "arrhythmia", "tachycardia",
    "bradycardia", "atrial", "fibrillation", "ventricular",
    "hemoglobin", "hematocrit", "platelets", "leukocytes", "erythrocytes",
    "creatinine", "bilirubin", "cholesterol", "triglycerides",
    "metformin", "lisinopril", "atorvastatin", "amlodipine", "losartan",
    "levothyroxine", "omeprazole", "warfarin", "rivaroxaban", "apixaban",
    "ibuprofen", "paracetamol", "acetaminophen", "aspirin",
    "amoxicillin", "azithromycin", "ciprofloxacin",
    "diagnosis", "prescription", "medication", "treatment", "procedure",
    "patient", "symptoms", "history", "examination", "laboratory",
    "radiology", "ultrasound", "tomography", "resonance", "endoscopy",
    "ibs", "irritable", "bowel", "syndrome", "diverticulitis",
    "oxalates", "crystalluria", "epilepsy", "seizure", "migraine",
    "hba1c", "glucose", "insulin", "sodium", "potassium", "calcium",
}


class OCRValidator:
    """Scores OCR output and surfaces likely errors."""

    _NUMBER_LETTER_PATTERNS: list[tuple[re.Pattern[str], str]] = [
        # Numbers adjacent to an out-of-place capital letter that looks like a digit.
        (re.compile(r"\b\d*[O]\d*[a-zA-Z]*\b"),  "O→0 confusion"),
        (re.compile(r"\b\d*[I]\d+\b"),           "I→1 confusion"),
        (re.compile(r"\b\d*[S]\d+\b"),           "S→5 confusion"),
        (re.compile(r"\b\d*[B]\d+\b"),           "B→8 confusion"),
        # Lowercase 'l' used as digit 1 inside a number-like token.
        (re.compile(r"\b[l]\d+\b"),              "l→1 confusion"),
        (re.compile(r"\b\d+[l]\d+\b"),           "l→1 confusion"),
        # 'metf0rmin', 'amox1cillin' — digit embedded in alphabetic token.
        (re.compile(r"\b[a-zA-Z]{3,}\d[a-zA-Z]{2,}\b"), "digit inside word (likely OCR)"),
    ]

    _SPECIAL_CHAR_RE = re.compile(r"[^\w\s\-\.,:;/()%°µ]")

    def __init__(
        self,
        extra_terms: Optional[set[str]] = None,
        special_char_ratio_threshold: float = 0.10,
        long_word_threshold: int = 20,
    ):
        self._terms = _MEDICAL_TERMS | (extra_terms or set())
        self._special_ratio_max = special_char_ratio_threshold
        self._long_word_max = long_word_threshold

    # ── public API ──────────────────────────────────────────────────────────

    def validate_ocr_quality(
        self,
        text: str,
        image_path: Optional[str | Path] = None,
    ) -> dict[str, Any]:
        """Return OCR quality report.

        Returns:
            {
                'confidence':      float 0.0..1.0,
                'errors_detected': bool,
                'error_examples':  list[str]  (first 5),
                'checks':          dict[str, dict]  detailed breakdown,
                'image_path':      str | None,
            }
        """
        if not text:
            return {
                "confidence": 0.0,
                "errors_detected": True,
                "error_examples": ["empty text"],
                "checks": {"empty": {"failed": True}},
                "image_path": str(image_path) if image_path else None,
            }

        checks: dict[str, dict[str, Any]] = {}
        error_examples: list[str] = []

        # 1. Number/letter confusion
        nl = self._check_number_letter(text)
        checks["number_letter_confusion"] = nl
        error_examples.extend(nl.get("examples", []))

        # 2. Medical term misspellings
        sp = self._check_misspellings(text)
        checks["misspellings"] = sp
        error_examples.extend(sp.get("examples", []))

        # 3. Excessive special characters
        sc = self._check_special_chars(text)
        checks["special_characters"] = sc
        if sc["failed"]:
            error_examples.append(f"special-char ratio {sc['ratio']:.2%}")

        # 4. Missing spaces (very long tokens)
        ms = self._check_long_words(text)
        checks["missing_spaces"] = ms
        error_examples.extend(ms.get("examples", []))

        # 5. Inconsistent line spacing
        ls = self._check_line_spacing(text)
        checks["line_spacing"] = ls
        if ls["failed"]:
            error_examples.append(ls.get("detail", "inconsistent line spacing"))

        # Overall confidence — each failed check subtracts a weighted penalty.
        weights = {
            "number_letter_confusion": 0.30,
            "misspellings":            0.25,
            "special_characters":      0.20,
            "missing_spaces":          0.15,
            "line_spacing":            0.10,
        }
        confidence = 1.0
        for name, weight in weights.items():
            c = checks.get(name, {})
            if c.get("failed"):
                # scale penalty by severity if available
                severity = min(1.0, float(c.get("severity", 1.0)))
                confidence -= weight * severity
        confidence = max(0.0, min(1.0, confidence))

        errors_detected = any(c.get("failed") for c in checks.values())

        return {
            "confidence": confidence,
            "errors_detected": errors_detected,
            "error_examples": error_examples[:5],
            "checks": checks,
            "image_path": str(image_path) if image_path else None,
        }

    # ── individual checks ───────────────────────────────────────────────────

    def _check_number_letter(self, text: str) -> dict[str, Any]:
        matches: list[str] = []
        for pat, label in self._NUMBER_LETTER_PATTERNS:
            for m in pat.finditer(text):
                tok = m.group(0)
                # Exclude tokens that are obviously intentional abbreviations
                # (e.g. 'O2', 'CO2', 'B12').
                if tok in {"O2", "CO2", "B12", "B6", "K1", "K2"}:
                    continue
                matches.append(f"{tok} ({label})")
                if len(matches) >= 20:
                    break
            if len(matches) >= 20:
                break

        failed = len(matches) > 0
        # Even a single number/letter confusion is suspicious; scale aggressively.
        severity = 0.0 if not failed else min(1.0, 0.5 + 0.2 * len(matches))
        return {
            "failed": failed,
            "count": len(matches),
            "severity": severity,
            "examples": matches[:5],
        }

    def _check_misspellings(self, text: str) -> dict[str, Any]:
        tokens = re.findall(r"[A-Za-zА-Яа-я]{4,}", text)
        misspelled: list[str] = []
        seen: set[str] = set()
        for tok in tokens:
            lower = tok.lower()
            if lower in seen or lower in self._terms:
                continue
            seen.add(lower)
            near = get_close_matches(lower, self._terms, n=1, cutoff=0.85)
            if near and near[0] != lower:
                misspelled.append(f"{tok} ≈ {near[0]}")
                if len(misspelled) >= 20:
                    break

        failed = len(misspelled) > 0
        severity = min(1.0, len(misspelled) / 5.0)
        return {
            "failed": failed,
            "count": len(misspelled),
            "severity": severity,
            "examples": misspelled[:5],
        }

    def _check_special_chars(self, text: str) -> dict[str, Any]:
        total = max(1, len(text))
        special = len(self._SPECIAL_CHAR_RE.findall(text))
        ratio = special / total
        failed = ratio > self._special_ratio_max
        severity = min(1.0, (ratio - self._special_ratio_max) / 0.10) if failed else 0.0
        return {
            "failed": failed,
            "ratio": ratio,
            "count": special,
            "severity": max(0.0, severity),
        }

    def _check_long_words(self, text: str) -> dict[str, Any]:
        long_tokens = [w for w in re.findall(r"\S{%d,}" % (self._long_word_max + 1), text)]
        # Exclude URLs and obvious hex IDs.
        filtered = [
            w for w in long_tokens
            if not w.startswith(("http://", "https://"))
            and not re.fullmatch(r"[A-Fa-f0-9]{20,}", w)
        ]
        failed = len(filtered) > 0
        severity = min(1.0, len(filtered) / 3.0)
        return {
            "failed": failed,
            "count": len(filtered),
            "severity": severity,
            "examples": filtered[:5],
        }

    def _check_line_spacing(self, text: str) -> dict[str, Any]:
        lines = text.split("\n")
        if len(lines) < 4:
            return {"failed": False, "count": 0, "severity": 0.0}
        non_empty_lens = [len(l) for l in lines if l.strip()]
        if not non_empty_lens:
            return {"failed": False, "count": 0, "severity": 0.0}
        avg = sum(non_empty_lens) / len(non_empty_lens)

        merged = sum(1 for l in non_empty_lens if l > avg * 3 and l > 200)
        gaps = 0
        run = 0
        for l in lines:
            if l.strip() == "":
                run += 1
                if run >= 4:
                    gaps += 1
                    run = 0
            else:
                run = 0

        failed = merged > 0 or gaps > 0
        severity = min(1.0, (merged + gaps) / 3.0)
        detail = []
        if merged:
            detail.append(f"{merged} merged line(s)")
        if gaps:
            detail.append(f"{gaps} excessive gap(s)")
        return {
            "failed": failed,
            "count": merged + gaps,
            "severity": severity,
            "detail": ", ".join(detail) if detail else "",
        }
