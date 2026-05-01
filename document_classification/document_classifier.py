"""Phase 43 — Deterministic document-type and language pre-classifier.

Pure-regex, no model dependencies. Decides whether lab normalization
should run for a given document, distinguishes prescriptions and
microbiology/PCR reports from standard lab reports, and recommends
language-aware OCR when Cyrillic or pseudo-Cyrillic content dominates.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass
from typing import Any


# ---------------------------------------------------------------------------
# Token sets — English / Latin
# ---------------------------------------------------------------------------

_LAB_TOKENS_EN = re.compile(
    r"\b(?:reference\s+range|reference\s+interval|mg/dL|g/dL|mmol/L|"
    r"x10E3/uL|x10E6/uL|x10E6/mL|/hpf|CFU/mL|IU/L|U/L|"
    r"cbc|complete\s+blood\s+count|wbc|rbc|hemoglobin|hematocrit|platelets?|"
    r"urinalysis|specific\s+gravity|ketones|nitrite|leukocytes?|"
    r"glucose|cholesterol|creatinine|sodium|potassium|calcium|albumin|bilirubin|"
    r"hdl|ldl|triglycerides?|tsh|hba1c|alt|ast)\b",
    re.I,
)

_LAB_INDICATOR_RE = re.compile(
    r"\b(?:result|results|units?|specimen|test\s+name|reference\s+range)\b",
    re.I,
)

_MICROBIOLOGY_TOKENS_EN = re.compile(
    r"\b(?:pcr|androflor|microbiology|microflora|"
    r"candida|ureaplasma|enterobacteriaceae|enterococcus|chlamydia|mycoplasma|"
    r"trichomonas|neisseria)\b",
    re.I,
)

_PRESCRIPTION_TOKENS_EN = re.compile(
    r"\b(?:rx|prescription|tablet|tablets|capsule|capsules|"
    r"suppository|suppositories|"
    r"\d+\s*mg\s+(?:daily|bid|tid|qid|po|prn))\b",
    re.I,
)


# ---------------------------------------------------------------------------
# Token sets — UTF-8 Cyrillic (when OCR returns proper Cyrillic)
# ---------------------------------------------------------------------------

_PRESCRIPTION_TOKENS_RU = re.compile(
    # Specific medication brand/INN names
    r"(?:диклофенак|метронидазол|левофлоксацин|тамсулозин|флуконазол|"
    r"линекс|витапрост|"
    # Dosage form words
    r"свеч[аиеуюя]|таблетк|капсул|"
    # Dose units (Russian milligram/milliliter)
    r"\b\d+\s*мг\b|\b\d+\s*мл\b|\bмг\b|\bмл\b|"
    # Generic prescription / instruction vocabulary
    r"рецепт|назначени|препарат|лекарств|принимать|дозировк|"
    # Frequency / timing instructions
    r"раз\s+в\s+день|после\s+еды|до\s+еды|на\s+ночь|"
    # Clinical context words common on prescription forms
    r"врач|аптек|пациент|диагноз|рекомендаци)",
    re.I,
)

_MICROBIOLOGY_TOKENS_RU = re.compile(
    r"(?:пцр|андрофлор|микрофлор|днк\b|"
    r"выявлено|не\s+выявлено|обнаружено|"
    r"микроорганизм|бактери|урогенитал|полимеразн)",
    re.I,
)


# ---------------------------------------------------------------------------
# Token sets — Latin homoglyphs of mangled Cyrillic (OCR failures)
#
# These match the byte-level Latin output that OCR produces when reading
# Cyrillic without a Cyrillic language pack. Each pattern targets a Russian
# medical term whose mangled form is distinctive enough not to collide with
# real English text.
# ---------------------------------------------------------------------------

_PRESCRIPTION_PSEUDO_RU = re.compile(
    # Number followed by "Mr" / "MR" — "100Мг" mangled to "100Mr" (милли-граммы)
    r"(?:\b\d+\s*Mr\b)"
    # "Свеч-" prefix for suppositories
    r"|(?:\bC[Bs]e[LJ:cqWl]+(?:[Hh]|e))"
    # "Дикло-" prefix (Diclofenac)
    r"|(?:\bIUI?KJio)"
    # "Метрон-" prefix (Metronidazole)
    r"|(?:\bMeTpO[HuhI])"
    # "Флукон-" / "Левофлокс-" / "Тамсулозин"
    r"|(?:\b<[Dl]>?nyKO)"
    r"|(?:\bJ1e[Bs][Oo][crpf¢]nO[Kk])"
    r"|(?:\bTaMcyJio)"
    # "Линекс" — Linex
    r"|(?:\bJ1[Hh]HeKC)"
    # "Препарат" header → "PEIIAPAT" / "l1PEIIAPAT" / "l1PEnAPAT"
    r"|(?:\b[lI1]?[lI1]?PE[Iil]+[ApA]+PAT)"
    # "ДАТА" header → "L(ATA"  /  "JJATA" / ",l(ATA"
    r"|(?:\bL\(ATA\b|\bJJ_ATA\b|\b,l\(ATA\b)",
)

_MICROBIOLOGY_PSEUDO_RU = re.compile(
    # "Андрофлор" → "AHApocpnop" or "AHA[Pp]oc[fr]nop"
    r"(?:\bAH[A,]?[Pp]o[cs][pfrPF]?nop)"
    # "Микрофлора" → "MIKpocpnop" / "MHKpocpnop"
    r"|(?:\bM[VlIiHHvr]+Kpo[cs][pfrPF]?nop)"
    # "ПЦР" → "nl..tp" / "nu.p" / "nliP" — short, very distinctive
    r"|(?:\bnl?[\.,_]+[Llqp1tr]+p\b)"
    # "урогенитальн-" → "yporeH[hH]Tail[bB]H"
    r"|(?:\byporeH[Hhi]+T[ai][il1L]+[bB]?H)"
    # "полимеразн-" → "nOIIIIIMepa3H" / "noilllllMepa3H"
    r"|(?:\bn[OoIli01]+[il1lI]+Mepa3H)"
    # "уретропростатит" → "ypeTponpO?CTaT[Hi]+T"
    r"|(?:\bypeTponp[Oo0]?CTa[TtHh][HiIl1]+T)"
    # "микрофлор-" alternate "MliKpoc[pf]nop" guarded above
)


# Pseudo-Russian general indicator: dense special-char/digit substitution into
# Latin script (a strong signal that text is OCR-mangled non-English).
_HOMOGLYPH_DENSITY_RE = re.compile(r"[A-Za-z](?:[¢®©§]|\d)[A-Za-z]")


# ---------------------------------------------------------------------------
# Language detection helpers
# ---------------------------------------------------------------------------

def cyrillic_ratio(text: str) -> float:
    if not text:
        return 0.0
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    cyr = sum(1 for c in letters if "CYRILLIC" in unicodedata.name(c, ""))
    return cyr / len(letters)


def latin_ratio(text: str) -> float:
    if not text:
        return 0.0
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    lat = sum(1 for c in letters if "LATIN" in unicodedata.name(c, ""))
    return lat / len(letters)


def _detect_language(text: str, *, pseudo_ru_signals: int) -> str:
    cyr = cyrillic_ratio(text)
    lat = latin_ratio(text)
    if cyr >= 0.20 and lat >= 0.20:
        return "mixed"
    if cyr >= 0.20:
        return "ru"
    # OCR-mangled Russian: Latin-dominant bytes but pseudo-RU markers fire
    if pseudo_ru_signals >= 2 and lat >= 0.30:
        return "mixed"
    if lat >= 0.30:
        return "en"
    return "unknown"


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------

DOCUMENT_TYPES = {
    "lab_report",
    "microbiology_pcr_report",
    "prescription",
    "clinical_note",
    "imaging_report",
    "unknown_medical",
    "unknown_nonmedical",
}


@dataclass(frozen=True)
class DocumentClassification:
    document_type: str
    confidence: float
    language_hint: str
    should_run_lab_normalization: bool
    should_run_prescription_path: bool
    should_recommend_language_aware_ocr: bool
    review_reason: str | None
    evidence: list[str]
    warnings: list[str]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Classifier entry point
# ---------------------------------------------------------------------------

def classify_document(
    text: str,
    *,
    ocr_metadata: dict[str, Any] | None = None,
) -> DocumentClassification:
    source = text or ""
    ocr_metadata = dict(ocr_metadata or {})

    cyr = cyrillic_ratio(source)
    lat = latin_ratio(source)

    lab_en_count = len(_LAB_TOKENS_EN.findall(source))
    lab_indicator_count = len(_LAB_INDICATOR_RE.findall(source))
    micro_en_count = len(_MICROBIOLOGY_TOKENS_EN.findall(source))
    rx_en_count = len(_PRESCRIPTION_TOKENS_EN.findall(source))
    rx_ru_count = len(_PRESCRIPTION_TOKENS_RU.findall(source))
    rx_pseudo_count = len(_PRESCRIPTION_PSEUDO_RU.findall(source))
    micro_ru_count = len(_MICROBIOLOGY_TOKENS_RU.findall(source))
    micro_pseudo_count = len(_MICROBIOLOGY_PSEUDO_RU.findall(source))

    pseudo_ru_signals = rx_pseudo_count + micro_pseudo_count
    language = _detect_language(source, pseudo_ru_signals=pseudo_ru_signals)

    rx_signal = rx_ru_count + rx_en_count + rx_pseudo_count
    micro_signal = micro_en_count + micro_ru_count + micro_pseudo_count
    # Strong lab signal = specific lab tokens only. Weak signal = generic
    # words like "result" that appear in many medical doc types.
    lab_signal_strong = lab_en_count
    lab_signal = lab_en_count + lab_indicator_count

    metadata: dict[str, Any] = {
        "cyrillic_ratio": round(cyr, 3),
        "latin_ratio": round(lat, 3),
        "lab_token_count_en": lab_en_count,
        "lab_indicator_count": lab_indicator_count,
        "microbiology_token_count_en": micro_en_count,
        "microbiology_token_count_ru": micro_ru_count,
        "microbiology_pseudo_count": micro_pseudo_count,
        "prescription_token_count_en": rx_en_count,
        "prescription_token_count_ru": rx_ru_count,
        "prescription_pseudo_count": rx_pseudo_count,
        "rx_signal_total": rx_signal,
        "micro_signal_total": micro_signal,
        "lab_signal_total": lab_signal,
        "text_length": len(source.strip()),
        "ocr_metadata": ocr_metadata,
    }

    evidence: list[str] = []
    warnings: list[str] = []
    document_type = "unknown_medical"
    confidence = 0.30
    review_reason: str | None = None

    # Rule 1: clear lab report (specific English lab tokens dominate and
    # outweigh microbiology/prescription evidence)
    if (
        lab_signal_strong >= 2
        and rx_signal < 2
        and micro_pseudo_count == 0
        and lab_signal_strong > micro_signal
    ):
        document_type = "lab_report"
        confidence = min(0.55 + 0.07 * lab_signal, 0.95)
        evidence.append(f"lab_tokens_en={lab_en_count}")
        if lab_indicator_count:
            evidence.append(f"lab_indicators={lab_indicator_count}")
        if micro_en_count >= 1:
            evidence.append("microbiology_subtype_present")
        review_reason = "lab_report_detected"

    # Rule 2: prescription dominates (no specific lab tokens) — RU or EN
    elif rx_signal >= 2 and lab_signal_strong == 0:
        document_type = "prescription"
        confidence = min(0.55 + 0.08 * rx_signal, 0.95)
        evidence.append(f"prescription_terms={rx_signal}")
        if rx_ru_count or rx_pseudo_count:
            evidence.append("russian_or_pseudo_russian_prescription_markers")
            warnings.append("non_english_source")
        review_reason = "document_type_prescription_not_lab"

    # Rule 3: microbiology / PCR (English or pseudo-Russian).
    # Generic indicator words like "result" do not disqualify this rule.
    elif micro_signal >= 2 and lab_signal_strong == 0:
        document_type = "microbiology_pcr_report"
        confidence = min(0.55 + 0.08 * micro_signal, 0.92)
        evidence.append(f"microbiology_terms={micro_signal}")
        if micro_ru_count or micro_pseudo_count:
            evidence.append("russian_or_pseudo_russian_pcr_markers")
            warnings.append("non_english_source")
        review_reason = "microbiology_pcr_report_detected"

    # Rule 4: Cyrillic-dominant fallback
    elif (cyr >= 0.20 or pseudo_ru_signals >= 1):
        if rx_signal >= 1 or rx_pseudo_count >= 1:
            document_type = "prescription"
            confidence = 0.50
            evidence.append("cyrillic_dominant_with_rx_marker")
            review_reason = "document_type_prescription_not_lab"
        elif micro_signal >= 1:
            document_type = "microbiology_pcr_report"
            confidence = 0.50
            evidence.append("cyrillic_dominant_with_micro_marker")
            review_reason = "microbiology_pcr_report_detected"
        else:
            document_type = "unknown_medical"
            confidence = 0.35
            evidence.append("cyrillic_dominant_no_typed_markers")
            warnings.append("non_english_source")
            review_reason = "unknown_document_type"

    # Rule 5: nothing strong — keep safe default
    else:
        document_type = "unknown_medical"
        confidence = 0.30
        evidence.append("no_strong_signals")
        review_reason = "unknown_document_type"

    # Routing decisions
    # Lab normalization runs for: lab_report, unknown (preserve existing behavior),
    # and English microbiology (parser already handles paired microbiology rows).
    should_run_lab_normalization = (
        document_type == "lab_report"
        or document_type == "unknown_medical"
        or (document_type == "microbiology_pcr_report" and language == "en")
    )
    should_run_prescription_path = document_type == "prescription"

    should_recommend_language_aware_ocr = bool(
        cyr >= 0.20
        or pseudo_ru_signals >= 1
        or language in {"ru", "mixed"}
    )

    if confidence < 0.50:
        warnings.append("low_confidence_document_type")

    return DocumentClassification(
        document_type=document_type,
        confidence=round(confidence, 3),
        language_hint=language,
        should_run_lab_normalization=should_run_lab_normalization,
        should_run_prescription_path=should_run_prescription_path,
        should_recommend_language_aware_ocr=should_recommend_language_aware_ocr,
        review_reason=review_reason,
        evidence=evidence,
        warnings=warnings,
        metadata=metadata,
    )
