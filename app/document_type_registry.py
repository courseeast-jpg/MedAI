"""Metadata-only document family registry for Run & Review labels.

The registry deliberately emits safe cue keys and family labels only. It does
not parse values, interpret medical meaning, adjust confidence, or change
review/acceptance gates.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable


LAB_RESULT_LABEL = "Lab result"
URINALYSIS_LABEL = "Urinalysis"
IMAGING_REPORT_LABEL = "Imaging report"
TREATMENT_PLAN_LABEL = "Treatment plan"
MEDICATION_PLAN_LABEL = "Medication plan"
CLINICAL_NOTE_LABEL = "Clinical note"
DISCHARGE_SUMMARY_LABEL = "Discharge summary"
REFERRAL_ORDER_LABEL = "Referral / Order"
PROCEDURE_REPORT_LABEL = "Procedure report"
PATHOLOGY_REPORT_LABEL = "Pathology report"
ADMINISTRATIVE_INSURANCE_LABEL = "Administrative / Insurance"
UNKNOWN_DOCUMENT_LABEL = "Unknown"

SUPPORTED_DOCUMENT_FAMILIES = (
    LAB_RESULT_LABEL,
    IMAGING_REPORT_LABEL,
    TREATMENT_PLAN_LABEL,
    MEDICATION_PLAN_LABEL,
    CLINICAL_NOTE_LABEL,
    DISCHARGE_SUMMARY_LABEL,
    REFERRAL_ORDER_LABEL,
    PROCEDURE_REPORT_LABEL,
    PATHOLOGY_REPORT_LABEL,
    ADMINISTRATIVE_INSURANCE_LABEL,
    UNKNOWN_DOCUMENT_LABEL,
)

SUPPORTED_LANGUAGE_PACKS = ("english", "russian", "polish", "albanian")


@dataclass(frozen=True)
class FamilyRule:
    label: str
    cue_groups: dict[str, dict[str, tuple[str, ...]]]
    threshold: int
    required_any: tuple[str, ...] = ()
    subtype_label: str | None = None


DOCUMENT_FAMILY_REGISTRY: dict[str, FamilyRule] = {
    LAB_RESULT_LABEL: FamilyRule(
        label=LAB_RESULT_LABEL,
        threshold=3,
        required_any=("lab_report_heading", "specimen_or_biomaterial", "result_or_report", "table_header"),
        cue_groups={
            "english": {
                "lab_report_heading": ("lab result", "laboratory result", "laboratory report", "test result"),
                "specimen_or_biomaterial": ("specimen", "specimen type", "sample", "biomaterial"),
                "result_or_report": ("result", "reported", "report"),
                "table_header": ("component", "analyte", "value", "units", "flag"),
                "reference_range_pattern": ("reference range", "reference interval"),
            },
            "russian": {
                "lab_report_heading": (
                    "\u043b\u0430\u0431\u043e\u0440\u0430\u0442\u043e\u0440\u043d",
                    "\u0430\u043d\u0430\u043b\u0438\u0437",
                    "\u0438\u0441\u0441\u043b\u0435\u0434\u043e\u0432\u0430\u043d\u0438\u0435",
                ),
                "specimen_or_biomaterial": (
                    "\u0431\u0438\u043e\u043c\u0430\u0442\u0435\u0440\u0438\u0430\u043b",
                    "\u043c\u0430\u0442\u0435\u0440\u0438\u0430\u043b",
                    "\u0441\u044b\u0432\u043e\u0440\u043e\u0442\u043a\u0430",
                    "\u043a\u0440\u043e\u0432\u044c",
                ),
                "result_or_report": (
                    "\u0440\u0435\u0437\u0443\u043b\u044c\u0442\u0430\u0442",
                    "\u043e\u0442\u0447\u0435\u0442",
                    "\u043f\u0440\u043e\u0442\u043e\u043a\u043e\u043b",
                ),
                "table_header": (
                    "\u043f\u043e\u043a\u0430\u0437\u0430\u0442\u0435\u043b\u044c",
                    "\u0437\u043d\u0430\u0447\u0435\u043d\u0438\u0435",
                    "\u0435\u0434\u0438\u043d\u0438\u0446",
                    "\u043f\u0430\u0440\u0430\u043c\u0435\u0442\u0440",
                ),
                "reference_range_pattern": (
                    "\u0440\u0435\u0444\u0435\u0440\u0435\u043d\u0441",
                    "\u043d\u043e\u0440\u043c\u0430",
                    "\u0438\u043d\u0442\u0435\u0440\u0432\u0430\u043b",
                ),
            },
            "polish": {
                "lab_report_heading": ("wynik badania", "wyniki badan", "laboratorium", "badanie laboratoryjne"),
                "specimen_or_biomaterial": ("material", "probka", "krew", "surowica"),
                "result_or_report": ("wynik", "sprawozdanie"),
                "table_header": ("parametr", "wartosc", "jednostka"),
                "reference_range_pattern": ("zakres referencyjny", "norma"),
            },
            "albanian": {
                "lab_report_heading": ("rezultat laboratorik", "analiza laboratorike", "laborator"),
                "specimen_or_biomaterial": ("mostra", "material", "gjak"),
                "result_or_report": ("rezultat", "raport"),
                "table_header": ("parameter", "vlere", "njesi"),
                "reference_range_pattern": ("interval referimi", "vlera normale"),
            },
        },
    ),
    URINALYSIS_LABEL: FamilyRule(
        label=LAB_RESULT_LABEL,
        subtype_label=URINALYSIS_LABEL,
        threshold=2,
        required_any=("urinalysis_terms",),
        cue_groups={
            "english": {
                "urinalysis_terms": ("urinalysis", "urine", "specific gravity", "leukocyte esterase"),
                "table_header": ("protein", "glucose", "ketones", "nitrite"),
            },
            "russian": {
                "urinalysis_terms": (
                    "\u0430\u043d\u0430\u043b\u0438\u0437 \u043c\u043e\u0447\u0438",
                    "\u043e\u0430\u043c",
                    "\u043c\u043e\u0447\u0430",
                    "\u0443\u0434\u0435\u043b\u044c\u043d\u044b\u0439 \u0432\u0435\u0441",
                ),
                "table_header": (
                    "\u0431\u0435\u043b\u043e\u043a",
                    "\u043a\u0435\u0442\u043e\u043d",
                    "\u043d\u0438\u0442\u0440\u0438\u0442",
                    "\u043b\u0435\u0439\u043a\u043e\u0446\u0438\u0442",
                ),
            },
            "polish": {
                "urinalysis_terms": ("badanie moczu", "mocz"),
                "table_header": ("bialko", "glukoza", "ketony"),
            },
            "albanian": {
                "urinalysis_terms": ("analiza e urines", "urine"),
                "table_header": ("proteine", "glukoze", "ketone"),
            },
        },
    ),
    IMAGING_REPORT_LABEL: FamilyRule(
        label=IMAGING_REPORT_LABEL,
        threshold=2,
        required_any=("imaging_modality",),
        cue_groups={
            "english": {
                "imaging_modality": ("mri", "ct", "ultrasound", "x ray", "x-ray", "radiograph"),
                "imaging_device_header": ("scanner", "machine", "device"),
                "imaging_description_section": ("description", "technique", "findings"),
                "imaging_conclusion_section": ("impression", "conclusion"),
                "radiology_series_wording": ("series", "slice", "sequence"),
            },
            "russian": {
                "imaging_modality": (
                    "\u043c\u0440\u0442",
                    "\u043c\u0440 \u0442\u043e\u043c\u043e\u0433\u0440\u0430\u0444",
                    "\u043a\u0442",
                    "\u0443\u0437\u0438",
                    "\u0440\u0435\u043d\u0442\u0433\u0435\u043d",
                ),
                "imaging_device_header": ("\u0430\u043f\u043f\u0430\u0440\u0430\u0442", "\u0441\u043a\u0430\u043d\u0435\u0440"),
                "imaging_description_section": ("\u043e\u043f\u0438\u0441\u0430\u043d\u0438\u0435",),
                "imaging_conclusion_section": ("\u0437\u0430\u043a\u043b\u044e\u0447\u0435\u043d\u0438\u0435",),
                "radiology_series_wording": (
                    "\u0442\u043e\u043c\u043e\u0433\u0440\u0430\u043c\u043c",
                    "\u0441\u0435\u0440\u0438\u044f",
                    "\u0441\u0440\u0435\u0437",
                ),
            },
            "polish": {
                "imaging_modality": ("mri", "rezonans magnetyczny", "tomografia", "usg", "rentgen"),
                "imaging_device_header": ("aparat", "skaner"),
                "imaging_description_section": ("opis",),
                "imaging_conclusion_section": ("wniosek", "podsumowanie"),
                "radiology_series_wording": ("seria", "przekroj"),
            },
            "albanian": {
                "imaging_modality": ("mri", "rezonance magnetike", "skaner", "ultratinguj", "radiografi"),
                "imaging_device_header": ("aparat", "pajisje"),
                "imaging_description_section": ("pershkrim",),
                "imaging_conclusion_section": ("perfundim", "konkluzion"),
                "radiology_series_wording": ("seri", "prerje"),
            },
        },
    ),
    TREATMENT_PLAN_LABEL: FamilyRule(
        label=TREATMENT_PLAN_LABEL,
        threshold=2,
        required_any=("treatment_recommendation_section", "administration_schedule_pattern"),
        cue_groups={
            "english": {
                "treatment_recommendation_section": ("treatment plan", "recommendations", "therapy plan"),
                "administration_schedule_pattern": ("schedule", "course", "morning", "evening"),
                "date_grid": ("date", "day"),
                "physiotherapy_section": ("physiotherapy", "procedure"),
            },
            "russian": {
                "treatment_recommendation_section": (
                    "\u043f\u043b\u0430\u043d \u043b\u0435\u0447\u0435\u043d\u0438\u044f",
                    "\u0441\u0445\u0435\u043c\u0430 \u043b\u0435\u0447\u0435\u043d\u0438\u044f",
                    "\u0440\u0435\u043a\u043e\u043c\u0435\u043d\u0434\u0430\u0446\u0438\u0438",
                ),
                "administration_schedule_pattern": (
                    "\u0441\u0445\u0435\u043c\u0430",
                    "\u0433\u0440\u0430\u0444\u0438\u043a",
                    "\u0440\u0430\u0441\u043f\u0438\u0441\u0430\u043d\u0438\u0435",
                    "\u043a\u0443\u0440\u0441",
                ),
                "date_grid": ("\u0434\u0430\u0442\u0430",),
                "physiotherapy_section": (
                    "\u0444\u0438\u0437\u0438\u043e\u043f\u0440\u043e\u0446\u0435\u0434\u0443\u0440\u044b",
                    "\u0444\u0438\u0437\u0438\u043e",
                ),
            },
            "polish": {
                "treatment_recommendation_section": ("plan leczenia", "zalecenia"),
                "administration_schedule_pattern": ("schemat", "harmonogram", "kurs"),
                "date_grid": ("data",),
            },
            "albanian": {
                "treatment_recommendation_section": ("plan trajtimi", "rekomandime"),
                "administration_schedule_pattern": ("skeme", "orar", "kurs"),
                "date_grid": ("date",),
            },
        },
    ),
    MEDICATION_PLAN_LABEL: FamilyRule(
        label=MEDICATION_PLAN_LABEL,
        threshold=2,
        required_any=("medication_schedule_header",),
        cue_groups={
            "english": {
                "medication_schedule_header": ("medications", "medicine", "drug schedule"),
                "administration_schedule_pattern": ("take", "dose", "morning", "evening"),
                "date_grid": ("date", "day"),
            },
            "russian": {
                "medication_schedule_header": (
                    "\u043f\u0440\u0435\u043f\u0430\u0440\u0430\u0442\u044b",
                    "\u043b\u0435\u043a\u0430\u0440\u0441\u0442\u0432\u0430",
                    "\u043d\u0430\u0437\u043d\u0430\u0447\u0435\u043d\u0438\u044f",
                ),
                "administration_schedule_pattern": (
                    "\u043f\u0440\u0438\u0435\u043c",
                    "\u0434\u043e\u0437\u0430",
                    "\u0443\u0442\u0440\u043e",
                    "\u0432\u0435\u0447\u0435\u0440",
                ),
                "date_grid": ("\u0434\u0430\u0442\u0430",),
            },
            "polish": {
                "medication_schedule_header": ("leki", "lek", "lista lekow"),
                "administration_schedule_pattern": ("dawka", "rano", "wieczor"),
                "date_grid": ("data",),
            },
            "albanian": {
                "medication_schedule_header": ("barna", "ilaç", "medikamente"),
                "administration_schedule_pattern": ("doze", "mengjes", "mbremje"),
                "date_grid": ("date",),
            },
        },
    ),
    CLINICAL_NOTE_LABEL: FamilyRule(
        label=CLINICAL_NOTE_LABEL,
        threshold=3,
        cue_groups={
            "english": {
                "complaint_section": ("chief complaint", "complaint"),
                "history_section": ("history of present illness", "medical history"),
                "examination_section": ("physical exam", "examination"),
                "assessment_section": ("assessment", "plan"),
            },
            "russian": {
                "complaint_section": ("\u0436\u0430\u043b\u043e\u0431\u044b",),
                "history_section": ("\u0430\u043d\u0430\u043c\u043d\u0435\u0437",),
                "examination_section": ("\u043e\u0441\u043c\u043e\u0442\u0440",),
                "assessment_section": ("\u043e\u0446\u0435\u043d\u043a\u0430",),
            },
            "polish": {
                "complaint_section": ("skargi", "dolegliwosci"),
                "history_section": ("wywiad",),
                "examination_section": ("badanie przedmiotowe", "badanie"),
                "assessment_section": ("ocena",),
            },
            "albanian": {
                "complaint_section": ("ankesa",),
                "history_section": ("anamneza", "histori"),
                "examination_section": ("ekzaminim",),
                "assessment_section": ("vleresim",),
            },
        },
    ),
    DISCHARGE_SUMMARY_LABEL: FamilyRule(
        label=DISCHARGE_SUMMARY_LABEL,
        threshold=2,
        required_any=("discharge_diagnosis_section", "hospital_course_section"),
        cue_groups={
            "english": {
                "admission_discharge_dates": ("admission date", "discharge date"),
                "discharge_diagnosis_section": ("discharge diagnosis",),
                "hospital_course_section": ("hospital course",),
                "discharge_recommendation_section": ("discharge recommendations",),
            },
            "russian": {
                "admission_discharge_dates": (
                    "\u0434\u0430\u0442\u0430 \u043f\u043e\u0441\u0442\u0443\u043f\u043b\u0435\u043d\u0438\u044f",
                    "\u0434\u0430\u0442\u0430 \u0432\u044b\u043f\u0438\u0441\u043a\u0438",
                ),
                "discharge_diagnosis_section": ("\u0434\u0438\u0430\u0433\u043d\u043e\u0437 \u043f\u0440\u0438 \u0432\u044b\u043f\u0438\u0441\u043a\u0435",),
                "hospital_course_section": ("\u0445\u043e\u0434 \u043b\u0435\u0447\u0435\u043d\u0438\u044f",),
                "discharge_recommendation_section": ("\u0440\u0435\u043a\u043e\u043c\u0435\u043d\u0434\u0430\u0446\u0438\u0438 \u043f\u0440\u0438 \u0432\u044b\u043f\u0438\u0441\u043a\u0435",),
            },
            "polish": {
                "admission_discharge_dates": ("data przyjecia", "data wypisu"),
                "discharge_diagnosis_section": ("rozpoznanie przy wypisie",),
                "hospital_course_section": ("przebieg hospitalizacji",),
            },
            "albanian": {
                "admission_discharge_dates": ("data e pranimit", "data e daljes"),
                "discharge_diagnosis_section": ("diagnoza ne dalje",),
                "hospital_course_section": ("rrjedha spitalore",),
            },
        },
    ),
    REFERRAL_ORDER_LABEL: FamilyRule(
        label=REFERRAL_ORDER_LABEL,
        threshold=2,
        cue_groups={
            "english": {
                "referral_heading": ("referral", "order"),
                "ordered_test_or_service": ("requested test", "ordered service"),
                "referring_provider_section": ("referring provider",),
            },
            "russian": {
                "referral_heading": ("\u043d\u0430\u043f\u0440\u0430\u0432\u043b\u0435\u043d\u0438\u0435", "\u0437\u0430\u043a\u0430\u0437"),
                "ordered_test_or_service": ("\u0438\u0441\u0441\u043b\u0435\u0434\u043e\u0432\u0430\u043d\u0438\u0435", "\u0443\u0441\u043b\u0443\u0433\u0430"),
                "referring_provider_section": ("\u043d\u0430\u043f\u0440\u0430\u0432\u0438\u0432\u0448\u0438\u0439 \u0432\u0440\u0430\u0447",),
            },
            "polish": {
                "referral_heading": ("skierowanie", "zlecenie"),
                "ordered_test_or_service": ("badanie", "usluga"),
            },
            "albanian": {
                "referral_heading": ("referim", "urdher"),
                "ordered_test_or_service": ("ekzaminim", "sherbim"),
            },
        },
    ),
    PROCEDURE_REPORT_LABEL: FamilyRule(
        label=PROCEDURE_REPORT_LABEL,
        threshold=2,
        cue_groups={
            "english": {"procedure_heading": ("procedure report", "operative report"), "procedure_description": ("procedure", "technique")},
            "russian": {"procedure_heading": ("\u043f\u0440\u043e\u0442\u043e\u043a\u043e\u043b \u043f\u0440\u043e\u0446\u0435\u0434\u0443\u0440\u044b",), "procedure_description": ("\u043c\u0435\u0442\u043e\u0434\u0438\u043a\u0430",)},
            "polish": {"procedure_heading": ("opis zabiegu", "protokol zabiegu"), "procedure_description": ("procedura", "technika")},
            "albanian": {"procedure_heading": ("raport procedure",), "procedure_description": ("procedure", "teknike")},
        },
    ),
    PATHOLOGY_REPORT_LABEL: FamilyRule(
        label=PATHOLOGY_REPORT_LABEL,
        threshold=2,
        cue_groups={
            "english": {
                "specimen_section": ("specimen",),
                "microscopic_description_section": ("microscopic description",),
                "pathology_conclusion_section": ("pathology diagnosis", "final diagnosis"),
            },
            "russian": {
                "specimen_section": ("\u043c\u0430\u0442\u0435\u0440\u0438\u0430\u043b",),
                "microscopic_description_section": ("\u043c\u0438\u043a\u0440\u043e\u0441\u043a\u043e\u043f\u0438\u0447\u0435\u0441\u043a\u043e\u0435 \u043e\u043f\u0438\u0441\u0430\u043d\u0438\u0435",),
                "pathology_conclusion_section": ("\u043f\u0430\u0442\u043e\u043b\u043e\u0433\u0438\u044f",),
            },
            "polish": {"specimen_section": ("material",), "microscopic_description_section": ("opis mikroskopowy",), "pathology_conclusion_section": ("rozpoznanie patomorfologiczne",)},
            "albanian": {"specimen_section": ("mostra",), "microscopic_description_section": ("pershkrim mikroskopik",), "pathology_conclusion_section": ("diagnoze patologjike",)},
        },
    ),
    ADMINISTRATIVE_INSURANCE_LABEL: FamilyRule(
        label=ADMINISTRATIVE_INSURANCE_LABEL,
        threshold=2,
        cue_groups={
            "english": {"insurance_terms": ("insurance", "claim", "policy"), "administrative_form": ("authorization", "billing")},
            "russian": {"insurance_terms": ("\u0441\u0442\u0440\u0430\u0445\u043e\u0432\u043a\u0430", "\u043f\u043e\u043b\u0438\u0441"), "administrative_form": ("\u0441\u0447\u0435\u0442", "\u043e\u043f\u043b\u0430\u0442\u0430")},
            "polish": {"insurance_terms": ("ubezpieczenie", "polisa"), "administrative_form": ("rachunek", "platnosc")},
            "albanian": {"insurance_terms": ("sigurim", "police"), "administrative_form": ("fature", "pagese")},
        },
    ),
}


def classify_document_family(text: str | None) -> str:
    diagnostic = document_family_classification_diagnostic(text)
    return str(diagnostic["candidate_family"])


def document_family_classification_diagnostic(text: str | None) -> dict[str, object]:
    normalized = normalize_text(text)
    if not normalized:
        return _unknown_diagnostic("no_text_available", [])

    matches: list[dict[str, object]] = []
    for key, rule in DOCUMENT_FAMILY_REGISTRY.items():
        matched_keys, matched_language_groups = _matched_rule_cues(normalized, rule)
        if not _rule_meets_threshold(rule, matched_keys):
            continue
        matches.append(
            {
                "registry_key": key,
                "label": rule.subtype_label or rule.label,
                "family": rule.label,
                "score": len(matched_keys),
                "matched_family_cue_keys": matched_keys,
                "matched_language_cue_groups": matched_language_groups,
            }
        )

    if not matches:
        return _unknown_diagnostic("too_few_safe_family_cue_keys", _script_labels(normalized))

    matches.sort(key=lambda item: int(item["score"]), reverse=True)
    top_score = int(matches[0]["score"])
    top_matches = [item for item in matches if int(item["score"]) == top_score]
    if len(top_matches) > 1:
        return {
            **_base_diagnostic(normalized),
            "candidate_family": UNKNOWN_DOCUMENT_LABEL,
            "matched_family_cue_keys": [],
            "matched_language_cue_groups": [],
            "classification_block_reason": "ambiguous_family_candidates",
            "ambiguous_candidates": sorted(str(item["label"]) for item in top_matches),
            "review_only": True,
            "auto_accept_allowed": False,
        }

    winner = top_matches[0]
    return {
        **_base_diagnostic(normalized),
        "candidate_family": str(winner["label"]),
        "matched_family_cue_keys": list(winner["matched_family_cue_keys"]),
        "matched_language_cue_groups": list(winner["matched_language_cue_groups"]),
        "classification_block_reason": "classified",
        "ambiguous_candidates": [],
        "review_only": True,
        "auto_accept_allowed": False,
    }


def normalize_text(text: str | None) -> str:
    value = str(text or "").lower().replace("\u0451", "\u0435")
    value = re.sub(r"[\u00a0\t\r\n|;:,./\\()\[\]{}<>_+=-]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _matched_rule_cues(text: str, rule: FamilyRule) -> tuple[list[str], list[str]]:
    cue_keys: set[str] = set()
    language_groups: set[str] = set()
    for language, groups in rule.cue_groups.items():
        for cue_key, terms in groups.items():
            if _any_term_matches(text, terms):
                cue_keys.add(cue_key)
                language_groups.add(f"{language}:{cue_key}")
    return sorted(cue_keys), sorted(language_groups)


def _rule_meets_threshold(rule: FamilyRule, matched_keys: Iterable[str]) -> bool:
    keys = set(matched_keys)
    if len(keys) < rule.threshold:
        return False
    if rule.required_any and not (keys & set(rule.required_any)):
        return False
    return True


def _any_term_matches(text: str, terms: tuple[str, ...]) -> bool:
    return any(_term_matches(text, term) for term in terms)


def _term_matches(text: str, term: str) -> bool:
    normalized_term = normalize_text(term)
    if not normalized_term:
        return False
    if re.fullmatch(r"[a-z0-9]{2,5}", normalized_term):
        return bool(re.search(rf"\b{re.escape(normalized_term)}\b", text, re.I))
    return normalized_term in text


def _base_diagnostic(text: str) -> dict[str, object]:
    scripts = _script_labels(text)
    return {
        "cyrillic_detected": "cyrillic" in scripts,
        "language_script_detected": scripts,
    }


def _unknown_diagnostic(reason: str, scripts: list[str]) -> dict[str, object]:
    return {
        "candidate_family": UNKNOWN_DOCUMENT_LABEL,
        "matched_family_cue_keys": [],
        "matched_language_cue_groups": [],
        "classification_block_reason": reason,
        "ambiguous_candidates": [],
        "cyrillic_detected": "cyrillic" in scripts,
        "language_script_detected": scripts,
        "review_only": True,
        "auto_accept_allowed": False,
    }


def _script_labels(text: str) -> list[str]:
    labels: list[str] = []
    if re.search(r"[\u0400-\u04ff]", text):
        labels.append("cyrillic")
    if re.search(r"[a-z]", text):
        labels.append("latin")
    return labels or ["unknown"]
