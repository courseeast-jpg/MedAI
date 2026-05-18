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
        required_any=(
            "lab_report_heading",
            "specimen_or_biomaterial",
            "result_or_report",
            "table_header",
            "lab_table_column_structure",
            "analyte_value_unit_pattern",
            "reference_range_column_pattern",
            "specimen_result_report_structure",
            "laboratory_panel_abbreviation_latin",
            "biomaterial_result_table_structure",
        ),
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
                    "\u043c\u0440 \u0442\u043e\u043c\u043e\u0433\u0440\u0430\u043c\u043c",
                    "\u043c\u0440 \u0442\u043e\u043c\u043e\u0433\u0440\u0430\u0444",
                    "\u0442\u043e\u043c\u043e\u0433\u0440\u0430\u043c\u043c",
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
                    "t1",
                    "t2",
                    "flair",
                    "dwi",
                ),
                "brain_mri_context": (
                    "\u0433\u043e\u043b\u043e\u0432\u043d\u043e\u0433\u043e \u043c\u043e\u0437\u0433\u0430",
                    "\u0441\u0430\u0433\u0438\u0442\u0442\u0430\u043b\u044c",
                    "\u0430\u043a\u0441\u0438\u0430\u043b\u044c",
                    "\u043a\u043e\u0440\u043e\u043d\u0430\u0440",
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
        threshold=3,
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
    raw_text = str(text or "")
    normalized = normalize_text(text)
    if not normalized:
        return _unknown_diagnostic("no_text_available", [])

    matches: list[dict[str, object]] = []
    for key, rule in DOCUMENT_FAMILY_REGISTRY.items():
        matched_keys, matched_language_groups = _matched_rule_cues(normalized, rule)
        if key == LAB_RESULT_LABEL:
            structure_keys, structure_groups = _matched_latin_lab_structure_cues(raw_text, normalized)
            matched_keys = sorted(set(matched_keys) | set(structure_keys))
            matched_language_groups = sorted(set(matched_language_groups) | set(structure_groups))
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

    conflict = _resolve_family_conflict(normalized, matches)
    if conflict is not None:
        return conflict

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
            "conflict_resolution_reason": "tie_without_safe_precedence",
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
        "conflict_resolution_reason": "none",
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


def _matched_latin_lab_structure_cues(raw_text: str, normalized: str) -> tuple[list[str], list[str]]:
    if not re.search(r"[a-z]", normalized):
        return [], []

    cue_keys: set[str] = set()
    language_groups: set[str] = set()
    raw_lower = str(raw_text or "").lower()
    lines = [line.strip().lower() for line in str(raw_text or "").splitlines() if line.strip()]

    if _latin_lab_table_column_structure(raw_lower, normalized):
        cue_keys.add("lab_table_column_structure")
    if _latin_analyte_value_unit_pattern(raw_lower, normalized, lines):
        cue_keys.add("analyte_value_unit_pattern")
    if _latin_reference_range_column_pattern(raw_lower, normalized):
        cue_keys.add("reference_range_column_pattern")
    if _latin_flag_or_status_column_pattern(raw_lower, normalized):
        cue_keys.add("flag_or_status_column_pattern")
    if _latin_specimen_result_report_structure(normalized):
        cue_keys.add("specimen_result_report_structure")
    if _latin_laboratory_panel_abbreviation(normalized):
        cue_keys.add("laboratory_panel_abbreviation_latin")
    if _latin_biomaterial_result_table_structure(raw_lower, normalized):
        cue_keys.add("biomaterial_result_table_structure")

    strong_structure = {
        "lab_table_column_structure",
        "analyte_value_unit_pattern",
        "reference_range_column_pattern",
        "specimen_result_report_structure",
        "laboratory_panel_abbreviation_latin",
        "biomaterial_result_table_structure",
    }
    if len(cue_keys & strong_structure) < 2:
        return [], []
    if _latin_treatment_schedule_without_lab_structure(normalized, cue_keys):
        return [], []
    language_groups = {f"latin_structure:{key}" for key in cue_keys}
    return sorted(cue_keys), sorted(language_groups)


def _latin_lab_table_column_structure(raw_lower: str, normalized: str) -> bool:
    header_terms = {
        "component",
        "analyte",
        "test",
        "test name",
        "result",
        "value",
        "unit",
        "units",
        "reference",
        "reference range",
        "reference interval",
        "flag",
        "status",
    }
    header_hits = sum(1 for term in header_terms if term in raw_lower or term in normalized)
    return header_hits >= 3 and bool(re.search(r"\d", raw_lower))


def _latin_analyte_value_unit_pattern(raw_lower: str, normalized: str, lines: list[str]) -> bool:
    units = r"(mg/dl|g/dl|mmol/l|umol/l|µmol/l|u/l|iu/l|miu/l|ng/ml|pg/ml|10\^?3/ul|10\^?6/ul|%)"
    line_matches = sum(1 for line in lines if re.search(rf"\b[a-z][a-z0-9 ()/-]{{2,40}}\s+\d+(?:[.,]\d+)?\s*{units}\b", line))
    if line_matches >= 2:
        return True
    normalized_units = {
        "mg dl",
        "g dl",
        "mmol l",
        "umol l",
        "u l",
        "iu l",
        "ng ml",
        "pg ml",
    }
    return bool(re.search(r"\b[a-z][a-z0-9 ]{2,40}\s+\d+(?: \d+)?\b", normalized)) and sum(
        1 for unit in normalized_units if unit in normalized
    ) >= 1


def _latin_reference_range_column_pattern(raw_lower: str, normalized: str) -> bool:
    if "reference range" in raw_lower or "reference interval" in raw_lower:
        return True
    if "ref range" in raw_lower or "normal range" in raw_lower:
        return True
    has_range = bool(re.search(r"\b\d+(?:[.,]\d+)?\s*[-–]\s*\d+(?:[.,]\d+)?\b", raw_lower))
    return has_range and any(term in normalized for term in ("result", "value", "unit", "units", "component", "analyte"))


def _latin_flag_or_status_column_pattern(raw_lower: str, normalized: str) -> bool:
    if "flag" in normalized or "abnormal" in normalized:
        return True
    return bool(re.search(r"\b(high|low|normal|positive|negative|detected|not detected)\b", raw_lower)) and any(
        term in normalized for term in ("result", "value", "reference", "unit", "units")
    )


def _latin_specimen_result_report_structure(normalized: str) -> bool:
    specimen = any(term in normalized for term in ("specimen", "sample", "biomaterial", "serum", "plasma", "blood", "urine"))
    result = any(term in normalized for term in ("result", "reported", "report"))
    return specimen and result


def _latin_laboratory_panel_abbreviation(normalized: str) -> bool:
    panel_terms = {
        "cbc",
        "cmp",
        "bmp",
        "lipid panel",
        "comprehensive metabolic panel",
        "basic metabolic panel",
        "hemoglobin",
        "hematocrit",
        "platelet",
        "glucose",
        "creatinine",
        "cholesterol",
        "triglycerides",
        "tsh",
        "hdl",
        "ldl",
        "wbc",
        "rbc",
    }
    hits = sum(1 for term in panel_terms if re.search(rf"\b{re.escape(term)}\b", normalized))
    return hits >= 2


def _latin_biomaterial_result_table_structure(raw_lower: str, normalized: str) -> bool:
    return _latin_specimen_result_report_structure(normalized) and _latin_lab_table_column_structure(raw_lower, normalized)


def _latin_treatment_schedule_without_lab_structure(normalized: str, cue_keys: set[str]) -> bool:
    treatment_terms = {"schedule", "dose", "dosage", "take", "daily", "morning", "evening", "therapy", "treatment"}
    treatment_hits = sum(1 for term in treatment_terms if re.search(rf"\b{re.escape(term)}\b", normalized))
    lab_specific = cue_keys & {
        "analyte_value_unit_pattern",
        "reference_range_column_pattern",
        "specimen_result_report_structure",
        "laboratory_panel_abbreviation_latin",
        "biomaterial_result_table_structure",
    }
    return treatment_hits >= 2 and len(lab_specific) < 2


def _rule_meets_threshold(rule: FamilyRule, matched_keys: Iterable[str]) -> bool:
    keys = set(matched_keys)
    if len(keys) < rule.threshold:
        return False
    if rule.required_any and not (keys & set(rule.required_any)):
        return False
    return True


def _resolve_family_conflict(text: str, matches: list[dict[str, object]]) -> dict[str, object] | None:
    imaging = next((item for item in matches if item["label"] == IMAGING_REPORT_LABEL), None)
    treatment_like = [
        item
        for item in matches
        if item["label"] in {TREATMENT_PLAN_LABEL, MEDICATION_PLAN_LABEL}
    ]
    if imaging is None:
        return None

    imaging_keys = set(imaging["matched_family_cue_keys"])
    strong_imaging = _has_strong_imaging_evidence(imaging_keys)
    if not treatment_like:
        weak_treatment_keys = _weak_treatment_like_keys(text)
        if strong_imaging and len(weak_treatment_keys) >= 2:
            return {
                **_base_diagnostic(text),
                "candidate_family": IMAGING_REPORT_LABEL,
                "matched_family_cue_keys": list(imaging["matched_family_cue_keys"]),
                "matched_language_cue_groups": list(imaging["matched_language_cue_groups"]),
                "classification_block_reason": "classified",
                "ambiguous_candidates": [IMAGING_REPORT_LABEL, TREATMENT_PLAN_LABEL],
                "conflict_resolution_reason": "imaging_modality_and_report_structure_overrode_generic_treatment_cues",
                "review_only": True,
                "auto_accept_allowed": False,
            }
        return None

    strong_treatment = any(_has_strong_treatment_evidence(set(item["matched_family_cue_keys"])) for item in treatment_like)
    ambiguous = sorted(str(item["label"]) for item in [imaging, *treatment_like])

    if strong_imaging and not strong_treatment:
        return {
            **_base_diagnostic(text),
            "candidate_family": IMAGING_REPORT_LABEL,
            "matched_family_cue_keys": list(imaging["matched_family_cue_keys"]),
            "matched_language_cue_groups": list(imaging["matched_language_cue_groups"]),
            "classification_block_reason": "classified",
            "ambiguous_candidates": ambiguous,
            "conflict_resolution_reason": "imaging_modality_and_report_structure_overrode_generic_treatment_cues",
            "review_only": True,
            "auto_accept_allowed": False,
        }
    if strong_imaging and strong_treatment:
        return {
            **_base_diagnostic(text),
            "candidate_family": UNKNOWN_DOCUMENT_LABEL,
            "matched_family_cue_keys": [],
            "matched_language_cue_groups": [],
            "classification_block_reason": "ambiguous_family_candidates",
            "ambiguous_candidates": ambiguous,
            "conflict_resolution_reason": "strong_imaging_and_treatment_cues_require_review",
            "review_only": True,
            "auto_accept_allowed": False,
        }
    return {
        **_base_diagnostic(text),
        "candidate_family": UNKNOWN_DOCUMENT_LABEL,
        "matched_family_cue_keys": [],
        "matched_language_cue_groups": [],
        "classification_block_reason": "ambiguous_family_candidates",
        "ambiguous_candidates": ambiguous,
        "conflict_resolution_reason": "weak_mixed_family_cues_not_forced",
        "review_only": True,
        "auto_accept_allowed": False,
    }


def _has_strong_imaging_evidence(keys: set[str]) -> bool:
    has_modality = "imaging_modality" in keys
    has_structure = bool(
        keys
        & {
            "imaging_device_header",
            "imaging_description_section",
            "imaging_conclusion_section",
            "radiology_series_wording",
            "brain_mri_context",
        }
    )
    return has_modality and has_structure


def _has_strong_treatment_evidence(keys: set[str]) -> bool:
    if "medication_schedule_header" in keys and (
        "administration_schedule_pattern" in keys or "date_grid" in keys
    ):
        return True
    return "treatment_recommendation_section" in keys and bool(
        keys & {"administration_schedule_pattern", "physiotherapy_section"}
    )


def _weak_treatment_like_keys(text: str) -> set[str]:
    treatment_rule = DOCUMENT_FAMILY_REGISTRY[TREATMENT_PLAN_LABEL]
    treatment_keys, _ = _matched_rule_cues(text, treatment_rule)
    medication_rule = DOCUMENT_FAMILY_REGISTRY[MEDICATION_PLAN_LABEL]
    medication_keys, _ = _matched_rule_cues(text, medication_rule)
    return set(treatment_keys) | set(medication_keys)


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
        "conflict_resolution_reason": "none",
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
