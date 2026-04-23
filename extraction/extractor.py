"""
MedAI v1.1 — Entity Extractor (Track B)
Google Gemini API primary with instructor for schema enforcement.
spaCy rules-based fallback when Gemini unavailable.
"""
import io
import os
import sys
import warnings
from typing import Optional
from loguru import logger

# Suppress PyTorch / HuggingFace stderr noise before any heavy imports
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    logger.warning("google-generativeai not available")

try:
    import instructor
    INSTRUCTOR_AVAILABLE = True
except ImportError:
    INSTRUCTOR_AVAILABLE = False
    logger.warning("instructor not available — using raw Gemini with JSON parsing")

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logger.warning("spaCy not available — rules-based fallback disabled")

from app.config import GEMINI_API_KEY, GEMINI_MODEL, GEMINI_MAX_TOKENS
from app.schemas import (
    ExtractionOutput, ExtractedDiagnosis, ExtractedMedication, ExtractedTestResult,
    FoodEntry, FoodGuideOutput,
)


EXTRACTION_PROMPT = """You are a medical entity extractor. Read the following medical or dietary text and extract all clinically relevant facts.

IMPORTANT: The text may be in Russian or other languages. If the text is not in English:
1. First translate medical/dietary terms to English
2. Then extract entities in English
3. All extracted entity values must be in English

Extract these entity types:
- diagnoses: conditions, diseases, disorders with dates if mentioned
- medications: drug names, doses, frequencies, routes
- test_results: lab values, imaging findings, EEG results with values and dates
- symptoms: patient-reported symptoms with onset and severity
- notes: important clinical observations, specialist recommendations, dietary restrictions, food guidelines
- recommendations: suggested treatments, follow-up actions, referrals, dietary recommendations, food plans

Rules:
- If a field is unknown, use null
- Do not invent information not in the text
- Extract only what is explicitly stated
- For medications, include dose and frequency if mentioned
- ALL extracted values must be in ENGLISH (translate from source language if needed)

Text to extract from:
{text}"""


FOOD_GUIDE_PROMPT = """You are extracting structured data from a Russian food reference guide or dietary scoring table.

The document lists foods with numeric scores and safety ratings for medical conditions.
Common conditions in these guides: IBS (irritable bowel syndrome), Diverticulitis, Oxalates/kidney stones, Crystalluria.

For each food item extract:
- food_name: English name of the food (translate from Russian)
- food_name_ru: original Russian food name (keep as-is)
- ibs_score: numeric score for IBS (0-10, higher = more restricted; null if absent)
- diverticulitis_score: numeric score for Diverticulitis (0-10; null if absent)
- oxalates_score: numeric score for Oxalates/kidney stones (0-10; null if absent)
- crystalluria_score: numeric score for Crystalluria (0-10; null if absent)
- safety_category: one of exactly: Safe, Suitable, Caution, Undesirable, Exclude
  Russian translations — Безопасно/Разрешено=Safe, Подходит/Можно=Suitable,
  Осторожно/С осторожностью=Caution, Нежелательно=Undesirable, Исключить/Запрещено=Exclude
- notes: any food-specific notes (null if none)

Also extract:
- title: document title if present (null otherwise)
- conditions_covered: list of medical conditions this page/section addresses
- general_notes: any guidelines not tied to a specific food

Rules:
- Translate ALL Russian food names to English
- Keep original Russian name in food_name_ru
- Extract every food row you can see — do not skip rows
- Do not invent scores not present in the text

Text to extract from:
{text}"""


_FOOD_GUIDE_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "conditions_covered": {"type": "array", "items": {"type": "string"}},
        "foods": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "food_name":            {"type": "string"},
                    "food_name_ru":         {"type": "string"},
                    "ibs_score":            {"type": "number"},
                    "diverticulitis_score": {"type": "number"},
                    "oxalates_score":       {"type": "number"},
                    "crystalluria_score":   {"type": "number"},
                    "safety_category":      {"type": "string"},
                    "notes":                {"type": "string"},
                },
            },
        },
        "general_notes": {"type": "array", "items": {"type": "string"}},
    },
}

_CLINICAL_SCHEMA = {
    "type": "object",
    "properties": {
        "diagnoses":     {"type": "array", "items": {"type": "string"}},
        "medications":   {"type": "array", "items": {"type": "string"}},
        "test_results":  {"type": "array", "items": {"type": "string"}},
        "symptoms":      {"type": "array", "items": {"type": "string"}},
        "notes":         {"type": "array", "items": {"type": "string"}},
        "recommendations": {"type": "array", "items": {"type": "string"}},
    },
}


class Extractor:
    def __init__(self):
        self._gemini_available = bool(GEMINI_API_KEY and GENAI_AVAILABLE)
        self._spacy_nlp = None

        if self._gemini_available:
            genai.configure(api_key=GEMINI_API_KEY)
            self.client = genai.GenerativeModel(GEMINI_MODEL)
            logger.info("Extractor: Gemini mode")
        else:
            self.client = None
            logger.warning("Extractor: no API key — rules-based only")

    # ── Public API ─────────────────────────────────────────────────────────────

    def extract(self, text: str, specialty: str = "general") -> ExtractionOutput:
        """Extract clinical entities from text."""
        if not text or len(text.strip()) < 20:
            return ExtractionOutput()

        if self.client and self._gemini_available:
            try:
                return self._extract_gemini(text, specialty)
            except Exception as e:
                logger.warning(f"Gemini extraction failed (API error): {e}. Falling back to rules.")
                self._gemini_available = False

        return self._extract_rules(text)

    def extract_food_guide(self, text: str, specialty: str = "general") -> FoodGuideOutput:
        """Extract structured food/scoring table data from a food reference guide."""
        if not text or len(text.strip()) < 20:
            return FoodGuideOutput()

        if self.client and self._gemini_available:
            try:
                return self._extract_food_guide_gemini(text, specialty)
            except Exception as e:
                logger.warning(f"Gemini food guide extraction failed (API error): {e}")
        return FoodGuideOutput()

    # ── Clinical extractor ─────────────────────────────────────────────────────

    def _extract_gemini(self, text: str, specialty: str) -> ExtractionOutput:
        import json

        prompt = EXTRACTION_PROMPT.format(text=text[:8000])

        logger.info(f"[DIAG] Gemini clinical input: {len(text)} chars, truncated to {len(text[:8000])}")
        logger.debug(f"[DIAG] First 500 chars: {text[:500]}")

        raw = self._call_gemini(prompt, _CLINICAL_SCHEMA)

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.error(f"[DIAG] JSON parse failed: {exc}")
            logger.error(f"[DIAG] Full raw response ({len(raw)} chars): {raw}")
            if raw and not raw.rstrip().endswith('}'):
                logger.error(
                    f"[DIAG] Response appears truncated — GEMINI_MAX_TOKENS={GEMINI_MAX_TOKENS}"
                )
            else:
                logger.error("[DIAG] Response not truncated — possible stderr pollution or schema mismatch")
            return ExtractionOutput()

        result = ExtractionOutput()
        for d in data.get('diagnoses', []):
            result.diagnoses.append(ExtractedDiagnosis(name=str(d), date=None))
        for m in data.get('medications', []):
            result.medications.append(ExtractedMedication(name=str(m), dose=None, frequency=None))
        for t in data.get('test_results', []):
            result.test_results.append(ExtractedTestResult(test_name=str(t), value=None, date=None))
        result.symptoms = [str(s) for s in data.get('symptoms', [])]
        result.notes = [str(n) for n in data.get('notes', [])]
        result.recommendations = [str(r) for r in data.get('recommendations', [])]

        logger.info(
            f"[DIAG] Clinical entities — diagnoses:{len(result.diagnoses)}, "
            f"meds:{len(result.medications)}, tests:{len(result.test_results)}, "
            f"symptoms:{len(result.symptoms)}, notes:{len(result.notes)}, "
            f"recs:{len(result.recommendations)}"
        )
        result.extraction_method = "gemini"
        result.confidence = 0.80
        return result

    # ── Food guide extractor ───────────────────────────────────────────────────

    def _extract_food_guide_gemini(self, text: str, specialty: str) -> FoodGuideOutput:
        import json

        # Food tables are dense; allow more input than clinical text
        prompt = FOOD_GUIDE_PROMPT.format(text=text[:12000])

        logger.info(f"[DIAG] Gemini food guide input: {len(text)} chars, truncated to {len(text[:12000])}")

        raw = self._call_gemini(prompt, _FOOD_GUIDE_SCHEMA)

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.error(f"[DIAG] Food guide JSON parse failed: {exc}")
            logger.error(f"[DIAG] Full raw response ({len(raw)} chars): {raw}")
            return FoodGuideOutput()

        foods = []
        for f in data.get('foods', []):
            try:
                foods.append(FoodEntry(
                    food_name=str(f.get('food_name', '')),
                    food_name_ru=f.get('food_name_ru') or None,
                    ibs_score=_to_float(f.get('ibs_score')),
                    diverticulitis_score=_to_float(f.get('diverticulitis_score')),
                    oxalates_score=_to_float(f.get('oxalates_score')),
                    crystalluria_score=_to_float(f.get('crystalluria_score')),
                    safety_category=f.get('safety_category') or None,
                    notes=f.get('notes') or None,
                ))
            except Exception as exc:
                logger.warning(f"[DIAG] Skipping malformed food entry {f}: {exc}")

        result = FoodGuideOutput(
            title=data.get('title') or None,
            conditions_covered=data.get('conditions_covered', []),
            foods=foods,
            general_notes=data.get('general_notes', []),
        )
        logger.info(f"[DIAG] Food guide chunk: {len(foods)} foods, {len(result.general_notes)} notes")
        return result

    # ── Shared Gemini call with stderr capture ─────────────────────────────────

    def _call_gemini(self, prompt: str, schema: dict) -> str:
        """Call Gemini with JSON mode and capture stderr for pollution diagnostics."""
        _stderr_buf = io.StringIO()
        _orig_stderr = sys.stderr
        sys.stderr = _stderr_buf
        try:
            response = self.client.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    max_output_tokens=GEMINI_MAX_TOKENS,
                    temperature=0.1,
                    response_mime_type="application/json",
                    response_schema=schema,
                ),
            )
        finally:
            sys.stderr = _orig_stderr
            captured = _stderr_buf.getvalue()
            if captured.strip():
                logger.warning(f"[DIAG] Stderr during Gemini call ({len(captured)} chars): {captured[:500]}")

        raw = response.text.strip()
        logger.info(f"[DIAG] Gemini response: {len(raw)} chars")
        logger.debug(f"[DIAG] Gemini response body: {raw}")
        return raw

    # ── Rules-based fallback ───────────────────────────────────────────────────

    def _extract_rules(self, text: str) -> ExtractionOutput:
        result = ExtractionOutput(extraction_method="rules_based", confidence=0.45)

        if SPACY_AVAILABLE:
            if self._spacy_nlp is None:
                try:
                    self._spacy_nlp = spacy.load("en_core_web_trf")
                except OSError:
                    try:
                        self._spacy_nlp = spacy.load("en_core_web_sm")
                    except OSError:
                        logger.error("No spaCy model available")
                        return result

            doc = self._spacy_nlp(text[:5000])
            for ent in doc.ents:
                if ent.label_ in ("DISEASE", "CONDITION"):
                    result.diagnoses.append(ExtractedDiagnosis(name=ent.text))
                elif ent.label_ in ("DRUG", "CHEMICAL"):
                    result.medications.append(ExtractedMedication(name=ent.text))

        import re
        med_pattern = re.findall(r'([A-Za-z]{4,})\s+(\d+(?:\.\d+)?)\s*(mg|mcg|ml|g|units?)', text, re.IGNORECASE)
        for name, dose, unit in med_pattern:
            if not any(m.name.lower() == name.lower() for m in result.medications):
                result.medications.append(ExtractedMedication(name=name, dose=f"{dose}{unit}"))

        return result

    # ── Legacy compatibility ───────────────────────────────────────────────────

    @property
    def claude_available(self) -> bool:
        return self._gemini_available

    def mark_claude_unavailable(self):
        self._gemini_available = False

    def mark_claude_available(self):
        self._gemini_available = True


MedicalExtractor = Extractor


def create_extractor() -> Extractor:
    """Legacy factory retained for golden-test and older caller compatibility."""
    return Extractor()


def _to_float(value) -> Optional[float]:
    """Safely coerce a Gemini score field to float."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
