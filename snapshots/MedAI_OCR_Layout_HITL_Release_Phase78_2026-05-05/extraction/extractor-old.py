"""
MedAI v1.1 — Entity Extractor (Track B)
Google Gemini API primary with instructor for schema enforcement.
spaCy rules-based fallback when Gemini unavailable.
"""
from typing import Optional
from loguru import logger

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
from app.schemas import ExtractionOutput, ExtractedDiagnosis, ExtractedMedication, ExtractedTestResult


EXTRACTION_PROMPT = """You are a medical entity extractor. Read the following medical or dietary text and extract all clinically relevant facts.

Extract these entity types:
- diagnoses: conditions, diseases, disorders with dates if mentioned
- medications: drug names, doses, frequencies, routes
- test_results: lab values, imaging findings, EEG results with values and dates
- symptoms: patient-reported symptoms with onset and severity
- notes: important clinical observations, specialist recommendations, dietary restrictions, food guidelines
- recommendations: suggested treatments, follow-up actions, referrals, dietary recommendations, food plans

For dietary/nutrition content, extract:
- Food items and their properties (high/low in specific nutrients)
- Dietary restrictions or conditions (IBS, diverticulitis, kidney stones, etc.)
- Food categories and classifications
- Nutritional guidelines and recommendations

Rules:
- If a field is unknown, use null
- Do not invent information not in the text
- Extract only what is explicitly stated
- For medications, include dose and frequency if mentioned
- For foods, include relevant properties and conditions they address

Text to extract from:
{text}"""


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

    def extract(self, text: str, specialty: str = "general") -> ExtractionOutput:
        """Extract medical entities from text. Returns ExtractionOutput."""
        if not text or len(text.strip()) < 20:
            return ExtractionOutput()

        if self.client and self._gemini_available:
            try:
                return self._extract_gemini(text, specialty)
            except Exception as e:
                logger.warning(f"Gemini extraction failed: {e}. Falling back to rules.")
                self._gemini_available = False

        return self._extract_rules(text)

    def _extract_gemini(self, text: str, specialty: str) -> ExtractionOutput:
        prompt = EXTRACTION_PROMPT.format(text=text[:8000])  # context limit guard
        
        # Gemini with JSON output
        import json
        prompt_with_json = prompt + "\n\nRespond ONLY with valid JSON matching this schema:\n"
        prompt_with_json += """{
  "diagnoses": [{"name": "string", "date": "string or null"}],
  "medications": [{"name": "string", "dose": "string or null", "frequency": "string or null"}],
  "test_results": [{"test_name": "string", "value": "string or null", "date": "string or null"}],
  "symptoms": ["string"],
  "notes": ["string"],
  "recommendations": ["string"]
}"""
        
        response = self.client.generate_content(
            prompt_with_json,
            generation_config=genai.GenerationConfig(
                max_output_tokens=GEMINI_MAX_TOKENS,
                temperature=0.1,
            )
        )
        
        raw = response.text.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        data = json.loads(raw)
        result = ExtractionOutput(**data)
        result.extraction_method = "gemini"
        result.confidence = 0.80
        return result

    def _extract_rules(self, text: str) -> ExtractionOutput:
        """spaCy + regex rules-based extraction fallback."""
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

        # Simple regex for common patterns
        import re
        # Medication doses: drug name + number + unit
        med_pattern = re.findall(r'([A-Za-z]{4,})\s+(\d+(?:\.\d+)?)\s*(mg|mcg|ml|g|units?)', text, re.IGNORECASE)
        for name, dose, unit in med_pattern:
            if not any(m.name.lower() == name.lower() for m in result.medications):
                result.medications.append(ExtractedMedication(name=name, dose=f"{dose}{unit}"))

        return result

    @property
    def claude_available(self) -> bool:
        """Legacy property name for compatibility"""
        return self._gemini_available

    def mark_claude_unavailable(self):
        """Legacy method name for compatibility"""
        self._gemini_available = False

    def mark_claude_available(self):
        """Legacy method name for compatibility"""
        self._gemini_available = True
