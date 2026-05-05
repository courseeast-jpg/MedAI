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

For dietary/nutrition content, extract:
- Food items and their properties (high/low in specific nutrients)
- Dietary restrictions or conditions (IBS, diverticulitis, kidney stones, etc.)
- Food categories and classifications
- Nutritional guidelines and recommendations
- Medical conditions related to diet (translate Russian medical terms to English equivalents)

Rules:
- If a field is unknown, use null
- Do not invent information not in the text
- Extract only what is explicitly stated
- For medications, include dose and frequency if mentioned
- For foods, include relevant properties and conditions they address
- ALL extracted values must be in ENGLISH (translate from source language if needed)

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
        
        # DEBUG: Log input text length
        logger.info(f"[DEBUG] Gemini input: {len(text)} chars, truncated to {len(text[:8000])}")
        logger.debug(f"[DEBUG] First 500 chars: {text[:500]}")
        
        # Use Gemini's native JSON mode - guarantees valid JSON output
        import json
        
        response = self.client.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                max_output_tokens=GEMINI_MAX_TOKENS,
                temperature=0.1,
                response_mime_type="application/json",
                response_schema={
                    "type": "object",
                    "properties": {
                        "diagnoses": {"type": "array", "items": {"type": "string"}},
                        "medications": {"type": "array", "items": {"type": "string"}},
                        "test_results": {"type": "array", "items": {"type": "string"}},
                        "symptoms": {"type": "array", "items": {"type": "string"}},
                        "notes": {"type": "array", "items": {"type": "string"}},
                        "recommendations": {"type": "array", "items": {"type": "string"}}
                    }
                }
            )
        )
        
        raw = response.text.strip()
        
        # DEBUG: Log Gemini response
        logger.info(f"[DEBUG] Gemini raw response length: {len(raw)} chars")
        logger.debug(f"[DEBUG] Gemini response: {raw}")
        
        # Native JSON mode guarantees valid JSON - no cleanup needed
        data = json.loads(raw)
        
        # Convert simple arrays to structured format
        result = ExtractionOutput()
        
        # Convert diagnosis strings to structured objects
        for d in data.get('diagnoses', []):
            result.diagnoses.append(ExtractedDiagnosis(name=str(d), date=None))
        
        # Convert medication strings to structured objects  
        for m in data.get('medications', []):
            result.medications.append(ExtractedMedication(name=str(m), dose=None, frequency=None))
        
        # Convert test result strings to structured objects
        for t in data.get('test_results', []):
            result.test_results.append(ExtractedTestResult(test_name=str(t), value=None, date=None))
        
        # Direct copy for string arrays
        result.symptoms = [str(s) for s in data.get('symptoms', [])]
        result.notes = [str(n) for n in data.get('notes', [])]
        result.recommendations = [str(r) for r in data.get('recommendations', [])]
        
        # DEBUG: Log parsed entity counts
        logger.info(f"[DEBUG] Parsed entities - diagnoses:{len(result.diagnoses)}, meds:{len(result.medications)}, tests:{len(result.test_results)}, symptoms:{len(result.symptoms)}, notes:{len(result.notes)}, recs:{len(result.recommendations)}")
        
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
