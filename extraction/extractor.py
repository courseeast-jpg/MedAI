"""
MedAI v1.1 — Entity Extractor (Track B)
Local Ollama LLM (e.g. llama3.2) invoked via subprocess for schema-constrained
JSON extraction. spaCy rules-based fallback when Ollama is unavailable.
"""
import json
import os
import re
import subprocess
import warnings
from typing import Optional
from loguru import logger

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logger.warning("spaCy not available — rules-based fallback disabled")

from app.config import OLLAMA_MODEL, OLLAMA_TIMEOUT_SEC
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

Output format: Respond with a SINGLE JSON object only. No prose, no markdown fences, no commentary.
The JSON object must have exactly these keys, each holding an array of strings:
{{"diagnoses": [], "medications": [], "test_results": [], "symptoms": [], "notes": [], "recommendations": []}}

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

Output format: Respond with a SINGLE JSON object only. No prose, no markdown fences, no commentary.
Shape:
{{"title": null, "conditions_covered": [], "foods": [{{"food_name": "", "food_name_ru": "", "ibs_score": null, "diverticulitis_score": null, "oxalates_score": null, "crystalluria_score": null, "safety_category": null, "notes": null}}], "general_notes": []}}

Text to extract from:
{text}"""


class Extractor:
    def __init__(self, model: str = OLLAMA_MODEL, timeout_sec: int = OLLAMA_TIMEOUT_SEC):
        self.model = model
        self.timeout_sec = timeout_sec
        self._ollama_available = self._check_ollama()
        self._spacy_nlp = None

        if self._ollama_available:
            logger.info(f"Extractor: Ollama mode (model={self.model})")
        else:
            logger.warning("Extractor: Ollama not available — rules-based only")

    # ── Public API ─────────────────────────────────────────────────────────────

    def extract(self, text: str, specialty: str = "general") -> ExtractionOutput:
        """Extract clinical entities from text."""
        if not text or len(text.strip()) < 20:
            return ExtractionOutput()

        if self._ollama_available:
            try:
                return self._extract_ollama(text, specialty)
            except Exception as e:
                logger.warning(f"Ollama extraction failed: {e}. Falling back to rules.")
                self._ollama_available = False

        return self._extract_rules(text)

    def extract_food_guide(self, text: str, specialty: str = "general") -> FoodGuideOutput:
        """Extract structured food/scoring table data from a food reference guide."""
        if not text or len(text.strip()) < 20:
            return FoodGuideOutput()

        if self._ollama_available:
            try:
                return self._extract_food_guide_ollama(text, specialty)
            except Exception as e:
                logger.warning(f"Ollama food guide extraction failed: {e}")
        return FoodGuideOutput()

    # ── Clinical extractor ─────────────────────────────────────────────────────

    def _extract_ollama(self, text: str, specialty: str) -> ExtractionOutput:
        prompt = EXTRACTION_PROMPT.format(text=text[:8000])
        logger.info(f"[DIAG] Ollama clinical input: {len(text)} chars, truncated to {len(text[:8000])}")
        logger.debug(f"[DIAG] First 500 chars: {text[:500]}")

        raw = self._call_ollama(prompt)
        data = self._parse_json(raw)
        if data is None:
            return ExtractionOutput()

        result = ExtractionOutput()
        for d in data.get('diagnoses', []) or []:
            result.diagnoses.append(ExtractedDiagnosis(name=str(d), date=None))
        for m in data.get('medications', []) or []:
            result.medications.append(ExtractedMedication(name=str(m), dose=None, frequency=None))
        for t in data.get('test_results', []) or []:
            result.test_results.append(ExtractedTestResult(test_name=str(t), value=None, date=None))
        result.symptoms = [str(s) for s in (data.get('symptoms') or [])]
        result.notes = [str(n) for n in (data.get('notes') or [])]
        result.recommendations = [str(r) for r in (data.get('recommendations') or [])]

        logger.info(
            f"[DIAG] Clinical entities — diagnoses:{len(result.diagnoses)}, "
            f"meds:{len(result.medications)}, tests:{len(result.test_results)}, "
            f"symptoms:{len(result.symptoms)}, notes:{len(result.notes)}, "
            f"recs:{len(result.recommendations)}"
        )
        result.extraction_method = "ollama"
        result.confidence = 0.75
        return result

    # ── Food guide extractor ───────────────────────────────────────────────────

    def _extract_food_guide_ollama(self, text: str, specialty: str) -> FoodGuideOutput:
        prompt = FOOD_GUIDE_PROMPT.format(text=text[:12000])
        logger.info(f"[DIAG] Ollama food guide input: {len(text)} chars, truncated to {len(text[:12000])}")

        raw = self._call_ollama(prompt)
        data = self._parse_json(raw)
        if data is None:
            return FoodGuideOutput()

        foods = []
        for f in data.get('foods', []) or []:
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
            conditions_covered=data.get('conditions_covered') or [],
            foods=foods,
            general_notes=data.get('general_notes') or [],
            extraction_method="ollama",
        )
        logger.info(f"[DIAG] Food guide chunk: {len(foods)} foods, {len(result.general_notes)} notes")
        return result

    # ── Ollama subprocess call ─────────────────────────────────────────────────

    def _check_ollama(self) -> bool:
        """Verify that the `ollama` binary exists and the target model is listed."""
        try:
            proc = subprocess.run(
                ["ollama", "list"],
                capture_output=True, text=True, timeout=10,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            logger.warning(f"Ollama check failed: {exc}")
            return False

        if proc.returncode != 0:
            logger.warning(f"`ollama list` returned {proc.returncode}: {proc.stderr[:200]}")
            return False

        model_root = self.model.split(":")[0]
        if model_root not in proc.stdout:
            logger.warning(f"Ollama model '{self.model}' not found in `ollama list`")
            return False
        return True

    def _call_ollama(self, prompt: str) -> str:
        """Call `ollama run <model>` piping the prompt to stdin and returning stdout."""
        try:
            proc = subprocess.run(
                ["ollama", "run", self.model],
                input=prompt,
                capture_output=True,
                text=True,
                timeout=self.timeout_sec,
                encoding="utf-8",
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"Ollama timed out after {self.timeout_sec}s") from exc
        except FileNotFoundError as exc:
            raise RuntimeError("`ollama` binary not found on PATH") from exc

        if proc.returncode != 0:
            raise RuntimeError(
                f"ollama run exited {proc.returncode}: {proc.stderr.strip()[:500]}"
            )

        if proc.stderr and proc.stderr.strip():
            logger.debug(f"[DIAG] Ollama stderr: {proc.stderr.strip()[:300]}")

        raw = (proc.stdout or "").strip()
        logger.info(f"[DIAG] Ollama response: {len(raw)} chars")
        logger.debug(f"[DIAG] Ollama response body: {raw}")
        return raw

    @staticmethod
    def _parse_json(raw: str) -> Optional[dict]:
        """Extract the first JSON object from an LLM response.

        Local models frequently wrap JSON in ```json fences or add preamble.
        Try strict parsing first, then strip fences, then grab the outermost
        {...} block.
        """
        if not raw:
            logger.error("[DIAG] Empty Ollama response")
            return None

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
        if fence:
            try:
                return json.loads(fence.group(1))
            except json.JSONDecodeError:
                pass

        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end > start:
            candidate = raw[start:end + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError as exc:
                logger.error(f"[DIAG] JSON parse failed on extracted block: {exc}")

        logger.error(f"[DIAG] Could not parse JSON from Ollama output ({len(raw)} chars)")
        logger.error(f"[DIAG] Raw: {raw[:1000]}")
        return None

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

        med_pattern = re.findall(r'([A-Za-z]{4,})\s+(\d+(?:\.\d+)?)\s*(mg|mcg|ml|g|units?)', text, re.IGNORECASE)
        for name, dose, unit in med_pattern:
            if not any(m.name.lower() == name.lower() for m in result.medications):
                result.medications.append(ExtractedMedication(name=name, dose=f"{dose}{unit}"))

        return result

    # ── Legacy compatibility ───────────────────────────────────────────────────

    @property
    def claude_available(self) -> bool:
        return self._ollama_available

    def mark_claude_unavailable(self):
        self._ollama_available = False

    def mark_claude_available(self):
        self._ollama_available = True


def _to_float(value) -> Optional[float]:
    """Safely coerce an LLM score field to float."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
