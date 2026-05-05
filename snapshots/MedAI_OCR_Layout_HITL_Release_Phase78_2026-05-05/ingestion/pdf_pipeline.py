"""
MedAI v1.1 - PDF ingestion pipeline.

Uses docling for unified digital + scanned PDF processing when available.
Falls back to PyMuPDF + Tesseract CLI if docling is unavailable or the
native extracted text is not readable enough for downstream extraction.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from loguru import logger

try:
    from docling.document_converter import DocumentConverter

    DOCLING_AVAILABLE = True
    logger.info("docling available for PDF processing")
except ImportError:
    DOCLING_AVAILABLE = False
    logger.warning("docling not available - falling back to PyMuPDF + Tesseract")

from app.config import PDF_STORAGE_PATH, TIER_ACTIVE, TRUST_CLINICAL
from app.schemas import ExtractionOutput, FoodGuideOutput, MKBRecord
from extraction.extractor import Extractor
from extraction.pii_stripper import PIIStripper

_FOOD_RU_KEYWORDS = [
    "РїСЂРѕРґСѓРєС‚",
    "РїРёС‚Р°РЅРёРµ",
    "РїРёС‰Р°",
    "Р±Р»СЋРґРѕ",
    "СЂР°С†РёРѕРЅ",
    "РїСЂРѕРґСѓРєС‚С‹",
    "РµРґР°",
    "РєР°Р»РѕСЂРёР№",
    "Р±РµР»РєРё",
    "Р¶РёСЂС‹",
    "СѓРіР»РµРІРѕРґС‹",
]
_FOOD_CONDITION_KEYWORDS = ["ibs", "diverticulitis", "oxalate", "crystalluria", "fodmap"]
_TEXT_QUALITY_MEDICAL_TOKENS = {
    "abnormal",
    "blood",
    "calcium",
    "crystals",
    "hpf",
    "lab",
    "negative",
    "normal",
    "oxalate",
    "positive",
    "rbc",
    "result",
    "trace",
    "ua",
    "urinalysis",
    "urine",
    "value",
}
_TEXT_QUALITY_READABLE_TOKEN_RE = re.compile(r"[A-Za-z]{2,}(?:[/-][A-Za-z0-9]{1,8})?")
_TEXT_QUALITY_ALLOWED_CONTROL = {"\n", "\r", "\t"}


class PDFPipeline:
    def __init__(self, extractor: Extractor, pii_stripper: PIIStripper):
        self.extractor = extractor
        self.pii_stripper = pii_stripper
        self.last_text_audit = self._build_document_text_audit(
            text_quality_status="not_evaluated",
            text_quality_score=0.0,
            ocr_fallback_used=False,
            ocr_engine=None,
            ocr_text_length=0,
            page_audits=[],
        )
        PDF_STORAGE_PATH.mkdir(parents=True, exist_ok=True)
        self.converter = DocumentConverter() if DOCLING_AVAILABLE else None

    def process(
        self,
        pdf_path: Path,
        specialty: str = "general",
        session_id: str = "",
    ) -> list[MKBRecord]:
        """
        Full pipeline: PDF -> text -> document type detection -> PII-stripped ->
        extracted -> MKB records.
        """
        session_id = session_id or str(uuid4())
        source_name = pdf_path.name

        stored_path = PDF_STORAGE_PATH / source_name
        if not stored_path.exists():
            shutil.copy2(pdf_path, stored_path)

        raw_text = self._extract_text(pdf_path)
        if not raw_text or len(raw_text.strip()) < 50:
            logger.warning(f"PDF {source_name}: insufficient text extracted")
            return []

        logger.info(f"PDF {source_name}: {len(raw_text)} chars extracted")

        doc_type = self._detect_document_type(raw_text)
        logger.info(f"PDF {source_name}: document type detected as '{doc_type}'")

        if doc_type == "food_guide":
            return self._process_food_guide(raw_text, source_name, specialty, session_id)

        if len(raw_text) > 10000:
            logger.info(f"PDF {source_name}: large clinical document, using chunked extraction")
            return self._process_chunked_clinical(raw_text, source_name, specialty, session_id)

        stripped_text, _pii_method = self.pii_stripper.strip(raw_text)
        is_clean, findings = self.pii_stripper.verify_clean(stripped_text)
        if not is_clean:
            logger.warning(f"PII audit: possible remnants in {source_name}: {findings}")

        extraction = self.extractor.extract(stripped_text, specialty)
        records = self._to_records(extraction, source_name, specialty, session_id)
        logger.info(f"PDF {source_name}: {len(records)} records extracted")
        return records

    def _detect_document_type(self, text: str) -> str:
        text_lower = text.lower()
        food_ru_hits = sum(1 for kw in _FOOD_RU_KEYWORDS if kw in text_lower)
        food_cond_hits = sum(1 for kw in _FOOD_CONDITION_KEYWORDS if kw in text_lower)
        score_matches = len(re.findall(r"\b[0-9]\b", text))

        if food_ru_hits >= 2 or food_cond_hits >= 2 or (food_ru_hits >= 1 and score_matches >= 20):
            return "food_guide"
        return "clinical_report"

    def _split_into_chunks(self, text: str, chunk_size: int = 6000) -> list[str]:
        if "[Page " in text:
            raw = text.split("[Page ")
            return ["[Page " + chunk if index > 0 else chunk for index, chunk in enumerate(raw) if chunk.strip()]

        lines = text.split("\n")
        chunks: list[str] = []
        current_lines: list[str] = []
        current_size = 0

        for line in lines:
            line_size = len(line) + 1
            if current_size + line_size > chunk_size and current_lines:
                chunks.append("\n".join(current_lines))
                current_lines = [line]
                current_size = line_size
            else:
                current_lines.append(line)
                current_size += line_size

        if current_lines:
            chunks.append("\n".join(current_lines))

        return [chunk for chunk in chunks if chunk.strip()]

    def _process_food_guide(
        self,
        raw_text: str,
        source_name: str,
        specialty: str,
        session_id: str,
    ) -> list[MKBRecord]:
        chunks = self._split_into_chunks(raw_text, chunk_size=6000)
        logger.info(f"Food guide '{source_name}': {len(chunks)} chunks to process")

        all_records: list[MKBRecord] = []
        seen_foods: set[str] = set()

        for index, chunk in enumerate(chunks):
            stripped_text, _ = self.pii_stripper.strip(chunk, doc_type="food_guide")
            food_output = self.extractor.extract_food_guide(stripped_text, specialty)
            records = self._food_guide_to_records(
                food_output,
                source_name,
                specialty,
                session_id,
                seen_foods,
            )
            all_records.extend(records)
            logger.info(
                f"Food guide chunk {index + 1}/{len(chunks)}: "
                f"{len(food_output.foods)} foods, {len(records)} records"
            )

        logger.info(f"Food guide '{source_name}': {len(all_records)} total records")
        return all_records

    def _food_guide_to_records(
        self,
        food_output: FoodGuideOutput,
        source_name: str,
        specialty: str,
        session_id: str,
        seen_foods: set[str],
    ) -> list[MKBRecord]:
        records: list[MKBRecord] = []
        confidence = food_output.confidence
        method = food_output.extraction_method

        for food in food_output.foods:
            name_key = food.food_name.lower().strip()
            if not name_key or name_key in seen_foods:
                continue
            seen_foods.add(name_key)

            parts = [f"Food: {food.food_name}"]
            if food.food_name_ru:
                parts.append(f"(ru: {food.food_name_ru})")

            scores = []
            if food.ibs_score is not None:
                scores.append(f"IBS={food.ibs_score}")
            if food.diverticulitis_score is not None:
                scores.append(f"Diverticulitis={food.diverticulitis_score}")
            if food.oxalates_score is not None:
                scores.append(f"Oxalates={food.oxalates_score}")
            if food.crystalluria_score is not None:
                scores.append(f"Crystalluria={food.crystalluria_score}")
            if scores:
                parts.append("| Scores: " + ", ".join(scores))
            if food.safety_category:
                parts.append(f"| Safety: {food.safety_category}")

            tags = ["food_guide", "food_entry"]
            if food.safety_category:
                tags.append(food.safety_category.lower())

            records.append(
                MKBRecord(
                    fact_type="note",
                    content=" ".join(parts)[:500],
                    structured=food.model_dump(exclude_none=True),
                    specialty=specialty,
                    source_type="document",
                    source_name=source_name,
                    trust_level=TRUST_CLINICAL,
                    confidence=confidence,
                    tier=TIER_ACTIVE,
                    extraction_method=method,
                    session_id=session_id,
                    tags=tags,
                )
            )

        for note in food_output.general_notes:
            records.append(
                MKBRecord(
                    fact_type="note",
                    content=note[:500],
                    structured={"text": note, "document_type": "food_guide"},
                    specialty=specialty,
                    source_type="document",
                    source_name=source_name,
                    trust_level=TRUST_CLINICAL,
                    confidence=confidence * 0.8,
                    tier=TIER_ACTIVE,
                    extraction_method=method,
                    session_id=session_id,
                    tags=["food_guide", "general_note"],
                )
            )

        return records

    def _process_chunked_clinical(
        self,
        full_text: str,
        source_name: str,
        specialty: str,
        session_id: str,
    ) -> list[MKBRecord]:
        chunks = self._split_into_chunks(full_text, chunk_size=6000)
        logger.info(f"Clinical '{source_name}': {len(chunks)} chunks to process")

        all_records: list[MKBRecord] = []
        for index, chunk in enumerate(chunks):
            stripped_text, _ = self.pii_stripper.strip(chunk)
            extraction = self.extractor.extract(stripped_text, specialty)
            records = self._to_records(
                extraction,
                f"{source_name} (chunk {index + 1})",
                specialty,
                session_id,
            )
            all_records.extend(records)
            logger.info(f"Clinical chunk {index + 1}/{len(chunks)}: {len(records)} records")

        logger.info(f"Clinical '{source_name}': {len(all_records)} total records")
        return all_records

    def _extract_text(self, pdf_path: Path) -> str:
        text, audit = self.extract_text_with_audit(pdf_path)
        self.last_text_audit = audit
        return text

    def extract_text_with_audit(self, pdf_path: Path) -> tuple[str, dict[str, Any]]:
        if DOCLING_AVAILABLE:
            return self._extract_docling(pdf_path)
        return self._extract_pymupdf(pdf_path)

    def _extract_docling(self, pdf_path: Path) -> tuple[str, dict[str, Any]]:
        try:
            result = self.converter.convert(str(pdf_path))
            text = result.document.export_to_markdown()
            quality = self._assess_text_quality(text)
            if self._text_requires_ocr(quality):
                logger.warning(
                    "docling text quality unusable for {}: status={} score={:.3f}; trying PyMuPDF OCR fallback",
                    pdf_path.name,
                    quality["status"],
                    quality["score"],
                )
                return self._extract_pymupdf(pdf_path)
            return text, self._build_document_text_audit(
                text_quality_status="readable_native",
                text_quality_score=float(quality["score"]),
                ocr_fallback_used=False,
                ocr_engine=None,
                ocr_text_length=0,
                page_audits=[],
            )
        except Exception as exc:
            logger.warning(f"docling failed: {exc}. Trying PyMuPDF.")
            return self._extract_pymupdf(pdf_path)

    def _extract_pymupdf(self, pdf_path: Path) -> tuple[str, dict[str, Any]]:
        try:
            import fitz

            doc = fitz.open(str(pdf_path))
            pages_text: list[str] = []
            page_audits: list[dict[str, Any]] = []
            for page_num, page in enumerate(doc, start=1):
                text, page_audit = self._extract_page_text(page, page_num=page_num)
                pages_text.append(f"[Page {page_num}]\n{text}")
                page_audits.append(page_audit)
            doc.close()

            document_text = "\n".join(pages_text)
            selected_quality = self._assess_text_quality(document_text)
            ocr_used = any(bool(item.get("ocr_fallback_used", False)) for item in page_audits)
            ocr_engine = next((item.get("ocr_engine") for item in page_audits if item.get("ocr_engine")), None)
            ocr_text_length = sum(int(item.get("ocr_text_length", 0)) for item in page_audits)
            text_quality_status = "ocr_fallback_applied" if ocr_used else str(selected_quality["status"])
            if not document_text.strip():
                text_quality_status = "empty"
            elif ocr_used and self._text_requires_ocr(selected_quality):
                text_quality_status = "unreadable_after_ocr"

            return document_text, self._build_document_text_audit(
                text_quality_status=text_quality_status,
                text_quality_score=float(selected_quality["score"]),
                ocr_fallback_used=ocr_used,
                ocr_engine=ocr_engine,
                ocr_text_length=ocr_text_length,
                page_audits=page_audits,
            )
        except Exception as exc:
            logger.error(f"PyMuPDF failed: {exc}")
            return "", self._build_document_text_audit(
                text_quality_status="empty",
                text_quality_score=0.0,
                ocr_fallback_used=False,
                ocr_engine=None,
                ocr_text_length=0,
                page_audits=[],
            )

    def _extract_page_text(self, page, *, page_num: int) -> tuple[str, dict[str, Any]]:
        native_text = page.get_text()
        native_quality = self._assess_text_quality(native_text)
        page_audit = {
            "page": page_num,
            "native_text_length": len(native_text.strip()),
            "native_text_quality_status": str(native_quality["status"]),
            "native_text_quality_score": float(native_quality["score"]),
            "text_quality_status": str(native_quality["status"]),
            "text_quality_score": float(native_quality["score"]),
            "ocr_fallback_used": False,
            "ocr_engine": None,
            "ocr_text_length": 0,
        }

        if not self._text_requires_ocr(native_quality):
            return native_text, page_audit

        ocr_text, ocr_engine = self._ocr_page(page)
        page_audit["ocr_engine"] = ocr_engine
        page_audit["ocr_text_length"] = len(ocr_text.strip())
        if not ocr_text.strip():
            return native_text, page_audit

        ocr_quality = self._assess_text_quality(ocr_text)
        page_audit["ocr_fallback_used"] = True
        page_audit["text_quality_status"] = (
            "ocr_fallback_applied" if not self._text_requires_ocr(ocr_quality) else "unreadable_after_ocr"
        )
        page_audit["text_quality_score"] = float(ocr_quality["score"])
        return ocr_text, page_audit

    def _ocr_page(self, page) -> tuple[str, str | None]:
        try:
            png_bytes = page.get_pixmap(dpi=200).tobytes("png")
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_dir_path = Path(temp_dir)
                image_path = temp_dir_path / "page.png"
                output_base = temp_dir_path / "ocr_output"
                image_path.write_bytes(png_bytes)
                completed = subprocess.run(
                    ["tesseract", str(image_path), str(output_base), "--psm", "6"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if completed.returncode != 0:
                    error_text = completed.stderr.strip() or completed.stdout.strip() or "tesseract_failed"
                    logger.warning(f"OCR failed: {error_text}")
                    return "", "tesseract_cli"
                text_path = output_base.with_suffix(".txt")
                if not text_path.exists():
                    logger.warning("OCR failed: tesseract output text file missing")
                    return "", "tesseract_cli"
                return text_path.read_text(encoding="utf-8", errors="ignore"), "tesseract_cli"
        except Exception as exc:
            logger.warning(f"OCR failed: {exc}")
            return "", None

    def _assess_text_quality(self, text: str) -> dict[str, Any]:
        stripped = text.strip()
        if not stripped:
            return {"status": "empty", "score": 0.0}

        total_chars = len(text)
        non_space_chars = max(1, sum(1 for char in text if not char.isspace()))
        printable_chars = sum(1 for char in text if char.isprintable() or char in _TEXT_QUALITY_ALLOWED_CONTROL)
        control_chars = sum(1 for char in text if ord(char) < 32 and char not in _TEXT_QUALITY_ALLOWED_CONTROL)
        alpha_numeric_chars = sum(1 for char in text if char.isalnum())
        symbol_chars = sum(1 for char in text if not char.isalnum() and not char.isspace())
        readable_tokens = [token.lower() for token in _TEXT_QUALITY_READABLE_TOKEN_RE.findall(text)]
        medical_token_hits = sum(1 for token in readable_tokens if token in _TEXT_QUALITY_MEDICAL_TOKENS)

        printable_ratio = printable_chars / max(1, total_chars)
        control_ratio = control_chars / max(1, total_chars)
        alpha_numeric_ratio = alpha_numeric_chars / non_space_chars
        symbol_ratio = symbol_chars / non_space_chars
        readable_token_score = min(len(readable_tokens) / 12.0, 1.0)
        medical_token_score = min(medical_token_hits / 3.0, 1.0)

        score = (
            (0.35 * printable_ratio)
            + (0.20 * alpha_numeric_ratio)
            + (0.20 * readable_token_score)
            + (0.25 * medical_token_score)
            - (0.60 * control_ratio)
            - (0.30 * max(0.0, symbol_ratio - 0.30))
        )
        score = max(0.0, min(1.0, round(score, 3)))

        if control_ratio >= 0.08 or printable_ratio < 0.85:
            status = "garbled_binary"
        elif symbol_ratio > 0.35:
            status = "mostly_symbols"
        elif len(readable_tokens) < 4:
            status = "too_few_readable_tokens"
        elif medical_token_hits == 0 and len(readable_tokens) < 8:
            status = "too_few_medical_tokens"
        else:
            status = "readable"

        return {
            "status": status,
            "score": score,
            "printable_ratio": printable_ratio,
            "control_ratio": control_ratio,
            "alpha_numeric_ratio": alpha_numeric_ratio,
            "symbol_ratio": symbol_ratio,
            "readable_token_count": len(readable_tokens),
            "medical_token_hits": medical_token_hits,
        }

    def _text_requires_ocr(self, quality: dict[str, Any]) -> bool:
        return str(quality.get("status")) != "readable" or float(quality.get("score", 0.0)) < 0.55

    def _build_document_text_audit(
        self,
        *,
        text_quality_status: str,
        text_quality_score: float,
        ocr_fallback_used: bool,
        ocr_engine: str | None,
        ocr_text_length: int,
        page_audits: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "text_quality_status": text_quality_status,
            "text_quality_score": round(float(text_quality_score), 3),
            "ocr_fallback_used": bool(ocr_fallback_used),
            "ocr_engine": ocr_engine,
            "ocr_text_length": int(ocr_text_length),
            "page_audits": page_audits,
        }

    def _to_records(
        self,
        extraction: ExtractionOutput,
        source_name: str,
        specialty: str,
        session_id: str,
    ) -> list[MKBRecord]:
        records = []
        method = extraction.extraction_method
        confidence = extraction.confidence
        trust = TRUST_CLINICAL

        for diag in extraction.diagnoses:
            records.append(
                MKBRecord(
                    fact_type="diagnosis",
                    content=f"Diagnosis: {diag.name}" + (f" ({diag.date})" if diag.date else ""),
                    structured=diag.model_dump(exclude_none=True),
                    specialty=specialty,
                    source_type="document",
                    source_name=source_name,
                    trust_level=trust,
                    confidence=confidence,
                    tier=TIER_ACTIVE,
                    extraction_method=method,
                    session_id=session_id,
                )
            )

        for med in extraction.medications:
            dose_str = f" {med.dose}" if med.dose else ""
            freq_str = f" {med.frequency}" if med.frequency else ""
            records.append(
                MKBRecord(
                    fact_type="medication",
                    content=f"Medication: {med.name}{dose_str}{freq_str}",
                    structured=med.model_dump(exclude_none=True),
                    specialty=specialty,
                    source_type="document",
                    source_name=source_name,
                    trust_level=trust,
                    confidence=confidence,
                    tier=TIER_ACTIVE,
                    extraction_method=method,
                    ddi_checked=False,
                    session_id=session_id,
                    tags=["medication"],
                )
            )

        for test in extraction.test_results:
            val_str = f": {test.value} {test.unit or ''}".rstrip() if test.value else ""
            records.append(
                MKBRecord(
                    fact_type="test_result",
                    content=f"Test: {test.test_name}{val_str}",
                    structured=test.model_dump(exclude_none=True),
                    specialty=specialty,
                    source_type="document",
                    source_name=source_name,
                    trust_level=trust,
                    confidence=confidence,
                    tier=TIER_ACTIVE,
                    extraction_method=method,
                    session_id=session_id,
                )
            )

        for note in extraction.notes:
            records.append(
                MKBRecord(
                    fact_type="note",
                    content=note[:500],
                    structured={"text": note},
                    specialty=specialty,
                    source_type="document",
                    source_name=source_name,
                    trust_level=trust,
                    confidence=confidence * 0.8,
                    tier=TIER_ACTIVE,
                    extraction_method=method,
                    session_id=session_id,
                )
            )

        return records
