"""
MedAI v1.1 — Web Ingestion Pipeline
Fetches and processes third-party sources: HTML pages, online PDFs, PubMed.
ENABLE_WEB_INGESTION must be True (default).
User triggers only — never auto-crawls.
"""
from pathlib import Path
from typing import Optional
from loguru import logger

import requests
from bs4 import BeautifulSoup

from app.config import ENABLE_WEB_INGESTION, TRUST_PEER_REVIEW, TRUST_REPUTABLE, TRUST_UNVERIFIED
from app.schemas import MKBRecord
from extraction.pii_stripper import PIIStripper
from extraction.extractor import Extractor


TRUSTED_DOMAINS = {
    "pubmed.ncbi.nlm.nih.gov": TRUST_PEER_REVIEW,
    "pmc.ncbi.nlm.nih.gov": TRUST_PEER_REVIEW,
    "who.int": TRUST_PEER_REVIEW,
    "ilae.org": TRUST_PEER_REVIEW,
    "aan.com": TRUST_PEER_REVIEW,
    "aga.com": TRUST_PEER_REVIEW,
    "auanet.org": TRUST_PEER_REVIEW,
    "mayoclinic.org": TRUST_REPUTABLE,
    "nhs.uk": TRUST_REPUTABLE,
    "medlineplus.gov": TRUST_REPUTABLE,
    "epilepsy.com": TRUST_REPUTABLE,
}


class WebPipeline:
    def __init__(self, extractor: Extractor, pii_stripper: PIIStripper):
        self.extractor = extractor
        self.pii = pii_stripper

    def process_url(self, url: str, specialty: str = "general", session_id: str = "") -> list[MKBRecord]:
        if not ENABLE_WEB_INGESTION:
            logger.info("Web ingestion disabled (ENABLE_WEB_INGESTION=False)")
            return []

        trust_level = self._get_trust_level(url)
        raw_text = self._fetch(url)
        if not raw_text:
            return []

        stripped, _ = self.pii.strip(raw_text)
        extraction = self.extractor.extract(stripped, specialty)
        records = []

        for diag in extraction.diagnoses:
            records.append(MKBRecord(
                fact_type="diagnosis",
                content=f"Web source: {diag.name}",
                structured=diag.model_dump(exclude_none=True),
                specialty=specialty,
                source_type="web",
                source_name=url[:100],
                source_url=url,
                trust_level=trust_level,
                confidence=extraction.confidence * 0.75,
                tier="hypothesis",  # All web sources → hypothesis
                extraction_method=extraction.extraction_method,
                session_id=session_id,
            ))

        for rec in extraction.recommendations:
            records.append(MKBRecord(
                fact_type="recommendation",
                content=f"Web recommendation: {rec[:300]}",
                structured={"text": rec, "source_url": url},
                specialty=specialty,
                source_type="web",
                source_name=url[:100],
                source_url=url,
                trust_level=trust_level,
                confidence=extraction.confidence * 0.70,
                tier="hypothesis",
                extraction_method=extraction.extraction_method,
                session_id=session_id,
            ))

        logger.info(f"Web pipeline: {len(records)} records from {url[:60]}")
        return records

    def _fetch(self, url: str) -> Optional[str]:
        try:
            headers = {"User-Agent": "MedAI/1.1 (personal medical assistant; non-commercial)"}
            resp = requests.get(url, headers=headers, timeout=15)
            resp.raise_for_status()

            if "pdf" in resp.headers.get("content-type", "").lower():
                return self._extract_pdf_bytes(resp.content)

            soup = BeautifulSoup(resp.text, "html.parser")
            # Remove navigation, scripts, styles
            for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
                tag.decompose()
            # Try main content areas first
            for selector in ["article", "main", "#content", ".content", "body"]:
                element = soup.select_one(selector)
                if element and len(element.get_text(strip=True)) > 200:
                    return element.get_text(separator=" ", strip=True)[:8000]
            return soup.get_text(separator=" ", strip=True)[:8000]

        except Exception as e:
            logger.warning(f"Web fetch failed for {url}: {e}")
            return None

    def _extract_pdf_bytes(self, content: bytes) -> Optional[str]:
        try:
            import fitz
            import io
            doc = fitz.open(stream=content, filetype="pdf")
            return "\n".join(page.get_text() for page in doc)[:8000]
        except Exception as e:
            logger.warning(f"PDF bytes extraction failed: {e}")
            return None

    def _get_trust_level(self, url: str) -> int:
        from urllib.parse import urlparse
        domain = urlparse(url).netloc.lstrip("www.")
        for trusted_domain, level in TRUSTED_DOMAINS.items():
            if trusted_domain in domain:
                return level
        return TRUST_UNVERIFIED
