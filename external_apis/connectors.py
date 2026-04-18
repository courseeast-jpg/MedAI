"""
MedAI v1.1 — External API Connectors (Track C)
BaseConnector interface + DxGPT (active) + stubs for all others.
All connectors receive only PII-stripped payloads.
"""
import asyncio
import aiohttp
from abc import ABC, abstractmethod
from typing import List, Optional
from loguru import logger

from app.schemas import AnonymizedPayload, ConnectorResponse, DDIFinding
from app.config import CONNECTOR_TIMEOUT_SEC


# ── Base Interface ─────────────────────────────────────────────────────────────

class BaseConnector(ABC):
    name: str = "base"

    @abstractmethod
    async def query(self, payload: AnonymizedPayload) -> ConnectorResponse:
        pass


# ── DxGPT Connector (ACTIVE) ──────────────────────────────────────────────────

class DxGPTConnector(BaseConnector):
    name = "dxgpt"
    BASE_URL = "https://dxgpt.app/api"

    async def query(self, payload: AnonymizedPayload) -> ConnectorResponse:
        prompt = self._build_prompt(payload)
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.BASE_URL}/diagnose",
                    json={"query": prompt, "specialty": payload.specialty},
                    timeout=aiohttp.ClientTimeout(total=CONNECTOR_TIMEOUT_SEC),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return ConnectorResponse(
                            connector_name=self.name,
                            content=data.get("response", ""),
                            confidence=data.get("confidence"),
                            citations=data.get("sources", []),
                            raw_response=data,
                            status="ok",
                        )
                    else:
                        logger.warning(f"DxGPT HTTP {resp.status}")
                        return self._error_response(f"HTTP {resp.status}")
        except asyncio.TimeoutError:
            return ConnectorResponse(connector_name=self.name, status="timeout")
        except Exception as e:
            logger.warning(f"DxGPT error: {e}")
            # Graceful fallback: return structured empty response
            return self._error_response(str(e))

    def _build_prompt(self, payload: AnonymizedPayload) -> str:
        context = "\n".join(f"- {f}" for f in payload.context_facts[:5])
        meds = ", ".join(payload.active_medications[:10]) if payload.active_medications else "none"
        return (
            f"Query: {payload.query_text}\n"
            f"Specialty: {payload.specialty}\n"
            f"Relevant context:\n{context}\n"
            f"Active medications: {meds}"
        )

    def _error_response(self, reason: str) -> ConnectorResponse:
        return ConnectorResponse(connector_name=self.name, status="error",
                                  raw_response={"error": reason})


# ── SAGE Connector (STUB — active in Phase 2) ─────────────────────────────────

class SAGEConnector(BaseConnector):
    name = "sage"

    async def query(self, payload: AnonymizedPayload) -> ConnectorResponse:
        logger.debug("SAGE connector: stub (activate in Phase 2)")
        return ConnectorResponse(
            connector_name=self.name,
            status="stub",
            content=None,
            raw_response={"note": "SAGE connector not yet activated"},
        )


# ── PatientNotes DDI Connector ────────────────────────────────────────────────

class PatientNotesDDIConnector(BaseConnector):
    name = "patientnotes_ddi"
    BASE_URL = "https://patientnotes.ai/api/interactions"

    async def query(self, payload: AnonymizedPayload) -> ConnectorResponse:
        """Used by MedicationSafetyGate directly, not standard query flow."""
        return ConnectorResponse(connector_name=self.name, status="ok", content="DDI connector (see check_interactions)")

    def check_interactions(
        self, new_meds: List[str], active_meds: List[str]
    ) -> List[DDIFinding]:
        """Synchronous DDI check. Returns list of findings."""
        import requests
        try:
            resp = requests.post(
                self.BASE_URL,
                json={"drugs": new_meds + active_meds},
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json()
                return self._parse_findings(data, new_meds, active_meds)
        except Exception as e:
            logger.warning(f"PatientNotes DDI error: {e}")
        return []

    def _parse_findings(self, data: dict, new_meds: List[str], active_meds: List[str]) -> List[DDIFinding]:
        findings = []
        for interaction in data.get("interactions", []):
            findings.append(DDIFinding(
                drug_a=interaction.get("drug1", ""),
                drug_b=interaction.get("drug2", ""),
                severity=interaction.get("severity", "LOW").upper(),
                mechanism=interaction.get("description"),
                management=interaction.get("management"),
            ))
        return findings


# ── Doctronic Connector (STUB) ────────────────────────────────────────────────

class DoctronicConnector(BaseConnector):
    name = "doctronic"

    async def query(self, payload: AnonymizedPayload) -> ConnectorResponse:
        logger.debug("Doctronic connector: stub (activate in Phase 2)")
        return ConnectorResponse(
            connector_name=self.name,
            status="stub",
            content=None,
        )


# ── Claude Synthesis Connector ────────────────────────────────────────────────

class ClaudeSynthesizer:
    """Not a connector — local synthesis using full MKB context (no PII stripping needed)."""

    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
        self._available = bool(api_key)

    @property
    def available(self) -> bool:
        return self._available

    def synthesize(self, query: str, scored_responses, mkb_context) -> str:
        import anthropic
        client = anthropic.Anthropic(api_key=self.api_key)

        ctx_facts = "\n".join(f"- [{r.tier.upper()}] {r.content}"
                               for r in mkb_context.structured_facts[:8])
        api_summaries = "\n".join(
            f"- {r.connector_name} (score={r.final_score:.2f}): {(r.content or '')[:400]}"
            for r in scored_responses[:3]
        )

        prompt = f"""You are a medical decision support assistant. Answer the following query using the provided context.

Query: {query}

Patient's medical records (from local MKB):
{ctx_facts or 'No relevant records found.'}

External AI responses (ranked by quality score):
{api_summaries or 'No external responses available.'}

Provide a clear, evidence-grounded response. Note the source of each key claim.
Clearly distinguish between: (1) what the patient's records show, (2) what external AI suggests.
End with: 'This is decision support, not a diagnosis. Consult your physician.'"""

        response = client.messages.create(
            model=self.model,
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text

    def mark_unavailable(self):
        self._available = False

    def mark_available(self):
        self._available = True


# ── Connector Registry ─────────────────────────────────────────────────────────

def build_connector_registry() -> dict:
    """Returns dict of {name: connector_instance} for all registered connectors."""
    return {
        "dxgpt":          DxGPTConnector(),
        "sage":           SAGEConnector(),
        "patientnotes_ddi": PatientNotesDDIConnector(),
        "doctronic":      DoctronicConnector(),
    }
