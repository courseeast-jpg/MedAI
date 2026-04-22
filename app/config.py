"""
MedAI Platform v1.1 — Global Configuration
All feature flags default to MVP activation state.
"""
import os
from pathlib import Path
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

# ── Feature Flags ─────────────────────────────────────────────────────────────
# Progressive activation model: all layers present, controlled by flags

ENABLE_GRAPH          = False   # NetworkX knowledge graph (Phase 3)
ENABLE_LOCAL_LLM      = False   # Ollama local model (Phase 4)
ENRICH_PROMOTE        = False   # Auto-promote hypothesis→active (Phase 2)
ENABLE_EPUB           = False   # ePub ingestion (Phase 2)
ENABLE_YOUTUBE        = False   # YouTube transcripts (Phase 2)
ENABLE_WEB_INGESTION  = True    # HTML + PDF web fetch (MVP)
ENABLE_ENRICHMENT     = True    # Enrichment engine — hypothesis tier only

# Active connectors — only listed connectors make real API calls
# Others are present as stubs returning structured empty responses
ACTIVE_CONNECTORS     = ["dxgpt"]  # Expand in Phase 2

# Confidence thresholds
SAFE_MODE_THRESHOLD       = 0.40   # Below this → safe mode
RESPONSE_DISCARD_THRESHOLD = 0.30  # Below this → discard response
DEDUP_SIMILARITY_THRESHOLD = 0.92  # ChromaDB cosine → deduplicate
MIN_CONFIDENCE_CLAUDE     = 0.40   # Min confidence for Claude extraction
MIN_CONFIDENCE_RULES      = 0.35   # Min confidence for rules-based extraction

# Connector timeouts
CONNECTOR_TIMEOUT_SEC = 12
CLAUDE_HEALTH_CHECK_INTERVAL_SEC = 60
CLAUDE_TIMEOUT_SEC = 30

# Trust levels
TRUST_CLINICAL    = 1
TRUST_PEER_REVIEW = 2
TRUST_AI          = 3
TRUST_REPUTABLE   = 4
TRUST_UNVERIFIED  = 5

# Tiers
TIER_ACTIVE      = "active"
TIER_HYPOTHESIS  = "hypothesis"
TIER_QUARANTINED = "quarantined"
TIER_SUPERSEDED  = "superseded"

# DDI severity
DDI_HIGH   = "HIGH"
DDI_MEDIUM = "MEDIUM"
DDI_LOW    = "LOW"
DDI_NONE   = "NONE"

# Paths
BASE_DIR              = Path(__file__).parent
DB_PATH               = Path(os.getenv("DB_PATH", "data/mkb.db"))
CHROMA_PATH           = Path(os.getenv("CHROMA_PATH", "data/chroma"))
PDF_STORAGE_PATH      = Path(os.getenv("PDF_STORAGE_PATH", "data/pdfs"))
PENDING_QUEUE_PATH    = Path(os.getenv("PENDING_QUEUE_PATH", "data/pending/enrichment_queue.jsonl"))
SPECIALTIES_DIR       = BASE_DIR / "specialties"

# API
ANTHROPIC_API_KEY     = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL          = "claude-sonnet-4-20250514"
CLAUDE_MAX_TOKENS     = 2000

# Local LLM (Ollama) — primary extractor backend
OLLAMA_MODEL          = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_TIMEOUT_SEC    = int(os.getenv("OLLAMA_TIMEOUT_SEC", "180"))

# Embedding model (runs locally, no external calls)
EMBEDDING_MODEL       = "all-MiniLM-L6-v2"

# Allowed specialties
ALLOWED_SPECIALTIES   = ["neurology", "epilepsy", "gastroenterology", "urology", "general"]
ALLOWED_TASK_TYPES    = ["differential_diagnosis", "medication_check", "evidence_lookup", "symptom_tracking", "general_query"]
ALLOWED_FACT_TYPES    = ["diagnosis", "medication", "test_result", "symptom", "note", "recommendation", "relationship", "event"]
