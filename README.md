# MedAI Platform v1.1

Personal multi-specialty medical AI — local-first, self-enriching, privacy-preserving.

> Decision support only. Not a medical device. All outputs require clinical verification.

---

## Prerequisites (one-time system installs)

```bash
# Ubuntu / Debian
sudo apt-get install -y tesseract-ocr sqlcipher python3.11

# macOS
brew install tesseract sqlcipher python@3.11
```

---

## Setup (one command)

```bash
# 1. Clone
git clone https://github.com/[user]/medai && cd medai

# 2. Add API key
cp .env.example .env
# Edit .env — set ANTHROPIC_API_KEY and DB_ENCRYPTION_KEY

# 3. Bootstrap
make install
```

`make install` handles: pip deps, spaCy model, SQLite init, ChromaDB init, embedding model download.

---

## Run

```bash
make run
# Opens http://localhost:8501
```

---

## Test

```bash
make test           # Full test suite
make test-golden    # Golden test set only (8 critical cases)
```

---

## Architecture

```
Ingestion (docling PDF + Presidio PII strip + Claude extraction)
    ↓
Quality Gate (dedup + conflict detection)
    ↓
Truth Resolution Engine (7 priority rules)
    ↓
Medication Safety Gate Layer 2 (DDI hard block)
    ↓
MKB Write (SQLite + ChromaDB, encrypted, local only)
    ↓
Decision Engine (classify → retrieve → execute → score → consensus)
    ↓  DDI Layer 1 fires here (evidence modifier)
Response Scoring (4-dimension formula)
    ↓
Controlled Enrichment (hypothesis tier only)
    ↓
UI (5 safety surfaces: confidence, conflicts, trust, degraded mode, DDI warnings)
```

## Feature Flags (config.py)

| Flag | Default | Purpose |
|------|---------|---------|
| `ENABLE_GRAPH` | False | Knowledge graph (Phase 3) |
| `ENRICH_PROMOTE` | False | Auto-promote hypothesis→active (Phase 2) |
| `ACTIVE_CONNECTORS` | `["dxgpt"]` | Expand in Phase 2 |
| `ENABLE_ENRICHMENT` | True | AI response enrichment |
| `ENABLE_WEB_INGESTION` | True | Web fetch pipeline |

## MVP Exit Criteria

All 8 golden test cases must pass before running against real medical data:

1. Real PDF → structured SQLite records within 60s
2. PII strip verified: no names/dates/facilities in outbound payload
3. Conflict: two contradicting facts → single resolved record + ledger entry
4. DDI HIGH block: incompatible drugs → blocked + user confirmation UI
5. Claude offline → safe mode banner + MKB-only response
6. Low score response → discarded + ledger entry
7. PII strip: patient name/DOB/facility not present in extracted text
8. Trust hierarchy: clinical doc beats AI fact, confidence=0.95

```bash
make test-golden
# All 8 must pass before using with real data
```

---

## Specialties

| Specialty | Status | Primary connector |
|-----------|--------|-------------------|
| Neurology | Active | DxGPT |
| Epilepsy | Active | SAGE (stub→activate Phase 2) |
| Gastroenterology | Stub | DxGPT (Phase 2) |
| Urology | Stub | DxGPT (Phase 2) |

Adding a new specialty: create `specialties/[name]/config.yaml` + `plugin.py`. Zero core changes.
