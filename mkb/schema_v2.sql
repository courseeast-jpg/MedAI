-- ═══════════════════════════════════════════════════════════════════════
-- MedAI MKB — Schema v2 (hybrid extraction + edge-case support)
-- ═══════════════════════════════════════════════════════════════════════
--
-- This schema is ADDITIVE on top of the v1 schema defined in
-- ``mkb/sqlite_store.py``. All new columns default to NULL / 0 so existing
-- v1 rows continue to load without migration.
--
-- Migration entry point: ``python -m mkb.migrate_to_v2``
--
-- Covers these edge cases directly in schema:
--   EC 1  Conflicting facts              -> conflicts + provenance
--   EC 2  Temporal ambiguity             -> temporal_confidence, date_range_start/end
--   EC 3  Negation                       -> is_negated, negation_type
--   EC 4A Subject attribution            -> subject, family_history
--   EC 4B Certainty                      -> certainty
--   EC 6  Unit normalization             -> value, unit, normalized_value, normalized_unit
--   EC 8  Semantic aliases               -> terminology_aliases
--   EC 8  Time-series                    -> timeline
--   EC 12 OCR quality                    -> ocr_confidence, ocr_errors_detected
--   EC 14 Provenance / occurrences       -> provenance, occurrence_count
--   EC 15 Source trust                   -> trust_level, tier
--   EC 17 Hypothesis tier                -> tier
-- ═══════════════════════════════════════════════════════════════════════

-- ── Facts table (v2 extension of records) ────────────────────────────
CREATE TABLE IF NOT EXISTS facts (
    id                      TEXT PRIMARY KEY,
    fact_type               TEXT NOT NULL,                 -- diagnosis|medication|test_result|symptom|measurement|note
    entity_type             TEXT NOT NULL,                 -- canonical kind (same as fact_type, separate field for flexibility)
    entity_name             TEXT NOT NULL,                 -- surface form, as extracted
    canonical_name          TEXT,                          -- resolved canonical term from terminology_aliases
    value                   TEXT,                          -- raw value as a string (may be numeric, range, or free-form)
    unit                    TEXT,                          -- raw unit
    normalized_value        REAL,                          -- numeric value normalized to canonical unit
    normalized_unit         TEXT,                          -- canonical unit (kg, %, °C, mmol/L, ...)
    date                    TEXT,                          -- ISO date of the event (best estimate)
    date_range_start        TEXT,                          -- lower bound when date is approximate
    date_range_end          TEXT,                          -- upper bound when date is approximate
    temporal_confidence     REAL DEFAULT 1.0,              -- 0..1 — EC 2

    -- Edge-case classification fields
    is_negated              INTEGER DEFAULT 0,             -- EC 3
    negation_type           TEXT,                          -- denied|ruled_out|absent|historical
    subject                 TEXT DEFAULT 'patient',        -- EC 4A: patient|mother|father|sibling|maternal_grandparent|paternal_grandparent|other
    certainty               TEXT DEFAULT 'confirmed',      -- EC 4B: confirmed|suspected|ruled_out|differential|historical

    -- OCR quality (EC 12)
    ocr_confidence          REAL,                          -- 0..1, null if source is digital text
    ocr_errors_detected     INTEGER DEFAULT 0,

    -- Trust & tier (EC 15, 17)
    trust_level             INTEGER DEFAULT 3,             -- 1 clinical | 2 peer | 3 AI | 4 reputable | 5 unverified
    confidence              REAL DEFAULT 0.5,
    tier                    TEXT DEFAULT 'active',         -- active|hypothesis|quarantined|superseded
    status                  TEXT DEFAULT 'active',         -- active|rejected|merged|superseded|quarantined

    -- Provenance / lifecycle (EC 14)
    occurrence_count        INTEGER DEFAULT 1,
    first_recorded          TEXT NOT NULL,
    last_confirmed          TEXT NOT NULL,

    -- Source + classification metadata
    specialty               TEXT DEFAULT 'general',
    source_type             TEXT NOT NULL,                 -- document|ai_response|manual|web|guideline
    source_name             TEXT DEFAULT '',
    source_url              TEXT,
    source_document_id      TEXT,                          -- FK -> documents.id (optional)
    extraction_method       TEXT DEFAULT 'hybrid',         -- hybrid|gemini|claude|rules_based|manual
    raw_text                TEXT,                          -- sentence/span the fact was extracted from

    -- Relationships
    parent_fact_id          TEXT,                          -- for derived facts (e.g., timeline points)
    superseded_by           TEXT,                          -- new fact id when this was replaced

    -- Review flags
    requires_review         INTEGER DEFAULT 0,
    conflict_id             TEXT,                          -- FK -> conflicts.id when quarantined

    session_id              TEXT DEFAULT '',
    metadata_json           TEXT DEFAULT '{}'              -- extension slot
);

CREATE INDEX IF NOT EXISTS idx_facts_entity        ON facts(entity_type, entity_name);
CREATE INDEX IF NOT EXISTS idx_facts_canonical     ON facts(canonical_name);
CREATE INDEX IF NOT EXISTS idx_facts_date          ON facts(date);
CREATE INDEX IF NOT EXISTS idx_facts_tier          ON facts(tier);
CREATE INDEX IF NOT EXISTS idx_facts_status        ON facts(status);
CREATE INDEX IF NOT EXISTS idx_facts_subject       ON facts(subject);
CREATE INDEX IF NOT EXISTS idx_facts_certainty     ON facts(certainty);
CREATE INDEX IF NOT EXISTS idx_facts_negated       ON facts(is_negated);
CREATE INDEX IF NOT EXISTS idx_facts_requires_rev  ON facts(requires_review);
CREATE INDEX IF NOT EXISTS idx_facts_source_doc    ON facts(source_document_id);

-- ── Timeline (EC 8: value changes over time) ─────────────────────────
CREATE TABLE IF NOT EXISTS timeline (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_type       TEXT NOT NULL,
    canonical_name    TEXT NOT NULL,                       -- e.g. "hba1c"
    subject           TEXT DEFAULT 'patient',
    value             REAL,
    unit              TEXT,
    raw_value         TEXT,                                -- original string (e.g. "55 mmol/mol")
    date              TEXT NOT NULL,
    source_fact_id    TEXT NOT NULL,
    confidence        REAL DEFAULT 0.8,
    created_at        TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_timeline_entity     ON timeline(entity_type, canonical_name);
CREATE INDEX IF NOT EXISTS idx_timeline_date       ON timeline(date);
CREATE INDEX IF NOT EXISTS idx_timeline_subject    ON timeline(subject);
CREATE UNIQUE INDEX IF NOT EXISTS idx_timeline_point
    ON timeline(canonical_name, subject, date, source_fact_id);

-- ── Conflicts (EC 1) ─────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS conflicts (
    id              TEXT PRIMARY KEY,
    fact1_id        TEXT,
    fact2_id        TEXT,
    fact1_snapshot  TEXT NOT NULL,                         -- JSON
    fact2_snapshot  TEXT NOT NULL,                         -- JSON
    conflict_type   TEXT NOT NULL,                         -- value_mismatch|type_mismatch|drug_interaction|temporal|implausible_change
    severity        TEXT NOT NULL DEFAULT 'medium',        -- low|medium|high|critical
    reason          TEXT DEFAULT '',
    status          TEXT NOT NULL DEFAULT 'pending',       -- pending|resolved|dismissed
    resolution      TEXT,                                  -- JSON of the user's choice
    resolution_notes TEXT,
    created_at      TEXT NOT NULL,
    resolved_at     TEXT,
    session_id      TEXT DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_conflicts_status     ON conflicts(status);
CREATE INDEX IF NOT EXISTS idx_conflicts_severity   ON conflicts(severity);
CREATE INDEX IF NOT EXISTS idx_conflicts_fact1      ON conflicts(fact1_id);
CREATE INDEX IF NOT EXISTS idx_conflicts_fact2      ON conflicts(fact2_id);

-- ── Family history (EC 4A) ───────────────────────────────────────────
CREATE TABLE IF NOT EXISTS family_history (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    fact_id           TEXT NOT NULL,                       -- FK -> facts.id
    relation          TEXT NOT NULL,                       -- mother|father|sibling|maternal_grandparent|paternal_grandparent|other
    condition         TEXT NOT NULL,
    age_at_onset      INTEGER,
    age_at_death      INTEGER,
    notes             TEXT,
    created_at        TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_family_fact_id      ON family_history(fact_id);
CREATE INDEX IF NOT EXISTS idx_family_relation     ON family_history(relation);

-- ── Provenance (EC 14: track every source appearance of a fact) ──────
CREATE TABLE IF NOT EXISTS provenance (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    fact_id           TEXT NOT NULL,
    source_type       TEXT NOT NULL,                       -- document|ai_response|manual|web|guideline
    source_name       TEXT,
    source_url        TEXT,
    source_document_id TEXT,
    page_number       INTEGER,
    raw_text          TEXT,
    extracted_at      TEXT NOT NULL,
    confidence        REAL DEFAULT 0.8
);

CREATE INDEX IF NOT EXISTS idx_provenance_fact     ON provenance(fact_id);
CREATE INDEX IF NOT EXISTS idx_provenance_source   ON provenance(source_type, source_name);
CREATE INDEX IF NOT EXISTS idx_provenance_doc      ON provenance(source_document_id);

-- ── Audit log (append-only) ──────────────────────────────────────────
CREATE TABLE IF NOT EXISTS audit_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type      TEXT NOT NULL,                         -- fact_inserted|fact_merged|conflict_quarantined|conflict_resolved|promotion|demotion|...
    entity_id       TEXT,                                  -- fact_id or conflict_id depending on event
    entity_kind     TEXT,                                  -- 'fact' | 'conflict' | 'document'
    actor           TEXT DEFAULT 'system',                 -- 'system' or a user identifier
    details_json    TEXT DEFAULT '{}',
    timestamp       TEXT NOT NULL,
    session_id      TEXT DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_audit_event         ON audit_log(event_type);
CREATE INDEX IF NOT EXISTS idx_audit_entity        ON audit_log(entity_id);
CREATE INDEX IF NOT EXISTS idx_audit_kind          ON audit_log(entity_kind);
CREATE INDEX IF NOT EXISTS idx_audit_timestamp     ON audit_log(timestamp);

-- ── Terminology aliases (EC 5, EC 8) ─────────────────────────────────
CREATE TABLE IF NOT EXISTS terminology_aliases (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    canonical         TEXT NOT NULL,                       -- "hypertension"
    alias             TEXT NOT NULL,                       -- "HTN" | "high blood pressure"
    entity_type       TEXT,                                -- "diagnosis" | "test_result" | null (any)
    language          TEXT DEFAULT 'en',
    trust_level       INTEGER DEFAULT 3,                   -- how strongly to trust this mapping
    source            TEXT DEFAULT 'builtin',              -- builtin|user|learned
    created_at        TEXT NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_aliases_pair
    ON terminology_aliases(canonical, alias, language, entity_type);
CREATE INDEX IF NOT EXISTS idx_aliases_canonical   ON terminology_aliases(canonical);
CREATE INDEX IF NOT EXISTS idx_aliases_alias       ON terminology_aliases(alias);

-- Seed the most common aliases. ``INSERT OR IGNORE`` prevents duplicates on
-- repeat runs. More aliases can be added via the UI or learning pipeline.
INSERT OR IGNORE INTO terminology_aliases (canonical, alias, entity_type, language, source, created_at) VALUES
    ('hypertension', 'htn',                   'diagnosis',  'en', 'builtin', datetime('now')),
    ('hypertension', 'high blood pressure',   'diagnosis',  'en', 'builtin', datetime('now')),
    ('hypertension', 'elevated bp',           'diagnosis',  'en', 'builtin', datetime('now')),
    ('diabetes mellitus', 'dm',               'diagnosis',  'en', 'builtin', datetime('now')),
    ('diabetes mellitus', 'diabetes',         'diagnosis',  'en', 'builtin', datetime('now')),
    ('diabetes mellitus', 't2dm',             'diagnosis',  'en', 'builtin', datetime('now')),
    ('diabetes mellitus', 't1dm',             'diagnosis',  'en', 'builtin', datetime('now')),
    ('myocardial infarction', 'mi',           'diagnosis',  'en', 'builtin', datetime('now')),
    ('myocardial infarction', 'ami',          'diagnosis',  'en', 'builtin', datetime('now')),
    ('myocardial infarction', 'heart attack', 'diagnosis',  'en', 'builtin', datetime('now')),
    ('hba1c', 'a1c',                          'test_result','en', 'builtin', datetime('now')),
    ('hba1c', 'glycated hemoglobin',          'test_result','en', 'builtin', datetime('now')),
    ('hba1c', 'glycated haemoglobin',         'test_result','en', 'builtin', datetime('now')),
    ('hemoglobin', 'hgb',                     'test_result','en', 'builtin', datetime('now')),
    ('hemoglobin', 'hb',                      'test_result','en', 'builtin', datetime('now')),
    ('blood pressure', 'bp',                  'measurement','en', 'builtin', datetime('now')),
    ('heart rate', 'hr',                      'measurement','en', 'builtin', datetime('now')),
    ('heart rate', 'pulse',                   'measurement','en', 'builtin', datetime('now')),
    ('weight', 'body weight',                 'measurement','en', 'builtin', datetime('now')),
    ('temperature', 'temp',                   'measurement','en', 'builtin', datetime('now'));

-- ── Documents (optional, for tier assignment tracking) ───────────────
CREATE TABLE IF NOT EXISTS documents (
    id                TEXT PRIMARY KEY,
    source_path       TEXT NOT NULL,
    source_name       TEXT NOT NULL,
    document_type     TEXT,                                -- clinical_note|lab_report|imaging|food_guide|generic
    specialty         TEXT DEFAULT 'general',
    trust_level       INTEGER DEFAULT 3,
    default_tier      TEXT DEFAULT 'active',               -- active|hypothesis
    language          TEXT DEFAULT 'en',
    ocr_confidence    REAL,
    page_count        INTEGER,
    ingested_at       TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_documents_type      ON documents(document_type);
CREATE INDEX IF NOT EXISTS idx_documents_specialty ON documents(specialty);
