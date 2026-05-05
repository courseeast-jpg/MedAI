# MedAI Architecture

This document describes the 18 edge cases the extraction and MKB pipeline is
designed to handle, the schema decisions that back them, the deduplication
strategies, and the conflict resolution workflow.

---

## 18 Edge Cases

| # | Edge Case | Handled by | How |
|---|-----------|-----------|-----|
| 1 | Conflicting facts | `mkb.deduplication_engine.detect_conflict` + `mkb.conflict_resolver` | Same-day value mismatches, mutually exclusive types, implausible changes, drug interactions, temporal impossibilities → quarantined with severity |
| 2 | Temporal ambiguity (“2 weeks ago”) | `extraction.hybrid_extractor.extract_temporal_info` | Explicit dates get confidence 1.0; relative phrases are resolved against `doc_date` with a reduced confidence and date-range hint |
| 3 | Negation (“denied”, “ruled out”, “no history”) | `extraction.hybrid_extractor.detect_negation` | Returns `(is_negated, neg_type)`; stored on the fact so it is never merged with a positive assertion |
| 4A | Subject attribution (patient vs family) | `extraction.hybrid_extractor.identify_subject` | Distinguishes patient / mother / father / sibling / paternal-maternal; family history cannot pollute patient problem list |
| 4B | Certainty level | `extraction.hybrid_extractor.assess_certainty` | `confirmed / suspected / ruled_out / differential / historical` attached as a first-class field |
| 5 | Acronym ambiguity (MS = multiple sclerosis vs mitral stenosis) | Canonical term resolution via `terminology_aliases` | Requires specialty + context to disambiguate |
| 6 | Unit conversion | `extraction.hybrid_extractor.normalize_measurement` | lbs↔kg, °F↔°C, mmol/mol↔%, mg/dL↔mmol/L; normalized value stored beside raw |
| 7 | OCR digit/letter confusion | `extraction.ocr_validator.OCRValidator` | Flags `50Omg`, `l00`, `metf0rmin` as low-confidence tokens |
| 8 | Semantic duplicates + time-series | `mkb.deduplication_engine.find_semantic_match` / `find_timeseries_match` | `hypertension`=`HTN`; 90kg→85kg→82kg stored as timeline |
| 9 | Drug interactions | `decision.medication_safety` + `deduplication_engine` drug pair table | Critical severity → block write; medium → warn |
| 10 | Missing / partial values | Schema permits `value=None`; optional fields null-safe | Prevents exclusion of facts with partial data |
| 11 | Multi-language (Russian source) | Extraction prompt translates to English | Food guide and clinical prompts both enforce English output |
| 12 | OCR quality | `OCRValidator.validate_ocr_quality` | Returns confidence 0..1, errors_detected, up to 5 examples; 5 sub-checks (number/letter, misspellings, special-char ratio, missing spaces, line spacing) |
| 13 | Encoding / Cyrillic normalization | `PIIStripper` + extractor prompts | Normalised before Presidio |
| 14 | Provenance / repeated appearances | Exact match merges with `occurrence_count++` | Keeps source list; powers trust scoring |
| 15 | Source trust hierarchy | `MKBRecord.trust_level` + Truth Resolution Rule 1-2 | Clinical > peer-reviewed > AI > web > unverified |
| 16 | Degraded mode | `SystemState.safe_mode` | UI banner + MKB-only response path |
| 17 | Hypothesis tier | `MKBRecord.tier='hypothesis'` | AI-derived facts never reach synthesis unless promoted |
| 18 | PII leakage | `extraction.pii_stripper.PIIStripper` | Strips PII before any payload leaves the machine |

---

## Database Schema Decisions

### `records`

Stores the primary MKB facts. Key columns for edge-case support:

- `tier` — `active | hypothesis | quarantined | superseded`. Drives visibility
  in synthesis and UI.
- `status` — soft life-cycle flag (`active | rejected | merged | ...`).
- `requires_review` — boolean short-circuit for the Conflict Review panel.
- `resolution_id` — links to the conflict that produced this row (if merged).
- `structured_json` — holds extensible metadata: temporal confidence,
  certainty level, subject, negation, unit-normalized value, OCR confidence.

### `conflicts` (new, Phase 5)

```
id              TEXT PRIMARY KEY
fact1_id        TEXT
fact2_id        TEXT
fact1_snapshot  TEXT NOT NULL   -- JSON
fact2_snapshot  TEXT NOT NULL   -- JSON
conflict_type   TEXT            -- value_mismatch|type_mismatch|
                                -- drug_interaction|temporal|implausible_change
severity        TEXT            -- low|medium|high|critical
reason          TEXT
status          TEXT            -- pending|resolved|dismissed
resolution      TEXT            -- JSON of {choice, merged_value, notes}
created_at      TEXT
resolved_at     TEXT
```

Indexes on `status`, `severity`, `fact1_id`, `fact2_id` keep the
review panel fast even with many conflicts.

### `conflict_audit_log`

Append-only audit stream — `quarantined`, `resolved:fact1`, `resolved:merge`, etc.
Supports reproducing the resolution sequence after the fact.

### Schema v2 migration

The new columns are additive; existing `records` rows retain their defaults.
No destructive ALTERs are performed. `ConflictResolver.__init__` creates the
two new tables via `CREATE TABLE IF NOT EXISTS`, so the migration is idempotent.

---

## Deduplication Strategies (detail)

All strategies operate on plain dict "facts" so they can be invoked from any
pipeline stage without Pydantic plumbing.

1. **Exact duplicate** — `entity_type + entity_name + value + subject`,
   dates within 7 days. Incremental merge with provenance.
2. **Semantic** — canonical term lookup (alias table), then fuzzy string
   ratio ≥ 0.92, then embedding cosine similarity ≥ 0.92 if an embedding
   function is injected.
3. **Time-series** — same canonical entity, different date > 7 days.
   Returned for append-to-timeline handling.
4. **Conflict detection** — in priority order:
    - Drug interaction (critical)
    - Temporal impossibility (medium)
    - Mutually exclusive type (high)
    - Same-day value mismatch (high / critical for meds)
    - Implausible physiological change (high)
5. **Implausibility thresholds**:
    - Weight: >5 kg in <7 d, >20 kg in <30 d
    - BP: >30 mmHg systolic same-day
    - Temperature: >2 °C same-day
    - HbA1c: >2 percentage points in <60 d

---

## Conflict Resolution Process

```
 Dedup Engine (conflict)
        │
        ▼
 ConflictResolver.quarantine_conflict(...)
     ├─ INSERT conflicts row (pending)
     ├─ mark fact1, fact2 tier='quarantined'
     └─ log "quarantined" to conflict_audit_log
        │
        ▼
 Streamlit UI: tab "Conflict Review"
        │
        ▼
 User picks {fact1 | fact2 | both | merge | neither}
        │
        ▼
 ConflictResolver.resolve_conflict(id, resolution)
     ├─ flip record tiers based on choice
     ├─ if merge → write new MKBRecord (trust=2)
     ├─ UPDATE conflicts.resolution + resolved_at
     └─ log "resolved:<choice>" to audit log
```

The engine refuses to apply an invalid `choice` or to resolve a conflict
twice, so the audit trail is always consistent.
