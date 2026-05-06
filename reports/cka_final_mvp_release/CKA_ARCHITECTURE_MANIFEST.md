# CKA Architecture Manifest

Final manifest of the Clinical Knowledge Architecture MVP scaffold,
delivered through CKA-B01 through CKA-B10 and packaged in CKA-B11.

Branch: `clinical-knowledge-architecture`

---

## 1. Block commit list

| Block | Commit | Title |
|---|---|---|
| CKA-B01 | `04477ca` | Clinical knowledge MKB foundation and ledger |
| CKA-B02 | `f42be80` | Privacy boundary and outbound PHI audit |
| CKA-B03 | `da45b71` | Decision engine safe mode and response scoring |
| CKA-B04 | `7011079` | Truth resolution and quarantine engine |
| CKA-B05 | `398568e` | Medication safety DDI dual-layer gate |
| CKA-B06 | `02b7955` | Controlled enrichment hypothesis tier |
| CKA-B07 | `0ad2815` | Medical coding interface and safe terminology mapping |
| CKA-B08 | `65aa131` | Multi-connector execution and consensus engine |
| CKA-B09 | `ff0adf2` | Operator UI clinical knowledge safety panels |
| CKA-B10 | `27d940e` | System preflight and scaffold |
| CKA-B11 | (this commit) | Final clinical knowledge MVP release package |

---

## 2. Modules created

### `clinical_knowledge/` package

```
clinical_knowledge/
  __init__.py
  config.py
  models.py
  store.py
  ledger.py
  safe_ids.py
  preflight.py
  scaffold.py
  privacy/
    __init__.py
    sanitizer.py
    outbound_audit.py
    report_privacy.py
    private_mapping.py
    patterns.py
  decision_engine/
    __init__.py
    models.py
    classifier.py
    connectors.py
    scoring.py
    safe_mode.py
    refusal.py
    engine.py
    context_retrieval.py
  truth_resolution/
    __init__.py
    models.py
    integration.py
    quarantine.py
    engine.py
    conflict_detection.py
    rules.py
  medication_safety/
    __init__.py
    models.py
    ddi_stub.py
    evidence_modifier.py
    write_gate.py
    integration.py
  enrichment/
    __init__.py
    models.py
    candidate_extractor.py
    enrichment_queue.py
    hypothesis_writer.py
    integration.py
    promotion.py
  medical_coding/
    __init__.py
    models.py
    terminology_source.py
    synthetic_mapper.py
    local_lookup.py
    validator.py
    integration.py
  connectors/
    __init__.py
    models.py
    registry.py
    request_builder.py
    stubs.py
    normalizer.py
    executor.py
  consensus/
    __init__.py
    models.py
    fact_extractor.py
    agreement.py
    contradiction.py
    engine.py
    integration.py
```

### Operator UI

```
app/clinical_knowledge_safety_viewer.py
```

Streamlit integration is wired in `app/main.py` under the
"Clinical Knowledge Safety" tab via an isolated try/except.

### Validation scripts

```
scripts/run_cka_block01_mkb_foundation_validation.py
scripts/run_cka_block02_privacy_boundary_validation.py
scripts/run_cka_block03_decision_engine_validation.py
scripts/run_cka_block04_truth_resolution_validation.py
scripts/run_cka_block05_medication_safety_validation.py
scripts/run_cka_block06_controlled_enrichment_validation.py
scripts/run_cka_block07_medical_coding_validation.py
scripts/run_cka_block08_multi_connector_consensus_validation.py
scripts/run_cka_block09_operator_ui_validation.py
scripts/run_cka_block10_preflight_scaffold_validation.py
scripts/run_cka_final_mvp_release_validation.py        # CKA-B11
```

### Tests

```
tests/test_cka_block01_mkb_foundation.py
tests/test_cka_block02_privacy_boundary.py
tests/test_cka_block03_decision_engine.py
tests/test_cka_block04_truth_resolution.py
tests/test_cka_block05_medication_safety.py
tests/test_cka_block06_controlled_enrichment.py
tests/test_cka_block07_medical_coding.py
tests/test_cka_block08_multi_connector_consensus.py
tests/test_cka_block09_operator_ui.py
tests/test_cka_block10_preflight_scaffold.py
tests/test_cka_final_mvp_release.py                    # CKA-B11
```

---

## 3. Validation status

- B01-B10 validation scripts: all pass on local synthetic fixtures.
- Full CKA test suite (B01-B10): 680 tests, all passing as of HEAD
  `27d940e`. CKA-B11 adds the release package test file.
- Preflight: 26 / 26 checks pass; HITL freeze confirmed; external API
  blocked.

---

## 4. Safety boundaries (reaffirmed)

- `_PRODUCTION_AUTONOMOUS = False`
- `allow_active_write=True` raises `ValueError` in scaffold and in
  `consensus_facts_to_enrichment_candidates`.
- `EXTERNAL_APIS_ENABLED=True` raises `ValueError` in scaffold.
- All connectors `allow_external=False`, `synthetic_only=True`.
- All AI-derived facts written as `hypothesis` tier only.
- `ENRICH_PROMOTE` defaults to `False`.
- DDI status on quarantined records is never cleared by the scaffold.
- Consensus does not synthesize a new value across contradictions.
- Coding does not invent codes for unmapped entities.
- No real UMLS / SNOMED / DxGPT / SAGE / PatientNotes / LLM API used.
- Frozen HITL release artifacts are unchanged.

---

## 5. Next possible roadmap tracks

The following tracks are **possible** continuations after CKA-B11.
None are activated in the MVP scaffold. Each would require its own
guarded feature-flag activation, validation, and operator review.

1. **Real connector activation under guarded feature flags**
   — replace synthetic stubs with real DxGPT / SAGE / PatientNotes
   integrations behind explicit per-connector enable flags, with
   privacy-gated outbound payloads and operator approval per call.
2. **Real terminology data integration**
   — license-checked UMLS / SNOMED / RxNorm subset import, validated
   coding tables, periodic refresh, version tracking.
3. **SQLCipher encryption activation**
   — replace stdlib SQLite store with SQLCipher; rotate keys via
   operator-managed key store; audit-log every key event.
4. **Russian / multilingual support**
   — extend extractor + sanitizer + coder for Cyrillic and other
   scripts, language-tagged records, locale-aware DDI checks.
5. **Final product UI polish**
   — refine operator-facing UI, add review queues for hypothesis →
   active promotion paths, add quarantine resolution workflow.
6. **Local LLM activation**
   — enable local-only LLM enrichment behind `ENABLE_LOCAL_LLM`,
   with sandboxed prompts, output sanitation, and hypothesis-only
   write boundary preserved.
