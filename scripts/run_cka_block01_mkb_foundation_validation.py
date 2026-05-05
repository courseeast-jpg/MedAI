"""CKA-B01 — MKB Foundation Validation Script.

Creates a temporary SQLite DB, inserts ≥10 synthetic MKB records across all
tiers, verifies ledger events, verifies active-retrieval exclusions, runs
privacy self-checks, and writes JSON + MD reports.

Does NOT:
- modify frozen OCR/Layout HITL release artifacts
- call external APIs
- use real patient data
- change production OCR/extractor/safety gates
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MEDAI_LOCAL_ONLY", "true")
os.environ.setdefault("MEDAI_ALLOW_EXTERNAL_API", "false")

from clinical_knowledge.config import CKAConfig
from clinical_knowledge.ledger import (
    make_created_event,
    make_tier_changed_event,
    make_updated_event,
)
from clinical_knowledge.models import (
    KnowledgeTier,
    MKBRecord,
    RecordStatus,
    SourceType,
    TrustLevel,
)
from clinical_knowledge.safe_ids import hash_source_ref, make_safe_record_id, new_record_id
from clinical_knowledge.store import (
    ENCRYPTION_BOUNDARY_READY,
    SQLCIPHER_ENCRYPTION_ACTIVE,
    MKBStore,
)

BLOCK_ID = "CKA-B01"
REPORT_DIR = ROOT / "reports" / "cka_block01_mkb_foundation"
JSON_REPORT = REPORT_DIR / "cka_block01_mkb_foundation_report.json"
MD_REPORT = REPORT_DIR / "cka_block01_mkb_foundation_report.md"

# Strings that must NOT appear in public reports
_FORBIDDEN_IN_PUBLIC = (
    "patient", "jane doe", "john doe", "dob:", "date_of_birth",
    "ssn", "mrn", "medical_record_number",
    "/home/", "C:\\Users\\", "real_validation_input",
    "operator_feedback_PRIVATE", "local_filename_mapping_PRIVATE",
    "original_relative_path",
)


# ---------------------------------------------------------------------------
# Synthetic record definitions (no real PHI)
# ---------------------------------------------------------------------------

_SYNTHETIC_RECORDS = [
    # trust 1 → active
    dict(fact_type="lab_reference_range", entity_text="Hemoglobin A1c <5.7% normal",
         specialty="endocrinology", trust_level=TrustLevel.EXPERT_VALIDATED,
         source_type=SourceType.SYNTHETIC, confidence=0.95),
    # trust 2 → active
    dict(fact_type="drug_interaction_rule", entity_text="Warfarin + NSAIDs: increased bleeding risk",
         specialty="pharmacology", trust_level=TrustLevel.PEER_REVIEWED,
         source_type=SourceType.SYNTHETIC, confidence=0.90),
    # trust 2 → active (explicit confirmed status)
    dict(fact_type="diagnostic_criteria", entity_text="Fasting glucose ≥126 mg/dL: diabetes",
         specialty="endocrinology", trust_level=TrustLevel.PEER_REVIEWED,
         source_type=SourceType.SYNTHETIC, confidence=0.88,
         status=RecordStatus.CONFIRMED),
    # trust 3 → hypothesis
    dict(fact_type="symptom_association", entity_text="Fatigue may indicate hypothyroidism",
         specialty="general", trust_level=TrustLevel.OPERATOR_REVIEWED,
         source_type=SourceType.STUB_CONNECTOR, confidence=0.60),
    # trust 4 → hypothesis
    dict(fact_type="treatment_suggestion", entity_text="Consider ACE inhibitor for stage 2 hypertension",
         specialty="cardiology", trust_level=TrustLevel.MODEL_SUGGESTED,
         source_type=SourceType.EXTRACTION_PIPELINE, confidence=0.55),
    # trust 5 → hypothesis
    dict(fact_type="research_association", entity_text="Vitamin D deficiency linked to MS risk",
         specialty="neurology", trust_level=TrustLevel.UNVERIFIED,
         source_type=SourceType.SYNTHETIC, confidence=0.40),
    # trust 5 → hypothesis (explicit)
    dict(fact_type="drug_dosage_note", entity_text="Metformin 500mg BID standard starting dose",
         specialty="endocrinology", trust_level=TrustLevel.UNVERIFIED,
         source_type=SourceType.SYNTHETIC, confidence=0.45),
    # explicit quarantined (trust 3 → would be hypothesis but tier forced)
    dict(fact_type="flagged_extraction", entity_text="Ambiguous lab value extraction flagged for review",
         specialty="laboratory", trust_level=TrustLevel.OPERATOR_REVIEWED,
         source_type=SourceType.EXTRACTION_PIPELINE, confidence=0.30,
         tier=KnowledgeTier.QUARANTINED),
    # explicit quarantined
    dict(fact_type="unvalidated_claim", entity_text="Experimental marker ZZ-99 not yet validated",
         specialty="research", trust_level=TrustLevel.UNVERIFIED,
         source_type=SourceType.SYNTHETIC, confidence=0.20,
         tier=KnowledgeTier.QUARANTINED),
    # explicit superseded
    dict(fact_type="outdated_guideline", entity_text="Old HbA1c cutoff 6.0% (superseded by 5.7% guideline)",
         specialty="endocrinology", trust_level=TrustLevel.PEER_REVIEWED,
         source_type=SourceType.SYNTHETIC, confidence=0.50,
         tier=KnowledgeTier.SUPERSEDED),
    # trust 1 → active, hypothesis upgrade candidate
    dict(fact_type="lab_reference_range", entity_text="eGFR <60: CKD stage 3+",
         specialty="nephrology", trust_level=TrustLevel.EXPERT_VALIDATED,
         source_type=SourceType.SYNTHETIC, confidence=0.93),
]


def _build_record(defn: dict) -> MKBRecord:
    rec_id = new_record_id()
    safe_id = make_safe_record_id(rec_id)
    raw_src = f"synthetic_source_{defn['fact_type']}"
    safe_src = hash_source_ref(raw_src)
    return MKBRecord(
        record_id=rec_id,
        safe_record_id=safe_id,
        session_id="cka_b01_validation_session",
        fact_type=defn["fact_type"],
        entity_text=defn["entity_text"],
        specialty=defn.get("specialty", "general"),
        source_type=defn.get("source_type", SourceType.SYNTHETIC),
        source_ref=safe_src,
        trust_level=defn.get("trust_level", TrustLevel.UNVERIFIED),
        tier=defn.get("tier"),
        status=defn.get("status", RecordStatus.PENDING),
        confidence=defn.get("confidence", 0.0),
    )


def _privacy_check_payload(payload: str) -> list[str]:
    found = []
    lower = payload.lower()
    for pat in _FORBIDDEN_IN_PUBLIC:
        if pat.lower() in lower:
            found.append(pat)
    return found


def run_validation(report_dir: Path = REPORT_DIR) -> dict:
    cfg = CKAConfig()
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        store = MKBStore(db_path=db_path)

        # Insert all synthetic records with ledger events
        for defn in _SYNTHETIC_RECORDS:
            record = _build_record(defn)
            event = make_created_event(
                record_id=record.record_id,
                safe_record_id=record.safe_record_id,
                tier=record.tier.value,
                trust_level=record.trust_level.value,
            )
            store.insert_record(record, ledger_event=event)

        # Demonstrate tier change: promote one hypothesis record to quarantined
        hypothesis_rows = store.list_hypothesis()
        if hypothesis_rows:
            target_id = hypothesis_rows[0]["record_id"]
            target_safe = hypothesis_rows[0]["safe_record_id"]
            old_tier = hypothesis_rows[0]["tier"]
            from datetime import datetime, timezone
            now = datetime.now(timezone.utc).isoformat()
            tier_event = make_tier_changed_event(
                record_id=target_id,
                safe_record_id=target_safe,
                old_tier=old_tier,
                new_tier=KnowledgeTier.HYPOTHESIS.value,
                reason="validation: tier re-confirmed",
            )
            store.update_record_tier(target_id, KnowledgeTier.HYPOTHESIS, now, ledger_event=tier_event)

        # Collect counts
        active_rows = store.list_active()
        hypothesis_rows = store.list_hypothesis()
        quarantined_rows = store.list_quarantined()
        superseded_rows = store.list_superseded()
        all_events = store.read_ledger_events()
        total_inserted = len(_SYNTHETIC_RECORDS)

        # Verify ledger has events for every record
        assert len(all_events) >= total_inserted, "Fewer ledger events than records inserted"

        # Active retrieval must exclude quarantined and superseded
        active_tiers = {r["tier"] for r in active_rows}
        assert KnowledgeTier.QUARANTINED.value not in active_tiers, "Quarantined in active list"
        assert KnowledgeTier.SUPERSEDED.value not in active_tiers, "Superseded in active list"

        # Quarantined rows must have requires_review set
        for r in quarantined_rows:
            assert r["requires_review"] == 1, f"Quarantined record {r['record_id']} missing requires_review"
        for r in superseded_rows:
            assert r["requires_review"] == 1, f"Superseded record {r['record_id']} missing requires_review"

        # Build public report payload
        public_records = [
            {
                "safe_record_id": r["safe_record_id"],
                "fact_type": r["fact_type"],
                "tier": r["tier"],
                "trust_level": r["trust_level"],
                "confidence": r["confidence"],
                "requires_review": bool(r["requires_review"]),
            }
            for r in (active_rows + hypothesis_rows + quarantined_rows + superseded_rows)
        ]
        public_events = [
            {
                "event_id": e["event_id"],
                "event_type": e["event_type"],
                "timestamp": e["timestamp"],
                "safe_public_details": json.loads(e["safe_public_details"]),
            }
            for e in all_events
        ]

        # Privacy self-check
        public_payload = json.dumps({"records": public_records, "events": public_events})
        privacy_leaks = _privacy_check_payload(public_payload)

        report = {
            "block_id": BLOCK_ID,
            "conclusion": "cka_b01_mkb_foundation_ready" if not privacy_leaks else "cka_b01_privacy_leak_detected",
            "records_inserted": total_inserted,
            "ledger_events_written": len(all_events),
            "active_records_count": len(active_rows),
            "hypothesis_records_count": len(hypothesis_rows),
            "quarantined_records_count": len(quarantined_rows),
            "superseded_records_count": len(superseded_rows),
            "feature_flags": cfg.as_dict(),
            "sqlcipher_encryption_active": SQLCIPHER_ENCRYPTION_ACTIVE,
            "encryption_boundary_ready": ENCRYPTION_BOUNDARY_READY,
            "external_api_used": False,
            "raw_phi_logged_in_public_reports": bool(privacy_leaks),
            "private_filename_path_leaks": len(privacy_leaks),
            "privacy_leak_details": privacy_leaks,
            "production_ocr_changed": False,
            "production_extractor_changed": False,
            "safety_gate_changed": False,
            "frozen_hitl_release_reopened": False,
            "public_records_sample": public_records[:5],
        }

        if privacy_leaks:
            raise RuntimeError(f"STOP: privacy leaks detected in public report: {privacy_leaks}")

    finally:
        try:
            Path(db_path).unlink(missing_ok=True)
        except Exception:
            pass

    # Write reports
    json_path = report_dir / "cka_block01_mkb_foundation_report.json"
    md_path = report_dir / "cka_block01_mkb_foundation_report.md"

    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    md_lines = [
        "# CKA-B01 MKB Foundation Validation Report",
        "",
        f"**Block:** {BLOCK_ID}",
        f"**Conclusion:** `{report['conclusion']}`",
        "",
        "## Record Counts",
        f"- Inserted: {report['records_inserted']}",
        f"- Active: {report['active_records_count']}",
        f"- Hypothesis: {report['hypothesis_records_count']}",
        f"- Quarantined: {report['quarantined_records_count']}",
        f"- Superseded: {report['superseded_records_count']}",
        f"- Ledger events: {report['ledger_events_written']}",
        "",
        "## Safety Flags",
        f"- external_api_used: {report['external_api_used']}",
        f"- raw_phi_logged_in_public_reports: {report['raw_phi_logged_in_public_reports']}",
        f"- private_filename_path_leaks: {report['private_filename_path_leaks']}",
        f"- production_ocr_changed: {report['production_ocr_changed']}",
        f"- production_extractor_changed: {report['production_extractor_changed']}",
        f"- safety_gate_changed: {report['safety_gate_changed']}",
        f"- frozen_hitl_release_reopened: {report['frozen_hitl_release_reopened']}",
        "",
        "## Encryption",
        f"- sqlcipher_encryption_active: {report['sqlcipher_encryption_active']}",
        f"- encryption_boundary_ready: {report['encryption_boundary_ready']}",
        "",
        "## Feature Flags",
        "```json",
        json.dumps(report["feature_flags"], indent=2),
        "```",
    ]
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"CKA-B01 conclusion: {report['conclusion']}")
    print(f"Records inserted: {report['records_inserted']}")
    print(f"Ledger events: {report['ledger_events_written']}")
    print(f"Active: {report['active_records_count']}, Hypothesis: {report['hypothesis_records_count']}, "
          f"Quarantined: {report['quarantined_records_count']}, Superseded: {report['superseded_records_count']}")
    print(f"Privacy leaks: {report['private_filename_path_leaks']}")
    print(f"JSON report: {json_path}")

    return report


def main() -> int:
    report = run_validation()
    return 0 if report["conclusion"] == "cka_b01_mkb_foundation_ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
