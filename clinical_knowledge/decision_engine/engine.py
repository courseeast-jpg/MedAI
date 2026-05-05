"""CKA-B03 Decision Engine orchestrator.

Pipeline:
  classify → retrieve_context → privacy_check → call_connectors →
  score → safe_mode → refusal → build_result → ledger
"""
from __future__ import annotations

from typing import List, Optional, TYPE_CHECKING

from clinical_knowledge.config import CKAConfig
from clinical_knowledge.decision_engine.classifier import classify_query
from clinical_knowledge.decision_engine.connectors import call_connector, CONNECTOR_IDS
from clinical_knowledge.decision_engine.context_retrieval import retrieve_context
from clinical_knowledge.decision_engine.models import (
    ConnectorRequest,
    ConnectorResponse,
    DecisionEngineResult,
    SafeModeState,
    ScoredResponse,
)
from clinical_knowledge.decision_engine.refusal import evaluate_refusal, diagnosis_disclaimer
from clinical_knowledge.decision_engine.safe_mode import evaluate_safe_mode, apply_safe_mode_prefix
from clinical_knowledge.decision_engine.scoring import score_all_responses
from clinical_knowledge.ledger import (
    make_privacy_audit_event,
    make_safe_mode_entry_event,
    make_response_discarded_event,
    make_validation_event,
)
from clinical_knowledge.privacy.sanitizer import sanitize_text
from clinical_knowledge.safe_ids import new_event_id

if TYPE_CHECKING:
    from clinical_knowledge.store import MKBStore

_ENGINE_SESSION_ID = "decision_engine_session"


def run_decision_engine(
    query: str,
    store: "MKBStore",
    config: Optional[CKAConfig] = None,
    *,
    manual_safe_mode: bool = False,
    simulated_connector_mode: Optional[str] = None,
) -> DecisionEngineResult:
    """Run the full decision engine pipeline for a query.

    Args:
        query: Raw user query string.
        store: Active MKBStore instance.
        config: CKAConfig; defaults to CKAConfig() if None.
        manual_safe_mode: Force safe mode regardless of scores.
        simulated_connector_mode: "all_fail" to simulate all connectors failing.

    Returns:
        DecisionEngineResult with full audit trail.
    """
    if config is None:
        config = CKAConfig()

    ledger_events_written = 0
    engine_record_id = f"engine_{new_event_id()}"
    engine_safe_id = f"safe_engine_{new_event_id()[:8]}"

    # Step 1: Sanitize query before any processing
    san = sanitize_text(query)
    raw_phi_in_query = san.raw_phi_detected

    # Step 2: Classify (uses sanitized hash — never raw query in logs)
    classification = classify_query(query)

    # Step 3: Refusal check
    refused, refusal_message = evaluate_refusal(classification)
    if refused:
        return DecisionEngineResult(
            query_hash=classification.raw_query_hash,
            classification=classification,
            context=retrieve_context(classification, store),
            connector_responses=[],
            scored_responses=[],
            safe_mode=SafeModeState(active=False, reason="refusal before safe mode evaluation"),
            final_response=refusal_message or "",
            refused=True,
            refusal_reason=refusal_message,
            external_api_used=False,
            ddi_layer1_checked=False,
            raw_phi_in_query=raw_phi_in_query,
            phi_sanitized_before_connectors=san.raw_phi_detected,
            ledger_events_written=0,
        )

    # Step 4: Retrieve MKB context
    context = retrieve_context(classification, store)

    # Step 5: Privacy check before connectors
    privacy_cleared = not san.secret_detected

    # Step 6: Call connectors
    active_connectors = config.ACTIVE_CONNECTORS
    connector_responses: List[ConnectorResponse] = []
    ddi_layer1_checked = False

    for cid in active_connectors:
        if cid not in CONNECTOR_IDS:
            continue

        if simulated_connector_mode == "all_fail":
            connector_responses.append(ConnectorResponse(
                connector_id=cid,
                success=False,
                content="",
                confidence=0.0,
                citations=[],
                error="simulated_connector_mode=all_fail",
            ))
            continue

        req = ConnectorRequest(
            connector_id=cid,
            query_hash=classification.raw_query_hash,
            specialty=classification.specialty.value,
            task_type=classification.task_type.value,
            privacy_cleared=privacy_cleared,
        )
        resp = call_connector(req)
        connector_responses.append(resp)

        if cid == "patientnotes_ddi_stub" and resp.success:
            ddi_layer1_checked = True

    # Step 7: Log privacy audit
    privacy_event = make_privacy_audit_event(
        record_id=engine_record_id,
        safe_record_id=engine_safe_id,
        findings_summary={"phi_in_query": raw_phi_in_query, "secret": san.secret_detected},
        passed=not san.secret_detected,
    )
    store.append_ledger_event(privacy_event)
    ledger_events_written += 1

    # Step 8: DDI Layer 1 modifier
    ddi_modifier = 1.0  # Layer 1 placeholder only

    # Step 9: Score responses
    scored_responses = score_all_responses(
        connector_responses, context, classification, ddi_modifier
    )

    # Log discarded responses
    for sr in scored_responses:
        if sr.discarded:
            evt = make_response_discarded_event(
                record_id=engine_record_id,
                safe_record_id=engine_safe_id,
                score=sr.composite_score,
                reason=f"Score {sr.composite_score} below discard threshold",
            )
            store.append_ledger_event(evt)
            ledger_events_written += 1

    # Step 10: Compute aggregate confidence
    non_discarded = [sr for sr in scored_responses if not sr.discarded]
    if non_discarded:
        aggregate_confidence = sum(sr.composite_score for sr in non_discarded) / len(non_discarded)
    else:
        aggregate_confidence = 0.0

    # Step 11: Evaluate safe mode
    safe_mode = evaluate_safe_mode(
        connector_responses=connector_responses,
        scored_responses=scored_responses,
        aggregate_confidence=aggregate_confidence,
        threshold=config.SAFE_MODE_THRESHOLD,
        manual_safe_mode=manual_safe_mode,
    )

    if safe_mode.active:
        evt = make_safe_mode_entry_event(
            record_id=engine_record_id,
            safe_record_id=engine_safe_id,
            reason=safe_mode.reason,
        )
        store.append_ledger_event(evt)
        ledger_events_written += 1

    # Step 12: Build final response
    final_response = _build_final_response(
        classification=classification,
        context=context,
        scored_responses=non_discarded,
        safe_mode=safe_mode,
        connector_responses=connector_responses,
    )

    return DecisionEngineResult(
        query_hash=classification.raw_query_hash,
        classification=classification,
        context=context,
        connector_responses=connector_responses,
        scored_responses=scored_responses,
        safe_mode=safe_mode,
        final_response=final_response,
        refused=False,
        refusal_reason=None,
        external_api_used=False,
        ddi_layer1_checked=ddi_layer1_checked,
        raw_phi_in_query=raw_phi_in_query,
        phi_sanitized_before_connectors=raw_phi_in_query,
        ledger_events_written=ledger_events_written,
    )


def _build_final_response(
    classification,
    context,
    scored_responses: List[ScoredResponse],
    safe_mode: SafeModeState,
    connector_responses: List[ConnectorResponse],
) -> str:
    disclaimer = diagnosis_disclaimer()

    if safe_mode.active:
        if context.mkb_records_found > 0:
            snippets = "; ".join(context.mkb_snippets[:3])
            body = (
                f"Based on {context.mkb_records_found} MKB record(s) "
                f"({context.context_tiers[0] if context.context_tiers else 'unknown'} tier): "
                f"{snippets}"
            )
        else:
            body = "No relevant MKB records found for this query."
        response = f"{body} {disclaimer}"
        return apply_safe_mode_prefix(response, safe_mode)

    if not scored_responses:
        body = "No scored responses available."
        if context.mkb_records_found > 0:
            body += (
                f" MKB context ({context.mkb_records_found} records) available "
                f"but no connector responses passed scoring threshold."
            )
        return f"{body} {disclaimer}"

    # Use highest-scored response
    best = max(scored_responses, key=lambda sr: sr.composite_score)
    response = f"{best.raw_content} {disclaimer}"
    return response
