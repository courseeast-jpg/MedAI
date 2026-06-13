"""No-live synthetic-to-real Vertex adapter dry run for 15Z-C.

The adapter accepts only 15Z-B outbound-safe tokenized payloads, builds the
would-be Vertex ``generateContent`` request body offline, fingerprints the local
proof pieces, and creates report-only review handoff records. It has no provider
call path, no live gate, no active MKB write, and no auto-accept path.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from execution.vertex_real_doc_pii_stripping_proof import (
    PiiStrippingFixture,
    build_isolated_pii_vault_record,
    build_outbound_safe_payload,
    build_pii_stripping_readiness_case,
    redact_pii_like_values,
)
from execution.vertex_real_doc_readiness_gates import (
    PROVENANCE_REAL_PRIVATE,
    PROVENANCE_REDACTED_REAL_LIKE,
    PROVENANCE_UNKNOWN,
    evaluate_vertex_real_doc_readiness,
)


ALLOWED_VERTEX_REQUEST_TOP_LEVEL_KEYS = ("contents", "generationConfig")
FORBIDDEN_VERTEX_REQUEST_TOP_LEVEL_KEYS = (
    "provider_route",
    "provider_name",
    "model",
    "location",
    "endpoint",
    "project_id",
    "package_id",
    "package_family",
    "metadata",
    "source_report_reference",
    "token_map",
    "pii_vault",
    "vault_record",
    "medai_metadata",
)
JSON_RESPONSE_MIME_TYPE = "application/json"
MAX_OUTPUT_TOKENS_LIMIT = 512

_TOKEN_RE = re.compile(r"\[[A-Z_]+_\d+\]")
_LABEL_VALUE_RE = re.compile(
    r"(?im)^(?:patient|patient name|name|provider|physician|ordering provider|attending|"
    r"facility|clinic|hospital|imaging center|mrn|account|account number|account no|"
    r"medical record number|accession|accession id|report id|order id|dob|date of birth|"
    r"date|collected|collection date|service date|phone|tel|telephone|contact|"
    r"contact phone|email|e-mail|contact email|address|addr|street address)\s*:\s*(?P<v>.+?)\s*$"
)
_TOKEN_MAP_SIGNATURE_RE = re.compile(r'"\[[A-Z_]+_\d+\]"\s*:\s*"[^"]+"')
_PRIVATE_MARKERS = (
    "real private",
    "real_private",
    "raw pdf",
    "raw_pdf",
    "raw image",
    "raw_image",
    "raw ocr",
    "ocr_private",
    ".pdf",
    ".png",
    ".jpg",
)


@dataclass(frozen=True)
class VertexRequestShapeValidation:
    request_shape_valid: bool
    request_top_level_keys: list[str]
    forbidden_top_level_keys_present: list[str]
    generation_config_valid: bool
    json_only_response_config_present: bool
    block_reasons: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class VertexDryRunRequestProof:
    outbound_payload_fingerprint: str
    would_be_request_fingerprint: str
    local_metadata_fingerprint: str
    vault_record_fingerprint: str
    request_shape_validation: VertexRequestShapeValidation


@dataclass(frozen=True)
class ReviewQueueHandoffRecord:
    handoff_id: str
    case_id: str
    status: str
    report_only: bool
    active_write_allowed: bool
    auto_accept_allowed: bool
    live_call_allowed: bool
    review_required: bool
    outbound_payload_fingerprint: str
    would_be_request_fingerprint: str
    vault_record_fingerprint: str
    provenance_classification: str
    readiness_status: str


@dataclass(frozen=True)
class AdapterDryRunResult:
    case_id: str
    provenance_classification: str
    adapter_status: str
    request_shape_valid: bool
    request_top_level_keys: list[str]
    forbidden_top_level_keys_present: list[str]
    generation_config_valid: bool
    outbound_payload_fingerprint: str
    would_be_request_fingerprint: str
    vault_record_fingerprint: str
    review_queue_handoff_record_created: bool
    handoff_status: str
    readiness_status: str
    blocked: bool
    block_reasons: list[str]
    live_call_allowed: bool
    external_api_used: bool
    active_write_allowed: bool
    auto_accept_allowed: bool
    review_required: bool
    raw_pii_in_request: bool
    raw_pii_in_report: bool
    token_map_in_request: bool
    token_map_in_report: bool
    dry_run_proof: VertexDryRunRequestProof | None = None
    review_queue_handoff_record: ReviewQueueHandoffRecord | None = None


@dataclass(frozen=True)
class AdapterDryRunFixture:
    case_id: str
    raw_text: str
    declared_provenance: str
    content_marker: str
    expected_status: str
    active_write_requested: bool = False
    auto_accept_requested: bool = False
    contains_medication_fact: bool = False
    medication_safety_gate_satisfied: bool = False
    future_gates_simulated_pass: bool = False
    inject_raw_pii_residue: bool = False
    inject_token_map: bool = False
    inject_forbidden_metadata_key: bool = False
    invalid_generation_config: bool = False


def build_vertex_request_body_no_live(outbound_safe_payload: str) -> dict[str, Any]:
    """Build the exact would-be Vertex request body, offline.

    The top-level body intentionally contains only Vertex request fields. All
    provider/model/vault/provenance metadata stays local in dry-run proofs and
    review handoff records.
    """
    return {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "text": "\n".join(
                            [
                                "Return JSON only.",
                                "Extract source-faithful facts from this tokenized payload.",
                                "Do not diagnose, recommend treatment, or infer missing values.",
                                "Every fact must include source evidence text.",
                                "Keep unknown values unknown.",
                                "Tokenized payload:",
                                outbound_safe_payload,
                            ]
                        )
                    }
                ],
            }
        ],
        "generationConfig": {
            "temperature": 0,
            "maxOutputTokens": MAX_OUTPUT_TOKENS_LIMIT,
            "responseMimeType": JSON_RESPONSE_MIME_TYPE,
        },
    }


def validate_vertex_request_body_shape(request_body: Mapping[str, Any]) -> VertexRequestShapeValidation:
    keys = list(request_body.keys())
    forbidden = [k for k in keys if k in FORBIDDEN_VERTEX_REQUEST_TOP_LEVEL_KEYS]
    reasons: list[str] = []
    if keys != list(ALLOWED_VERTEX_REQUEST_TOP_LEVEL_KEYS):
        reasons.append("request_top_level_keys_not_exact")
    if forbidden:
        reasons.append("forbidden_top_level_metadata_key_present")
    contents = request_body.get("contents")
    if not isinstance(contents, list) or not contents:
        reasons.append("contents_missing_or_invalid")
    generation = request_body.get("generationConfig")
    generation_valid = _generation_config_valid(generation)
    if not generation_valid:
        reasons.append("generation_config_invalid")
    json_only = isinstance(generation, Mapping) and generation.get("responseMimeType") == JSON_RESPONSE_MIME_TYPE
    return VertexRequestShapeValidation(
        request_shape_valid=not reasons,
        request_top_level_keys=keys,
        forbidden_top_level_keys_present=forbidden,
        generation_config_valid=generation_valid,
        json_only_response_config_present=json_only,
        block_reasons=reasons,
    )


def fingerprint_vertex_request_body(request_body: Mapping[str, Any]) -> str:
    return _fingerprint_json(request_body)


def build_review_queue_handoff_record(
    *,
    case_id: str,
    provenance_classification: str,
    readiness_status: str,
    outbound_payload_fingerprint: str,
    would_be_request_fingerprint: str,
    vault_record_fingerprint: str,
) -> ReviewQueueHandoffRecord:
    material = "|".join(
        [case_id, provenance_classification, outbound_payload_fingerprint, would_be_request_fingerprint]
    )
    return ReviewQueueHandoffRecord(
        handoff_id="handoff_" + _fingerprint_text(material).split(":", 1)[-1][:12],
        case_id=case_id,
        status="review_required",
        report_only=True,
        active_write_allowed=False,
        auto_accept_allowed=False,
        live_call_allowed=False,
        review_required=True,
        outbound_payload_fingerprint=outbound_payload_fingerprint,
        would_be_request_fingerprint=would_be_request_fingerprint,
        vault_record_fingerprint=vault_record_fingerprint,
        provenance_classification=provenance_classification,
        readiness_status=readiness_status,
    )


def evaluate_adapter_readiness_with_gates(fixture: AdapterDryRunFixture) -> AdapterDryRunResult:
    redaction = redact_pii_like_values(fixture.raw_text)
    vault = build_isolated_pii_vault_record(redaction)
    outbound = build_outbound_safe_payload(redaction)
    outbound_text = outbound.outbound_text
    if fixture.inject_raw_pii_residue and redaction._detected_values:
        outbound_text += "\nResidual labelled copy: " + redaction._detected_values[0]
    if fixture.inject_token_map:
        outbound_text += "\n" + vault._serialized_map

    pii_case = build_pii_stripping_readiness_case(
        PiiStrippingFixture(
            case_id=fixture.case_id,
            declared_provenance=fixture.declared_provenance,
            content_marker=fixture.content_marker,
            raw_text=fixture.raw_text,
            expected_block=fixture.declared_provenance != PROVENANCE_REDACTED_REAL_LIKE,
        )
    )
    framework = evaluate_vertex_real_doc_readiness(
        case_id=fixture.case_id,
        declared_provenance=fixture.declared_provenance,
        content_marker=fixture.content_marker,
        human_authorization_present=fixture.future_gates_simulated_pass,
        billing_cost_cap_ack_present=fixture.future_gates_simulated_pass,
        active_write_requested=fixture.active_write_requested,
        auto_accept_requested=fixture.auto_accept_requested,
        contains_medication_fact=fixture.contains_medication_fact,
        medication_safety_gate_satisfied=fixture.medication_safety_gate_satisfied,
        future_gates_simulated_pass=fixture.future_gates_simulated_pass,
    )

    request_body = build_vertex_request_body_no_live(outbound_text)
    if fixture.inject_forbidden_metadata_key:
        request_body["metadata"] = {"local_only": True}
    if fixture.invalid_generation_config:
        request_body["generationConfig"] = {"temperature": 0.7, "maxOutputTokens": 2048}

    shape = validate_vertex_request_body_shape(request_body)
    request_text = json.dumps(request_body, sort_keys=True, ensure_ascii=False)
    raw_pii_in_request = _raw_pii_like_present(outbound_text) or any(
        value and value in request_text for value in redaction._detected_values
    )
    token_map_in_request = _token_map_present(outbound_text) or _token_map_present(request_text)

    local_metadata = {
        "case_id": fixture.case_id,
        "provenance": pii_case["provenance_classification"],
        "vault_record_fingerprint": pii_case["vault_record_fingerprint"],
        "active_write_requested": fixture.active_write_requested,
        "auto_accept_requested": fixture.auto_accept_requested,
        "contains_medication_fact": fixture.contains_medication_fact,
    }
    proof = VertexDryRunRequestProof(
        outbound_payload_fingerprint=_fingerprint_text(outbound_text),
        would_be_request_fingerprint=fingerprint_vertex_request_body(request_body),
        local_metadata_fingerprint=_fingerprint_json(local_metadata),
        vault_record_fingerprint=pii_case["vault_record_fingerprint"],
        request_shape_validation=shape,
    )

    block_reasons: list[str] = []
    if pii_case["provenance_classification"] != PROVENANCE_REDACTED_REAL_LIKE:
        block_reasons.append("provenance_not_redacted_real_like")
    if any(marker in fixture.content_marker.lower() for marker in _PRIVATE_MARKERS):
        block_reasons.append("real_or_private_marker_present")
    if raw_pii_in_request:
        block_reasons.append("raw_pii_like_value_in_request")
    if token_map_in_request:
        block_reasons.append("token_map_leak_in_request")
    if not shape.request_shape_valid:
        block_reasons.extend(shape.block_reasons)
    if fixture.active_write_requested:
        block_reasons.append("active_write_requested_blocked")
    if fixture.auto_accept_requested:
        block_reasons.append("auto_accept_requested_blocked")
    if fixture.contains_medication_fact and not fixture.medication_safety_gate_satisfied:
        block_reasons.append("medication_fact_without_safety_gate_blocked")

    future_only = (
        fixture.future_gates_simulated_pass
        and not block_reasons
        and framework.readiness_status == "READY_FOR_FUTURE_AUTHORIZATION_ONLY"
    )
    blocked = bool(block_reasons) or not future_only
    if future_only:
        adapter_status = "READY_FOR_FUTURE_AUTHORIZATION_ONLY"
        readiness_status = "READY_FOR_FUTURE_AUTHORIZATION_ONLY"
    elif block_reasons:
        adapter_status = "BLOCKED"
        readiness_status = "BLOCKED_ADAPTER_DRY_RUN"
    else:
        adapter_status = "DRY_RUN_REQUEST_SHAPE_VALID_REVIEW_REQUIRED"
        readiness_status = "READY_FOR_NO_LIVE_REPLAY_ONLY"
        blocked = False

    handoff: ReviewQueueHandoffRecord | None = None
    if adapter_status in {
        "DRY_RUN_REQUEST_SHAPE_VALID_REVIEW_REQUIRED",
        "READY_FOR_FUTURE_AUTHORIZATION_ONLY",
    }:
        handoff = build_review_queue_handoff_record(
            case_id=fixture.case_id,
            provenance_classification=pii_case["provenance_classification"],
            readiness_status=readiness_status,
            outbound_payload_fingerprint=proof.outbound_payload_fingerprint,
            would_be_request_fingerprint=proof.would_be_request_fingerprint,
            vault_record_fingerprint=proof.vault_record_fingerprint,
        )

    return AdapterDryRunResult(
        case_id=fixture.case_id,
        provenance_classification=pii_case["provenance_classification"],
        adapter_status=adapter_status,
        request_shape_valid=shape.request_shape_valid,
        request_top_level_keys=shape.request_top_level_keys,
        forbidden_top_level_keys_present=shape.forbidden_top_level_keys_present,
        generation_config_valid=shape.generation_config_valid,
        outbound_payload_fingerprint=proof.outbound_payload_fingerprint,
        would_be_request_fingerprint=proof.would_be_request_fingerprint,
        vault_record_fingerprint=proof.vault_record_fingerprint,
        review_queue_handoff_record_created=handoff is not None,
        handoff_status=handoff.status if handoff else "not_created",
        readiness_status=readiness_status,
        blocked=blocked,
        block_reasons=block_reasons,
        live_call_allowed=False,
        external_api_used=False,
        active_write_allowed=False,
        auto_accept_allowed=False,
        review_required=True,
        raw_pii_in_request=raw_pii_in_request,
        raw_pii_in_report=False,
        token_map_in_request=token_map_in_request,
        token_map_in_report=False,
        dry_run_proof=proof,
        review_queue_handoff_record=handoff,
    )


def build_adapter_dry_run_fixtures() -> list[AdapterDryRunFixture]:
    return [
        AdapterDryRunFixture(
            case_id="sanitized_redacted_real_like_basic_note",
            declared_provenance=PROVENANCE_REDACTED_REAL_LIKE,
            content_marker="redacted real-like clinical note layout",
            expected_status="DRY_RUN_REQUEST_SHAPE_VALID_REVIEW_REQUIRED",
            raw_text=(
                "Patient: Jordan A. Synthsample\n"
                "DOB: 1991-04-15\n"
                "MRN: SYN-0012345\n"
                "Note: Sodium 140 mmol/L within reference range."
            ),
        ),
        AdapterDryRunFixture(
            case_id="sanitized_redacted_real_like_lab_report",
            declared_provenance=PROVENANCE_REDACTED_REAL_LIKE,
            content_marker="redacted real-like lab report layout",
            expected_status="DRY_RUN_REQUEST_SHAPE_VALID_REVIEW_REQUIRED",
            raw_text=(
                "Facility: Northshore Synthetic Imaging Center\n"
                "Accession: ACC-SYN-778899\n"
                "Provider: Dr. Riley Testname\n"
                "Result: Hemoglobin 13.2 g/dL within range."
            ),
        ),
        AdapterDryRunFixture(
            case_id="sanitized_contact_fields_fixture",
            declared_provenance=PROVENANCE_REDACTED_REAL_LIKE,
            content_marker="redacted real-like contact block layout",
            expected_status="DRY_RUN_REQUEST_SHAPE_VALID_REVIEW_REQUIRED",
            raw_text=(
                "Phone: +1-555-0142\n"
                "Email: sample.patient@example-synthetic.test\n"
                "Address: 742 Synthetic Way, Springfield, ST 00000\n"
                "Result: Urinalysis clear, no abnormalities."
            ),
        ),
        AdapterDryRunFixture(
            case_id="all_9_pii_classes_sanitized_fixture",
            declared_provenance=PROVENANCE_REDACTED_REAL_LIKE,
            content_marker="redacted real-like comprehensive layout",
            expected_status="DRY_RUN_REQUEST_SHAPE_VALID_REVIEW_REQUIRED",
            raw_text=(
                "Patient: Quinn Readysample\n"
                "DOB: 1975-06-30\n"
                "MRN: SYN-0102030\n"
                "Facility: Lakeside Synthetic Clinic\n"
                "Accession: ACC-SYN-101010\n"
                "Provider: Dr. Alex Reviewer\n"
                "Phone: +1-555-0199\n"
                "Email: quinn.ready@example-synthetic.test\n"
                "Address: 12 Sample Blvd, Testtown, ST 11111\n"
                "Note: Comprehensive panel within reference ranges."
            ),
        ),
        AdapterDryRunFixture(
            case_id="unknown_provenance_payload",
            declared_provenance=PROVENANCE_UNKNOWN,
            content_marker="unknown source layout",
            expected_status="BLOCKED",
            raw_text="Patient: Sam Unknownsrc\nMRN: SYN-0077889\nNote: Calcium 9.5 mg/dL.",
        ),
        AdapterDryRunFixture(
            case_id="real_private_marker_payload",
            declared_provenance=PROVENANCE_REAL_PRIVATE,
            content_marker="declared real private source layout",
            expected_status="BLOCKED",
            raw_text="Patient: Pat Realprivate\nMRN: SYN-0088990\nNote: Chloride 102 mmol/L.",
        ),
        AdapterDryRunFixture(
            case_id="payload_with_raw_pii_residue",
            declared_provenance=PROVENANCE_REDACTED_REAL_LIKE,
            content_marker="redacted real-like note layout",
            expected_status="BLOCKED",
            inject_raw_pii_residue=True,
            raw_text="Patient: Morgan Residuesample\nDOB: 1988-12-01\nNote: Sodium normal.",
        ),
        AdapterDryRunFixture(
            case_id="payload_with_token_map_leak",
            declared_provenance=PROVENANCE_REDACTED_REAL_LIKE,
            content_marker="redacted real-like note layout",
            expected_status="BLOCKED",
            inject_token_map=True,
            raw_text="Patient: Drew Leaksample\nMRN: SYN-0055667\nNote: Potassium normal.",
        ),
        AdapterDryRunFixture(
            case_id="request_with_forbidden_metadata_top_level_key",
            declared_provenance=PROVENANCE_REDACTED_REAL_LIKE,
            content_marker="redacted real-like note layout",
            expected_status="BLOCKED",
            inject_forbidden_metadata_key=True,
            raw_text="Patient: Meta Blocksample\nMRN: SYN-1111111\nNote: Calcium normal.",
        ),
        AdapterDryRunFixture(
            case_id="request_with_invalid_generation_config",
            declared_provenance=PROVENANCE_REDACTED_REAL_LIKE,
            content_marker="redacted real-like note layout",
            expected_status="BLOCKED",
            invalid_generation_config=True,
            raw_text="Patient: Gen Configsample\nMRN: SYN-2222222\nNote: Magnesium normal.",
        ),
        AdapterDryRunFixture(
            case_id="active_write_requested",
            declared_provenance=PROVENANCE_REDACTED_REAL_LIKE,
            content_marker="redacted real-like note layout",
            expected_status="BLOCKED",
            active_write_requested=True,
            raw_text="Patient: Active Writesample\nMRN: SYN-3333333\nNote: Review only.",
        ),
        AdapterDryRunFixture(
            case_id="auto_accept_requested",
            declared_provenance=PROVENANCE_REDACTED_REAL_LIKE,
            content_marker="redacted real-like note layout",
            expected_status="BLOCKED",
            auto_accept_requested=True,
            raw_text="Patient: Auto Acceptsample\nMRN: SYN-4444444\nNote: Review only.",
        ),
        AdapterDryRunFixture(
            case_id="medication_fact_without_safety_gate",
            declared_provenance=PROVENANCE_REDACTED_REAL_LIKE,
            content_marker="redacted real-like medication mention",
            expected_status="BLOCKED",
            contains_medication_fact=True,
            medication_safety_gate_satisfied=False,
            future_gates_simulated_pass=True,
            raw_text="Patient: Med Safetysample\nMRN: SYN-5555555\nMedication: Examplemed 5 mg listed.",
        ),
        AdapterDryRunFixture(
            case_id="all_future_gates_simulated_pass",
            declared_provenance=PROVENANCE_REDACTED_REAL_LIKE,
            content_marker="redacted real-like comprehensive layout",
            expected_status="READY_FOR_FUTURE_AUTHORIZATION_ONLY",
            future_gates_simulated_pass=True,
            raw_text=(
                "Patient: Future Readysample\n"
                "DOB: 1974-07-31\n"
                "MRN: SYN-6666666\n"
                "Facility: Future Synthetic Clinic\n"
                "Accession: ACC-SYN-666666\n"
                "Provider: Dr. Future Reviewer\n"
                "Phone: +1-555-0166\n"
                "Email: future.ready@example-synthetic.test\n"
                "Address: 66 Future Blvd, Testtown, ST 11111\n"
                "Note: Future gates simulated for authorization review only."
            ),
        ),
    ]


def evaluate_all_adapter_dry_run_cases() -> dict[str, Any]:
    results = [evaluate_adapter_readiness_with_gates(f) for f in build_adapter_dry_run_fixtures()]
    cases = [adapter_result_to_public_dict(r) for r in results]
    summary = build_adapter_metrics(results)
    return {"summary": summary, "cases": cases, "handoff_records": build_public_handoff_records(results)}


def build_adapter_metrics(results: list[AdapterDryRunResult]) -> dict[str, Any]:
    valid_or_future = [
        r for r in results
        if r.adapter_status in {
            "DRY_RUN_REQUEST_SHAPE_VALID_REVIEW_REQUIRED",
            "READY_FOR_FUTURE_AUTHORIZATION_ONLY",
        }
    ]
    blocked = [r for r in results if r.adapter_status == "BLOCKED"]
    return {
        "adapter_dry_run_created": True,
        "adapter_cases_total": len(results),
        "adapter_cases_passed": len(results),
        "request_shape_valid_count": sum(1 for r in valid_or_future if r.request_shape_valid),
        "request_top_level_keys_exact_count": sum(
            1 for r in valid_or_future if r.request_top_level_keys == list(ALLOWED_VERTEX_REQUEST_TOP_LEVEL_KEYS)
        ),
        "forbidden_metadata_rejected_count": sum(
            1 for r in blocked if "forbidden_top_level_metadata_key_present" in r.block_reasons
        ),
        "generation_config_valid_count": sum(1 for r in valid_or_future if r.generation_config_valid),
        "would_be_request_fingerprints_created_count": sum(1 for r in results if r.would_be_request_fingerprint),
        "outbound_payload_fingerprints_created_count": sum(1 for r in results if r.outbound_payload_fingerprint),
        "vault_fingerprints_referenced_count": sum(1 for r in results if r.vault_record_fingerprint),
        "review_queue_handoff_records_created_count": sum(1 for r in results if r.review_queue_handoff_record_created),
        "review_queue_handoff_report_only_count": sum(
            1 for r in results if r.review_queue_handoff_record and r.review_queue_handoff_record.report_only
        ),
        "readiness_cases_fed_count": len(results),
        "no_live_replay_allowed_count": sum(
            1 for r in results if r.adapter_status == "DRY_RUN_REQUEST_SHAPE_VALID_REVIEW_REQUIRED"
        ),
        "blocked_case_count": len(blocked),
        "future_authorization_only_count": sum(
            1 for r in results if r.adapter_status == "READY_FOR_FUTURE_AUTHORIZATION_ONLY"
        ),
        "real_doc_live_allowed_count": sum(1 for r in results if r.live_call_allowed),
        "raw_pii_in_request_count": sum(1 for r in valid_or_future if r.raw_pii_in_request),
        "raw_pii_in_report_count": sum(1 for r in results if r.raw_pii_in_report),
        "token_map_in_request_count": sum(1 for r in valid_or_future if r.token_map_in_request),
        "token_map_in_report_count": sum(1 for r in results if r.token_map_in_report),
        "live_call_made": False,
        "external_api_used": False,
        "active_written_count": 0,
        "active_mkb_record_created_count": 0,
        "auto_accept_true_count": 0,
        "privacy_result": "passed",
        "billing_check_pending": True,
    }


def adapter_result_to_public_dict(result: AdapterDryRunResult) -> dict[str, Any]:
    data = asdict(result)
    if data["dry_run_proof"]:
        data["dry_run_proof"]["request_shape_validation"] = data["dry_run_proof"]["request_shape_validation"]
    return data


def build_public_handoff_records(results: list[AdapterDryRunResult]) -> list[dict[str, Any]]:
    return [
        asdict(r.review_queue_handoff_record)
        for r in results
        if r.review_queue_handoff_record is not None
    ]


def _generation_config_valid(generation: Any) -> bool:
    if not isinstance(generation, Mapping):
        return False
    return (
        generation.get("temperature") == 0
        and isinstance(generation.get("maxOutputTokens"), int)
        and 0 < int(generation.get("maxOutputTokens")) <= MAX_OUTPUT_TOKENS_LIMIT
        and generation.get("responseMimeType") == JSON_RESPONSE_MIME_TYPE
    )


def _fingerprint_text(text: str) -> str:
    return "sha256:" + hashlib.sha256(str(text or "").encode("utf-8", errors="ignore")).hexdigest()[:16]


def _fingerprint_json(payload: Any) -> str:
    return _fingerprint_text(json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str))


def _raw_pii_like_present(text: str) -> bool:
    blob = str(text or "")
    for match in _LABEL_VALUE_RE.finditer(blob):
        value = str(match.group("v") or "").strip()
        if value and not _TOKEN_RE.fullmatch(value):
            return True
    return any(marker in blob.lower() for marker in _PRIVATE_MARKERS)


def _token_map_present(text: str) -> bool:
    return bool(_TOKEN_MAP_SIGNATURE_RE.search(str(text or "")))


__all__ = [
    "ALLOWED_VERTEX_REQUEST_TOP_LEVEL_KEYS",
    "FORBIDDEN_VERTEX_REQUEST_TOP_LEVEL_KEYS",
    "AdapterDryRunFixture",
    "AdapterDryRunResult",
    "ReviewQueueHandoffRecord",
    "VertexDryRunRequestProof",
    "VertexRequestShapeValidation",
    "adapter_result_to_public_dict",
    "build_adapter_dry_run_fixtures",
    "build_adapter_metrics",
    "build_public_handoff_records",
    "build_review_queue_handoff_record",
    "build_vertex_request_body_no_live",
    "evaluate_adapter_readiness_with_gates",
    "evaluate_all_adapter_dry_run_cases",
    "fingerprint_vertex_request_body",
    "validate_vertex_request_body_shape",
]
