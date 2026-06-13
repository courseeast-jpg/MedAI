"""No-live tests for the 15Z-B Vertex real-document PII-stripping proof."""
from __future__ import annotations

import inspect
import json

from execution.vertex_real_doc_pii_stripping_proof import (
    TOKEN_CLASSES,
    PiiStrippingFixture,
    build_isolated_pii_vault_record,
    build_outbound_safe_payload,
    build_pii_stripping_readiness_case,
    redact_pii_like_values,
    validate_no_raw_pii_in_payload,
    validate_vault_not_in_payload_or_report,
)
from execution.vertex_real_doc_readiness_gates import (
    PROVENANCE_REAL_PRIVATE,
    PROVENANCE_REDACTED_REAL_LIKE,
    PROVENANCE_UNKNOWN,
)

# Synthetic raw fixtures are held locally in tests; never written to reports.
NOTE = (
    "Patient: Jordan A. Synthsample\n"
    "DOB: 1991-04-15\n"
    "MRN: SYN-0012345\n"
    "Note: Sodium 140 mmol/L within reference range."
)
LAB = (
    "Facility: Northshore Synthetic Imaging Center\n"
    "Accession: ACC-SYN-778899\n"
    "Provider: Dr. Riley Testname\n"
    "Result: Hemoglobin 13.2 g/dL within range."
)
CONTACT = (
    "Phone: +1-555-0142\n"
    "Email: sample.patient@example-synthetic.test\n"
    "Address: 742 Synthetic Way, Springfield, ST 00000\n"
    "Result: Urinalysis clear."
)
REPEAT = (
    "MRN: SYN-0099001\n"
    "Patient: Casey Q. Sampleton\n"
    "MRN: SYN-0099001\n"
    "Note: Glucose 95 mg/dL."
)
ALL_FIELDS = (
    "Patient: Quinn Readysample\n"
    "DOB: 1975-06-30\n"
    "MRN: SYN-0102030\n"
    "Facility: Lakeside Synthetic Clinic\n"
    "Accession: ACC-SYN-101010\n"
    "Provider: Dr. Alex Reviewer\n"
    "Phone: +1-555-0199\n"
    "Email: quinn.ready@example-synthetic.test\n"
    "Address: 12 Sample Blvd, Testtown, ST 11111\n"
    "Note: Comprehensive panel within range."
)


def _classes(red):
    return {f.pii_class for f in red.findings}


def test_detects_and_tokenizes_patient_name():
    red = redact_pii_like_values(NOTE)
    assert "PATIENT_NAME" in _classes(red)
    assert "[PATIENT_NAME_1]" in red.redacted_text
    assert "Jordan A. Synthsample" not in red.redacted_text


def test_detects_and_tokenizes_dob_date():
    red = redact_pii_like_values(NOTE)
    assert "DATE" in _classes(red)
    assert "1991-04-15" not in red.redacted_text
    assert "[DATE_1]" in red.redacted_text


def test_detects_and_tokenizes_mrn():
    red = redact_pii_like_values(NOTE)
    assert "MRN" in _classes(red)
    assert "SYN-0012345" not in red.redacted_text


def test_detects_and_tokenizes_facility():
    red = redact_pii_like_values(LAB)
    assert "FACILITY" in _classes(red)
    assert "Northshore Synthetic Imaging Center" not in red.redacted_text


def test_detects_and_tokenizes_phone_email_address():
    red = redact_pii_like_values(CONTACT)
    classes = _classes(red)
    assert {"PHONE", "EMAIL", "ADDRESS"} <= classes
    for raw in ("+1-555-0142", "sample.patient@example-synthetic.test", "742 Synthetic Way, Springfield, ST 00000"):
        assert raw not in red.redacted_text


def test_detects_and_tokenizes_accession():
    red = redact_pii_like_values(LAB)
    assert "ACCESSION" in _classes(red)
    assert "ACC-SYN-778899" not in red.redacted_text


def test_detects_and_tokenizes_provider():
    red = redact_pii_like_values(LAB)
    assert "PROVIDER" in _classes(red)
    assert "Dr. Riley Testname" not in red.redacted_text


def test_repeated_identifier_maps_to_same_token():
    red = redact_pii_like_values(REPEAT)
    assert red.repeated_token_count >= 1
    # Only one MRN token despite two occurrences.
    assert red.redacted_text.count("[MRN_1]") == 2
    assert "[MRN_2]" not in red.redacted_text


def test_outbound_payload_contains_tokens_and_no_raw_pii():
    red = redact_pii_like_values(ALL_FIELDS)
    out = build_outbound_safe_payload(red)
    assert out.contains_tokens is True
    assert validate_no_raw_pii_in_payload(out.outbound_text, red) is True
    for raw in ("Quinn Readysample", "SYN-0102030", "quinn.ready@example-synthetic.test"):
        assert raw not in out.outbound_text


def test_token_map_not_in_outbound_payload():
    red = redact_pii_like_values(NOTE)
    out = build_outbound_safe_payload(red)
    vault = build_isolated_pii_vault_record(red)
    payload_safe, _ = validate_vault_not_in_payload_or_report(out.outbound_text, "", red)
    assert payload_safe is True
    assert vault._serialized_map not in out.outbound_text


def test_token_map_not_in_public_reports():
    import scripts.run_medai_vertex_real_doc_pii_stripping_proof_no_live_15z_b as mod
    cases = mod.build_cases()
    vault_proof = json.dumps(mod._vault_proof(cases))
    cases_blob = json.dumps([mod._public_case(c) for c in cases])
    for f in mod.build_fixtures():
        red = redact_pii_like_values(f.raw_text)
        vault = build_isolated_pii_vault_record(red)
        assert vault._serialized_map not in vault_proof
        assert vault._serialized_map not in cases_blob


def test_report_preview_contains_no_raw_pii():
    import scripts.run_medai_vertex_real_doc_pii_stripping_proof_no_live_15z_b as mod
    cases = mod.build_cases()
    preview = mod._outbound_preview(cases)
    for f in mod.build_fixtures():
        red = redact_pii_like_values(f.raw_text)
        for raw in red._detected_values:
            assert raw not in preview


def test_vault_record_isolated_proof_only():
    red = redact_pii_like_values(ALL_FIELDS)
    vault = build_isolated_pii_vault_record(red)
    pub = vault.public_dict()
    assert pub["isolated"] is True
    assert pub["vault_fingerprint"].startswith("sha256:")
    assert "token_map" not in pub and "_serialized_map" not in pub
    blob = json.dumps(pub)
    for raw in red._detected_values:
        assert raw not in blob


def test_unredacted_pii_residue_blocks():
    fx = PiiStrippingFixture(
        case_id="residue", declared_provenance=PROVENANCE_REDACTED_REAL_LIKE,
        content_marker="redacted real-like note layout", expected_block=True,
        raw_text="Patient: Morgan Residuesample\nNote: Reviewed with Morgan Residuesample.",
    )
    case = build_pii_stripping_readiness_case(fx)
    assert case["blocked"] is True
    assert case["raw_pii_in_outbound_payload"] is True
    assert case["readiness_status"] == "BLOCKED_UNREDACTED_PII_RESIDUE"


def test_token_map_leak_in_payload_blocks():
    fx = PiiStrippingFixture(
        case_id="leak_payload", declared_provenance=PROVENANCE_REDACTED_REAL_LIKE,
        content_marker="redacted real-like note layout", expected_block=True,
        inject_token_map_in_payload=True,
        raw_text="Patient: Drew Leaksample\nMRN: SYN-0055667\nNote: ok.",
    )
    case = build_pii_stripping_readiness_case(fx)
    assert case["blocked"] is True
    assert case["token_map_in_outbound_payload"] is True
    assert case["readiness_status"] == "BLOCKED_TOKEN_MAP_LEAK_IN_OUTBOUND_PAYLOAD"


def test_token_map_leak_in_report_blocks():
    fx = PiiStrippingFixture(
        case_id="leak_report", declared_provenance=PROVENANCE_REDACTED_REAL_LIKE,
        content_marker="redacted real-like note layout", expected_block=True,
        inject_token_map_in_report=True,
        raw_text="Patient: Avery Reportleak\nMRN: SYN-0066778\nNote: ok.",
    )
    case = build_pii_stripping_readiness_case(fx)
    assert case["blocked"] is True
    assert case["token_map_in_report"] is True
    assert case["readiness_status"] == "BLOCKED_TOKEN_MAP_LEAK_IN_REPORT"


def test_unknown_provenance_blocks_even_redacted():
    fx = PiiStrippingFixture(
        case_id="unknown", declared_provenance=PROVENANCE_UNKNOWN,
        content_marker="unknown source layout", expected_block=True,
        raw_text="Patient: Sam Unknownsrc\nMRN: SYN-0077889\nNote: ok.",
    )
    case = build_pii_stripping_readiness_case(fx)
    assert case["blocked"] is True
    assert case["readiness_status"] == "BLOCKED_PROVENANCE_NOT_REDACTED_REAL_LIKE"


def test_real_private_marker_blocks_even_redacted():
    fx = PiiStrippingFixture(
        case_id="real_private", declared_provenance=PROVENANCE_REAL_PRIVATE,
        content_marker="declared real private source layout", expected_block=True,
        raw_text="Patient: Pat Realprivate\nMRN: SYN-0088990\nNote: ok.",
    )
    case = build_pii_stripping_readiness_case(fx)
    assert case["blocked"] is True
    assert case["readiness_status"] == "BLOCKED_PROVENANCE_NOT_REDACTED_REAL_LIKE"


def test_sanitized_redacted_real_like_allows_no_live_replay_only():
    fx = PiiStrippingFixture(
        case_id="ready", declared_provenance=PROVENANCE_REDACTED_REAL_LIKE,
        content_marker="redacted real-like comprehensive layout", expected_block=False,
        raw_text=ALL_FIELDS,
    )
    case = build_pii_stripping_readiness_case(fx)
    assert case["blocked"] is False
    assert case["readiness_status"] == "READY_FOR_NO_LIVE_REPLAY_ONLY"
    assert case["framework_readiness_status"] == "READY_FOR_FUTURE_AUTHORIZATION_ONLY"


def test_readiness_framework_keeps_live_call_disallowed_and_review_bound():
    fx = PiiStrippingFixture(
        case_id="ready2", declared_provenance=PROVENANCE_REDACTED_REAL_LIKE,
        content_marker="redacted real-like comprehensive layout", expected_block=False,
        raw_text=ALL_FIELDS,
    )
    case = build_pii_stripping_readiness_case(fx)
    assert case["live_call_allowed"] is False
    assert case["external_api_used"] is False
    assert case["active_write_allowed"] is False
    assert case["auto_accept_allowed"] is False
    assert case["review_required"] is True


def test_nine_token_classes_defined():
    assert len(TOKEN_CLASSES) == 9
    assert set(TOKEN_CLASSES) == {
        "PATIENT_NAME", "PROVIDER", "FACILITY", "MRN", "ACCESSION",
        "DATE", "PHONE", "EMAIL", "ADDRESS",
    }


def test_no_provider_call_or_live_gate_in_module_source():
    import execution.vertex_real_doc_pii_stripping_proof as mod
    src = inspect.getsource(mod)
    for marker in (
        "requests.", "urllib.request", "httpx.", "generate_content",
        "acquire_google_cloud_access_token", "socket.", "MEDAI_VERTEX_LIVE_SMOKE_ALLOWED",
        "MEDAI_VERTEX_SEMANTIC_LIVE_COMPARE_ALLOWED",
        "MEDAI_VERTEX_CALIBRATION_BATCH_SYNTHETIC_LIVE_ALLOWED",
    ):
        assert marker not in src


def test_script_metrics_all_invariants_hold():
    import scripts.run_medai_vertex_real_doc_pii_stripping_proof_no_live_15z_b as mod
    cases = mod.build_cases()
    m = mod._build_metrics(cases)
    assert m["pii_stripping_proof_created"] is True
    assert m["pii_cases_total"] == 10
    assert m["pii_cases_passed"] == 10
    assert m["token_classes_detected_count"] == 9
    assert m["deterministic_repeated_token_count"] >= 1
    assert m["raw_pii_in_outbound_payload_count"] == 0
    assert m["raw_pii_in_report_count"] == 0
    assert m["token_map_in_outbound_payload_count"] == 0
    assert m["token_map_in_report_count"] == 0
    assert m["vault_records_created_count"] == 10
    assert m["vault_records_isolated_count"] == 10
    assert m["readiness_cases_fed_count"] == 10
    assert m["no_live_replay_allowed_count"] == 5
    assert m["blocked_case_count"] == 5
    assert m["real_doc_live_allowed_count"] == 0
    assert m["all_cases_review_required"] is True
    assert m["all_cases_live_call_blocked"] is True
    assert m["live_call_made"] is False and m["external_api_used"] is False
    assert m["active_written_count"] == 0 and m["auto_accept_true_count"] == 0
    assert m["active_mkb_record_created_count"] == 0


def test_no_credentials_or_raw_pii_in_published_payloads():
    import scripts.run_medai_vertex_real_doc_pii_stripping_proof_no_live_15z_b as mod
    cases = mod.build_cases()
    blob = "\n".join(
        [
            json.dumps([mod._public_case(c) for c in cases]),
            json.dumps(mod._vault_proof(cases)),
            mod._matrix_markdown(mod._build_metrics(cases), cases),
            mod._outbound_preview(cases),
            mod._implementation_markdown(mod._build_metrics(cases)),
        ]
    )
    for token in ("ya29.", "AIza", "Bearer ", "Authorization", "C:\\", "/home/"):
        assert token not in blob
    for f in mod.build_fixtures():
        red = redact_pii_like_values(f.raw_text)
        for raw in red._detected_values:
            assert raw not in blob
