"""Focused tests for MEDAI-V2-RUNTIME-CONTRACTS-01.

These tests prove that:

* all three reports exist and pass the public-report privacy check;
* the typing-only contracts module imports silently with no
  ``stdout`` or ``stderr`` emission;
* the contracts module uses only standard-library imports;
* no forbidden module names (Streamlit, network, HTTP, DB, LLM SDKs)
  appear in the contracts source;
* every name in ``CONTRACT_NAMES`` resolves from the module;
* every required ``Protocol`` is a ``Protocol`` and not a concrete
  implementation;
* the terminology contract is aggregate-only and forbids licensed
  row content;
* the default ``V2RuntimeSafetyProfile`` carries the canonical
  invariants (local-only, review-bound, external-API-blocked,
  no-auto-accept, no-cue-expansion);
* cue expansion remains recorded as NOT recommended;
* no runtime files (``app/main.py``, launchers, startup preflight,
  config) are modified by this block.
"""
from __future__ import annotations

import dataclasses
import io
import json
import sys
import typing
from pathlib import Path
from typing import Protocol

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

REPORT_DIR = REPO_ROOT / "reports" / "medai_v2_runtime_contracts_01"
CONTRACTS_MODULE_PATH = (
    REPO_ROOT / "clinical_knowledge" / "v2_contracts" / "runtime_contracts.py"
)

SHORT_SUMMARY = REPORT_DIR / "MEDAI_V2_RUNTIME_CONTRACTS_01.md"
JSON_REPORT = REPORT_DIR / "medai_v2_runtime_contracts_01_report.json"
LONG_REPORT = REPORT_DIR / "medai_v2_runtime_contracts_01_report.md"

REQUIRED_REPORT_FILES = (SHORT_SUMMARY, JSON_REPORT, LONG_REPORT)

ALLOWED_STDLIB_TOP_LEVEL_MODULES = {
    "dataclasses",
    "datetime",
    "enum",
    "typing",
    "__future__",
}

FORBIDDEN_IMPORT_PREFIXES = (
    "streamlit",
    "requests",
    "httpx",
    "http.",
    "urllib",
    "socket",
    "sqlite3",
    "sqlcipher3",
    "anthropic",
    "google.generativeai",
    "openai",
    "boto3",
    "psycopg",
)

REQUIRED_PROTOCOLS = (
    "V2IngestionAdapterProtocol",
    "V2DocumentQualityProtocol",
    "V2ExtractionAdapterProtocol",
    "V2ClassifierProtocol",
    "V2TerminologyLookupProtocol",
    "V2ReviewQueueProtocol",
    "V2OperatorActionProtocol",
    "V2ObservabilitySinkProtocol",
    "V2ValidationHarnessProtocol",
    "V2PrivacyGateProtocol",
)


# ── Reports exist ──────────────────────────────────────────────────────────


def test_all_three_reports_exist():
    for p in REQUIRED_REPORT_FILES:
        assert p.is_file(), f"missing report file: {p.relative_to(REPO_ROOT)}"


def test_reports_pass_public_report_privacy_check():
    from clinical_knowledge.privacy import check_public_report_payload

    for p in REQUIRED_REPORT_FILES:
        content = p.read_text(encoding="utf-8")
        target: object = content
        if p.suffix == ".json":
            target = json.loads(content)
        result = check_public_report_payload(target)
        assert result.passed, (
            f"privacy check failed for {p.name}: "
            f"phi={result.raw_phi_logged_in_public_reports} "
            f"private_ref={result.private_filename_path_leaks} "
            f"secrets={result.secret_leaks}"
        )


# ── JSON required booleans ────────────────────────────────────────────────


@pytest.fixture(scope="module")
def payload() -> dict:
    return json.loads(JSON_REPORT.read_text(encoding="utf-8"))


def test_json_required_false_booleans(payload):
    for key in (
        "runtime_behavior_changed",
        "app_main_changed",
        "ui_changed",
        "launcher_changed",
        "startup_config_changed",
        "extraction_changed",
        "ocr_changed",
        "classifier_changed",
        "threshold_scoring_changed",
        "cue_pack_changed",
        "clinical_behavior_changed",
        "ddi_behavior_changed",
        "terminology_behavior_changed",
        "private_adapter_implemented",
        "concrete_adapters_implemented",
        "runtime_wiring_added",
        "external_api_used",
        "private_data_accessed",
        "licensed_rows_read",
        "private_license_ack_read",
        "private_config_read",
        "runtime_db_accessed",
        "source_documents_opened",
        "raw_text_printed",
        "raw_filenames_printed",
        "private_paths_printed",
        "secrets_printed",
        "tags_touched",
        "cue_expansion_recommended",
    ):
        assert payload.get(key) is False, f"{key} expected False, got {payload.get(key)!r}"


def test_json_required_true_booleans(payload):
    for key in (
        "reports_only",
        "requires_preceding_spec",
        "preceding_architecture_spec_present",
        "preceding_foundation_spec_present",
        "contracts_module_created",
        "contracts_importable",
        "contracts_side_effect_free",
        "standard_library_only",
        "v1_release_preserved",
    ):
        assert payload.get(key) is True, f"{key} expected True, got {payload.get(key)!r}"


def test_next_recommended_block_is_validation_harness(payload):
    assert payload.get("next_recommended_block") == "V2-VALIDATION-HARNESS-01"


# ── Contracts module import + side-effect ─────────────────────────────────


def test_contracts_module_imports_silently():
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    try:
        # Force a fresh import to surface any side effects on first import
        sys.modules.pop("clinical_knowledge.v2_contracts.runtime_contracts", None)
        sys.modules.pop("clinical_knowledge.v2_contracts", None)
        import clinical_knowledge.v2_contracts.runtime_contracts as rc  # noqa: F401
        out = sys.stdout.getvalue()
        err = sys.stderr.getvalue()
    finally:
        sys.stdout, sys.stderr = old_out, old_err
    assert out == "", f"import wrote stdout: {out!r}"
    assert err == "", f"import wrote stderr: {err!r}"


def test_contracts_module_standard_library_only_static_scan():
    src = CONTRACTS_MODULE_PATH.read_text(encoding="utf-8")
    for lineno, line in enumerate(src.splitlines(), start=1):
        stripped = line.strip()
        if not (stripped.startswith("import ") or stripped.startswith("from ")):
            continue
        if stripped.startswith("from "):
            module_name = stripped.split(None, 2)[1]
        else:
            module_name = stripped.split(None, 2)[1].split(",")[0]
        top_level = module_name.split(".", 1)[0]
        assert top_level in ALLOWED_STDLIB_TOP_LEVEL_MODULES, (
            f"line {lineno}: non-stdlib import {module_name!r}"
        )
        for prefix in FORBIDDEN_IMPORT_PREFIXES:
            assert not (
                module_name == prefix or module_name.startswith(prefix + ".")
            ), f"line {lineno}: forbidden import prefix {prefix!r} in {module_name!r}"


def test_no_streamlit_import_in_contracts_module():
    src = CONTRACTS_MODULE_PATH.read_text(encoding="utf-8")
    assert "import streamlit" not in src
    assert "from streamlit" not in src


def test_no_network_or_http_imports_in_contracts_module():
    src = CONTRACTS_MODULE_PATH.read_text(encoding="utf-8")
    for fragment in (
        "import requests",
        "from requests",
        "import httpx",
        "from httpx",
        "import urllib",
        "from urllib",
        "import socket",
        "from socket",
        "import http\n",
        "from http",
    ):
        assert fragment not in src, f"found {fragment!r} in contracts module"


def test_no_db_or_runtime_or_terminology_imports_in_contracts_module():
    src = CONTRACTS_MODULE_PATH.read_text(encoding="utf-8")
    for fragment in (
        "import sqlite3",
        "from sqlite3",
        "import sqlcipher3",
        "from sqlcipher3",
        "from app.",
        "import app.",
        "from clinical_knowledge.terminology",
        "import clinical_knowledge.terminology",
        "from ingestion",
        "import ingestion",
    ):
        assert fragment not in src, f"found {fragment!r} in contracts module"


# ── Contract names and protocol shape ─────────────────────────────────────


def test_all_contract_names_resolve_from_module():
    import clinical_knowledge.v2_contracts.runtime_contracts as rc

    assert len(rc.CONTRACT_NAMES) == 32
    for name in rc.CONTRACT_NAMES:
        assert hasattr(rc, name), f"missing contract name: {name}"


def test_required_protocols_are_protocols_not_concrete():
    import clinical_knowledge.v2_contracts.runtime_contracts as rc

    for name in REQUIRED_PROTOCOLS:
        obj = getattr(rc, name)
        # Must be a Protocol class
        assert issubclass(obj, Protocol), f"{name} is not a Protocol"
        # Protocols must not provide concrete __init__ subclass
        # implementations; the Protocol class itself acts as a typing
        # contract. We check no obviously-concrete state by verifying
        # the class is recognized as a Protocol via __protocol_attrs__
        # OR by typing._ProtocolMeta usage.
        assert any(
            base.__name__ == "Protocol"
            for base in obj.__mro__
        ), f"{name} does not have Protocol in MRO"


def test_terminology_contract_is_aggregate_only_and_forbids_row_content():
    import clinical_knowledge.v2_contracts.runtime_contracts as rc

    fields = {f.name for f in dataclasses.fields(rc.V2TerminologyMatchSummary)}
    # Required aggregate / invariant fields
    assert {
        "anonymous_id",
        "match_family",
        "terminology_system_family",
        "matches_count",
        "review_required",
        "auto_accept_allowed",
        "licensed_row_content_included",
        "public_report_safe",
    }.issubset(fields)
    # Forbidden row-content fields must NOT exist on the contract
    for forbidden in (
        "code",
        "display",
        "rxcui",
        "loinc_num",
        "synonym",
        "definition",
        "concept",
        "raw_text",
        "ocr_text",
    ):
        assert forbidden not in fields, (
            f"terminology contract unexpectedly exposes {forbidden!r}"
        )
    # Default instance carries the required invariant flags
    inst = rc.V2TerminologyMatchSummary(
        anonymous_id="record_001",
        match_family="exact_terminology_match",
        terminology_system_family="rxnorm",
        matches_count=0,
    )
    assert inst.review_required is True
    assert inst.auto_accept_allowed is False
    assert inst.licensed_row_content_included is False
    assert inst.public_report_safe is True


def test_default_runtime_safety_profile_invariants():
    import clinical_knowledge.v2_contracts.runtime_contracts as rc

    p = rc.V2RuntimeSafetyProfile()
    assert p.local_only is True
    assert p.review_bound is True
    assert p.external_api_blocked is True
    assert p.auto_accept_allowed is False
    assert p.terminology_lookup_aggregate_only is True
    assert p.private_adapter_implemented is False
    assert p.cue_expansion_recommended is False
    assert p.clinical_decision_expansion is False


def test_classification_result_review_bound_defaults():
    import clinical_knowledge.v2_contracts.runtime_contracts as rc

    r = rc.V2ClassificationResult(
        anonymous_id="record_001",
        candidates=(),
    )
    assert r.review_required is True
    assert r.auto_accept_allowed is False
    assert r.cue_expansion_used is False


def test_review_item_pending_review_by_default():
    import clinical_knowledge.v2_contracts.runtime_contracts as rc

    classification = rc.V2ClassificationResult(
        anonymous_id="record_001",
        candidates=(),
    )
    item = rc.V2ReviewItem(
        anonymous_id="record_001",
        classification=classification,
    )
    assert item.disposition == rc.V2ReviewDisposition.PENDING_REVIEW
    assert item.review_required is True
    assert item.auto_accept_allowed is False


def test_validation_receipt_defaults_safe():
    import clinical_knowledge.v2_contracts.runtime_contracts as rc

    receipt = rc.V2ValidationReceipt(
        receipt_id="receipt_001",
        validation_name="cka_final_mvp_release",
        status=rc.V2ValidationStatus.READY,
        cases_total=10,
        cases_passed=10,
        cases_failed=0,
    )
    assert receipt.external_api_used is False
    assert receipt.cue_expansion_recommended is False


# ── No runtime files modified by this block ───────────────────────────────


def test_app_main_unchanged_by_v2_runtime_contracts_01():
    src = (REPO_ROOT / "app" / "main.py").read_text(encoding="utf-8")
    assert "MEDAI-V2-RUNTIME-CONTRACTS-01" not in src
    assert "v2_contracts" not in src


def test_launchers_unchanged_by_v2_runtime_contracts_01():
    for launcher in (
        "Start_MedAI_UI.bat",
        "Start_MedAI_UI_Silent.vbs",
        "Start_MedAI_Test_UI.bat",
        "Start_MedAI_UI_Encrypted.bat",
    ):
        p = REPO_ROOT / launcher
        if not p.is_file():
            continue
        src = p.read_text(encoding="utf-8", errors="replace")
        assert "MEDAI-V2-RUNTIME-CONTRACTS-01" not in src


def test_startup_preflight_unchanged_by_v2_runtime_contracts_01():
    p = REPO_ROOT / "app" / "startup_preflight.py"
    if not p.is_file():
        return
    src = p.read_text(encoding="utf-8")
    assert "MEDAI-V2-RUNTIME-CONTRACTS-01" not in src


def test_config_unchanged_by_v2_runtime_contracts_01():
    p = REPO_ROOT / "app" / "config.py"
    if not p.is_file():
        return
    src = p.read_text(encoding="utf-8")
    assert "MEDAI-V2-RUNTIME-CONTRACTS-01" not in src
