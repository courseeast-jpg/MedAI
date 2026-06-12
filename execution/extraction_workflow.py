"""Safe AI extraction workflow seam for 15A.

This orchestrator accepts local text/OCR layout summaries, passes through a
privacy-gate placeholder, calls only the fake local adapter, validates the
returned schema, and returns review-bound source packages. It never writes
active MKB records and never calls external APIs.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Any

from app.source_extraction_packages import source_package_from_ai_draft
from execution.ai_external_call_dry_run import (
    build_ai_external_call_dry_run,
    dry_run_audit_to_public_dict,
    dry_run_decision_to_public_dict,
)
from execution.ai_budget_guard import AIBudgetGuard, budget_guard_to_public_dict
from execution.ai_extraction_adapter import (
    AIExtractionAdapterInput,
    AIExtractionPackageDraft,
    ExtractionWorkflowContext,
    ExtractionWorkflowResult,
    FakeAIExtractionAdapter,
)
from execution.ai_payload_policy import AIPayloadPolicy, payload_policy_to_public_dict
from execution.gemini_extraction_adapter import build_gemini_adapter_status
from execution.ai_provider_enablement import (
    evaluate_real_provider_execution_readiness,
    real_provider_credential_to_public_dict,
    real_provider_enablement_to_public_dict,
    real_provider_safety_checklist_to_public_dict,
)
from execution.ai_privacy_gate import (
    AIExternalCallApprovalState,
    build_ai_external_call_audit_record,
    privacy_gate_to_public_dict,
    run_ai_privacy_gate,
)
from execution.ai_provider_registry import (
    AIProviderRegistry,
    provider_readiness_to_public_dict,
    provider_selection_to_public_dict,
)


PLACEHOLDER_TOKENS = (
    "[PATIENT_1]",
    "[DOB_1]",
    "[DATE_1]",
    "[MRN_1]",
    "[ACCESSION_1]",
    "[FACILITY_1]",
    "[PROVIDER_1]",
    "[ADDRESS_1]",
    "[PHONE_1]",
    "[EMAIL_1]",
    "[INSURANCE_ID_1]",
)


def run_ai_extraction_workflow(
    context: ExtractionWorkflowContext,
    *,
    adapter: Any | None = None,
) -> ExtractionWorkflowResult:
    adapter = adapter or FakeAIExtractionAdapter()
    privacy = run_ai_privacy_gate(raw_text=context.raw_text_local_only, payload_type=context.payload_type)
    approval_state = AIExternalCallApprovalState(state=context.operator_approval_state)
    budget = AIBudgetGuard(
        provider_name=context.provider_name,
        model_name=context.model_name,
        session_budget_cap_usd=context.session_budget_cap_usd,
        monthly_budget_cap_usd=context.monthly_budget_cap_usd,
    ).evaluate(
        estimated_input_tokens=context.estimated_input_tokens,
        estimated_output_tokens=context.estimated_output_tokens,
    )
    policy_approval_state = approval_state
    policy_provider_mode = context.provider_mode
    policy_provider_name = context.provider_name
    policy_fail_real_providers = True
    if context.external_call_mode == "dry_run":
        policy_approval_state = AIExternalCallApprovalState(
            state="approved" if context.operator_approval_state == "approved_for_dry_run" else context.operator_approval_state
        )
        policy_provider_mode = "external_candidate"
        policy_fail_real_providers = False
    payload_policy = AIPayloadPolicy(fail_real_providers_in_15b=policy_fail_real_providers).evaluate(
        payload_type=context.payload_type,
        privacy_gate_result=privacy,
        operator_approval_state=policy_approval_state,
        budget_result=budget,
        provider_mode=policy_provider_mode,
        provider_name=policy_provider_name,
    )
    provider_registry = AIProviderRegistry()
    provider_readiness = provider_registry.validate_provider_readiness(
        provider_name=context.provider_name,
        operator_approval_state=context.operator_approval_state,
        privacy_gate_result=privacy_gate_to_public_dict(privacy),
        payload_policy_result=payload_policy_to_public_dict(payload_policy),
        budget_guard_result=budget_guard_to_public_dict(budget),
    )
    provider_selection = provider_registry.build_selection_state(
        requested_provider=context.provider_name,
        operator_approval_state=context.operator_approval_state,
        privacy_gate_result=privacy_gate_to_public_dict(privacy),
        payload_policy_result=payload_policy_to_public_dict(payload_policy),
        budget_guard_result=budget_guard_to_public_dict(budget),
    )
    provider_selection_public = provider_selection_to_public_dict(provider_selection)
    provider_readiness_public = provider_readiness_to_public_dict(provider_readiness)
    dry_run = build_ai_external_call_dry_run(
        requested_provider=provider_selection_public["requested_provider"],
        effective_provider=provider_selection_public["effective_provider"],
        model_name=provider_readiness_public["model_name"],
        provider_enabled=provider_readiness_public["provider_enabled"],
        operator_approval_state=context.operator_approval_state,
        privacy_gate_result=privacy_gate_to_public_dict(privacy),
        payload_policy_result=payload_policy_to_public_dict(payload_policy),
        budget_guard_result=budget_guard_to_public_dict(budget),
        dry_run_mode=context.external_call_mode == "dry_run",
    )
    adapter_input = AIExtractionAdapterInput(
        source_class=context.source_class,
        safe_source_document_id=context.safe_source_document_id,
        selected_document_category=context.selected_document_category,
        selected_specialty_domain=context.selected_specialty_domain,
        source_modality=context.source_modality,
        text_character_bucket=context.text_character_bucket,
        line_count_bucket=context.line_count_bucket,
        layout_hints=dict(context.layout_hints or {}),
    )
    draft = adapter.extract(adapter_input)
    errors = validate_ai_package_draft(draft)
    packages = [] if errors else [source_package_from_ai_draft(draft)]
    audit = build_ai_external_call_audit_record(
        source_id=context.safe_source_document_id,
        adapter_name=str(getattr(adapter, "adapter_name", adapter.__class__.__name__)),
        privacy_gate_result=privacy,
        payload_policy_result=payload_policy,
        budget_result=budget,
        approval_state=approval_state,
    )
    privacy_public = privacy_gate_to_public_dict(privacy)
    payload_policy_public = payload_policy_to_public_dict(payload_policy)
    budget_public = budget_guard_to_public_dict(budget)
    provider_public = provider_readiness_public
    selection_public = provider_selection_public
    dry_run_decision_public = dry_run_decision_to_public_dict(dry_run)
    dry_run_audit_public = dry_run_audit_to_public_dict(dry_run)
    enablement_provider_name = (
        selection_public["requested_provider"]
        if context.real_provider_enablement_mode == "readiness_check"
        else "fake_local"
    )
    provider_enablement = evaluate_real_provider_execution_readiness(
        provider_name=enablement_provider_name,
        operator_approval_state=context.operator_approval_state,
        dry_run_decision_result=dry_run_decision_public,
    )
    provider_enablement_public = real_provider_enablement_to_public_dict(provider_enablement)
    credential_public = real_provider_credential_to_public_dict(provider_enablement)
    safety_checklist_public = real_provider_safety_checklist_to_public_dict(provider_enablement)
    gemini_adapter_status_public = build_gemini_adapter_status(
        selected_provider=selection_public["requested_provider"],
        real_provider_execution_block_reason=provider_enablement_public.get(
            "real_provider_execution_block_reason",
            "real_provider_execution_disabled_by_policy",
        ),
    )
    operator_preview = build_operator_preview(
        packages,
        privacy_gate_result=privacy_public,
        payload_policy_result=payload_policy_public,
        budget_guard_result=budget_public,
        provider_registry_result=provider_public,
        provider_selection_result=selection_public,
        dry_run_decision_result=dry_run_decision_public,
        real_provider_enablement_result=provider_enablement_public,
        credential_readiness_result=credential_public,
        gemini_adapter_status_result=gemini_adapter_status_public,
    )
    return ExtractionWorkflowResult(
        adapter_name=str(getattr(adapter, "adapter_name", adapter.__class__.__name__)),
        packages=packages,
        privacy_gate_status=privacy.privacy_gate_status,
        pii_redaction_required=privacy.pii_detected_count > 0,
        pii_detected_count=privacy.pii_detected_count,
        pii_redacted_count=privacy.pii_redacted_count,
        pii_token_map_local_only=privacy.pii_token_map_local_only,
        external_payload_allowed=False,
        external_payload_preview_available=privacy.redacted_payload_preview_available,
        external_call_requires_operator_approval=True,
        external_api_used=False,
        final_external_call_allowed=False,
        active_written_count=0,
        auto_accept=False,
        review_required=True,
        review_bound_package_count=len(packages),
        package_types_tested=[str(getattr(draft, "document_type", "Unknown"))],
        operator_preview=operator_preview,
        privacy_gate_result=privacy_public,
        payload_policy_result=payload_policy_public,
        budget_guard_result=budget_public,
        audit_result=audit,
        provider_registry_result=provider_public,
        provider_selection_result=selection_public,
        dry_run_decision_result=dry_run_decision_public,
        dry_run_audit_result=dry_run_audit_public,
        real_provider_enablement_result=provider_enablement_public,
        credential_readiness_result=credential_public,
        real_provider_safety_checklist_result=safety_checklist_public,
        validation_errors=errors,
        gemini_adapter_status_result=gemini_adapter_status_public,
    )


def validate_ai_package_draft(draft: AIExtractionPackageDraft) -> list[str]:
    errors: list[str] = []
    if getattr(draft, "external_api_used", False):
        errors.append("external_api_used_must_be_false")
    if getattr(draft, "auto_accept", True) is not False:
        errors.append("auto_accept_must_be_false")
    if getattr(draft, "review_required", False) is not True:
        errors.append("review_required_must_be_true")
    if str(getattr(draft, "package_status", "")) != "review-bound":
        errors.append("package_status_must_be_review_bound")
    if not list(getattr(draft, "sections", []) or []):
        errors.append("sections_required")
    for section in list(getattr(draft, "sections", []) or []):
        if not str(getattr(section, "heading", "") or "").strip():
            errors.append("section_heading_required")
        for observation in list(getattr(section, "observations", []) or []):
            if getattr(observation, "auto_accept", True) is not False:
                errors.append("observation_auto_accept_must_be_false")
            if getattr(observation, "review_required", False) is not True:
                errors.append("observation_review_required_must_be_true")
    return sorted(set(errors))


def build_operator_preview(
    packages: list[dict[str, Any]],
    *,
    privacy_gate_result: dict[str, Any] | None = None,
    payload_policy_result: dict[str, Any] | None = None,
    budget_guard_result: dict[str, Any] | None = None,
    provider_registry_result: dict[str, Any] | None = None,
    provider_selection_result: dict[str, Any] | None = None,
    dry_run_decision_result: dict[str, Any] | None = None,
    real_provider_enablement_result: dict[str, Any] | None = None,
    credential_readiness_result: dict[str, Any] | None = None,
    gemini_adapter_status_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    preview_packages: list[dict[str, Any]] = []
    for package in packages:
        preview_packages.append(
            {
                "package_id": package["package_id"],
                "label": "AI-assisted extraction draft - review-bound only",
                "document_type": package["detected_document_family_type"],
                "section_count": len(package["sections"]),
                "observation_count": sum(len(section["observations"]) for section in package["sections"]),
                "sections": [
                    {
                        "heading": section["heading"],
                        "narrative_label": section["narrative_label"],
                        "observations": [
                            {
                                "label": observation["label"],
                                "value": observation["value"],
                                "flag": observation["flag"],
                                "unit": observation["unit"],
                                "reference_interval": observation["reference_interval"],
                                "review_status": observation["review_status"],
                            }
                            for observation in section["observations"]
                        ],
                    }
                    for section in package["sections"]
                ],
            }
        )
    preview = {
        "visible": bool(preview_packages),
        "packages": preview_packages,
        "review_required": True,
        "auto_accept": False,
        "ai_external_api_status": "disabled",
        "privacy_gate_status": (privacy_gate_result or {}).get("privacy_gate_status", ""),
        "pii_detected_count": int((privacy_gate_result or {}).get("pii_detected_count", 0)),
        "pii_redacted_count": int((privacy_gate_result or {}).get("pii_redacted_count", 0)),
        "pii_categories": sorted((privacy_gate_result or {}).get("token_category_counts", {}).keys()),
        "redacted_payload_preview_available": bool(
            (privacy_gate_result or {}).get("redacted_payload_preview_available", False)
        ),
        "external_call_approval_status": (dry_run_decision_result or {}).get(
            "operator_approval_state",
            (payload_policy_result or {}).get("operator_approval_state", "not_requested"),
        ),
        "budget_allowed": bool((budget_guard_result or {}).get("budget_allowed", False)),
        "budget_fail_reason": (budget_guard_result or {}).get("budget_fail_reason", ""),
        "payload_policy_allowed": bool((payload_policy_result or {}).get("payload_policy_allowed", False)),
        "final_external_call_allowed": False,
        "selected_provider": (provider_registry_result or {}).get("provider_name", ""),
        "requested_provider": (provider_selection_result or {}).get("requested_provider", ""),
        "effective_provider": (provider_selection_result or {}).get("effective_provider", ""),
        "provider_enabled": bool((provider_registry_result or {}).get("provider_enabled", False)),
        "provider_model_name": (provider_registry_result or {}).get("model_name", ""),
        "provider_mode": (provider_registry_result or {}).get("provider_mode", ""),
        "provider_requires_operator_approval": bool(
            (provider_registry_result or {}).get("requires_operator_approval", False)
        ),
        "provider_fail_closed_reason": (provider_registry_result or {}).get("fail_closed_reason", ""),
        "provider_execution_allowed": bool(
            (provider_selection_result or {}).get("provider_execution_allowed", False)
        ),
        "provider_execution_block_reason": (provider_selection_result or {}).get(
            "provider_execution_block_reason", ""
        ),
        "dry_run_mode_status": (
            "dry-run only - real provider execution remains disabled"
            if (dry_run_decision_result or {}).get("dry_run_external_call_allowed")
            else "dry-run blocked"
        ),
        "dry_run_external_call_allowed": bool(
            (dry_run_decision_result or {}).get("dry_run_external_call_allowed", False)
        ),
        "dry_run_fail_closed_reason": (dry_run_decision_result or {}).get("fail_closed_reason", ""),
        "real_network_call_used": False,
        "real_provider_execution_enabled": False,
        "real_provider_execution_block_reason": (real_provider_enablement_result or {}).get(
            "real_provider_execution_block_reason",
            "real_provider_execution_disabled_by_policy",
        ),
        "real_provider_execution_notice": "Real provider execution disabled by policy",
        "provider_message": (
            "Provider disabled by policy"
            if not bool((provider_registry_result or {}).get("provider_enabled", False))
            else "Provider available for local fake path only"
        ),
        "operator_notice": "No external AI call was made",
        "operator_notice_sentence": "No external AI call was made.",
    }
    gemini_status = dict(gemini_adapter_status_result or {})
    # Non-credential Gemini status is always safe to surface (no "credential"
    # or "api_key" substrings).
    preview.update(
        {
            "gemini_adapter_installed": bool(gemini_status.get("gemini_adapter_installed", True)),
            "gemini_adapter_status_message": gemini_status.get(
                "gemini_adapter_status_message",
                "Gemini adapter installed but real execution disabled by policy",
            ),
            "gemini_selected": bool(gemini_status.get("gemini_selected", False)),
            "gemini_real_call_attempted": False,
            "gemini_prompt_contract_version": gemini_status.get("prompt_contract_version", ""),
            "gemini_schema_contract_version": gemini_status.get("schema_contract_version", ""),
            "gemini_real_provider_execution_enabled": False,
            "gemini_real_provider_execution_block_reason": gemini_status.get(
                "real_provider_execution_block_reason",
                "real_provider_execution_disabled_by_policy",
            ),
        }
    )
    # Credential-name/presence fields only appear once an explicit real-provider
    # readiness check has run (matching 15F behavior). 15D/15E disabled-mode
    # previews stay free of any credential/api-key strings.
    if (credential_readiness_result or {}).get("credential_env_var_name"):
        preview.update(
            {
                "credential_env_var_name": credential_readiness_result.get("credential_env_var_name", ""),
                "credential_present": bool(credential_readiness_result.get("credential_present", False)),
                "credential_value_redacted": bool(
                    credential_readiness_result.get("credential_value_redacted", False)
                ),
                "gemini_credential_env_var_name": gemini_status.get("credential_env_var_name", ""),
                "gemini_credential_present": bool(gemini_status.get("credential_present", False)),
            }
        )
    return preview


def workflow_result_to_public_dict(result: ExtractionWorkflowResult) -> dict[str, Any]:
    payload = asdict(result)
    payload["external_api_used"] = False
    payload["active_written_count"] = 0
    payload["auto_accept"] = False
    payload["review_required"] = True
    payload["final_external_call_allowed"] = False
    payload["external_payload_allowed"] = False
    return payload


__all__ = [
    "PLACEHOLDER_TOKENS",
    "run_ai_extraction_workflow",
    "validate_ai_package_draft",
    "build_operator_preview",
    "workflow_result_to_public_dict",
]
