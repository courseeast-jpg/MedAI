"""Fail-closed payload policy for future external AI extraction calls."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


ALLOWED_FUTURE_PAYLOAD_TYPE = "redacted_text_layout_summary"
FORBIDDEN_PAYLOAD_TYPES = {
    "raw_pdf",
    "raw_image",
    "scanned_image",
    "unredacted_ocr_body",
    "external_vision_payload",
    "raw_file_upload",
    "provider_sdk_request",
}
REAL_PROVIDER_NAMES = {
    "gemini",
    "google_gemini",
    "claude",
    "anthropic",
    "openai",
    "ollama",
    "local_vision",
}


@dataclass(frozen=True)
class AIPayloadPolicyResult:
    payload_type: str
    provider_mode: str
    provider_name: str
    operator_approval_state: str
    payload_policy_allowed: bool
    final_external_call_allowed: bool
    external_api_used: bool
    fail_closed_reason: str


@dataclass(frozen=True)
class AIPayloadPolicy:
    """Validate that only redacted text/layout summaries could ever be eligible."""

    fail_real_providers_in_15b: bool = True

    def evaluate(
        self,
        *,
        payload_type: str,
        privacy_gate_result: Any | None,
        operator_approval_state: Any,
        budget_result: Any | None,
        provider_mode: str = "disabled",
        provider_name: str = "disabled",
    ) -> AIPayloadPolicyResult:
        approval = str(getattr(operator_approval_state, "state", operator_approval_state) or "missing")
        reason = ""
        if not privacy_gate_result:
            reason = "missing_privacy_gate_result"
        elif payload_type in FORBIDDEN_PAYLOAD_TYPES:
            reason = "forbidden_payload_type"
        elif payload_type != ALLOWED_FUTURE_PAYLOAD_TYPE:
            reason = "unknown_payload_type"
        elif not bool(getattr(privacy_gate_result, "redaction_complete", False)):
            reason = "missing_redaction_status"
        elif bool(getattr(privacy_gate_result, "suspected_unredacted_pii", True)):
            reason = "suspected_unredacted_pii"
        elif provider_mode != "external_candidate":
            reason = "provider_mode_not_external_candidate"
        elif self.fail_real_providers_in_15b and provider_name.strip().lower() in REAL_PROVIDER_NAMES:
            reason = "real_provider_disabled_in_15b"
        elif approval != "approved":
            reason = "operator_approval_required"
        elif not budget_result:
            reason = "missing_budget_guard_result"
        elif not bool(getattr(budget_result, "budget_allowed", False)):
            reason = str(getattr(budget_result, "budget_fail_reason", "") or "budget_exceeded")

        return AIPayloadPolicyResult(
            payload_type=payload_type,
            provider_mode=provider_mode,
            provider_name=provider_name,
            operator_approval_state=approval,
            payload_policy_allowed=not bool(reason),
            final_external_call_allowed=False,
            external_api_used=False,
            fail_closed_reason=reason or "external_calls_disabled_in_15b",
        )


def payload_policy_to_public_dict(result: AIPayloadPolicyResult) -> dict:
    return asdict(result)


__all__ = [
    "ALLOWED_FUTURE_PAYLOAD_TYPE",
    "FORBIDDEN_PAYLOAD_TYPES",
    "REAL_PROVIDER_NAMES",
    "AIPayloadPolicy",
    "AIPayloadPolicyResult",
    "payload_policy_to_public_dict",
]
