"""Local deterministic budget guard for future external AI extraction calls."""
from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class AIBudgetGuardResult:
    provider_name: str
    model_name: str
    estimated_input_tokens: int
    estimated_output_tokens: int
    estimated_cost_usd: float
    session_budget_cap_usd: float
    monthly_budget_cap_usd: float
    budget_allowed: bool
    budget_fail_reason: str


@dataclass(frozen=True)
class AIBudgetGuard:
    provider_name: str = "disabled"
    model_name: str = "disabled"
    session_budget_cap_usd: float = 1.00
    monthly_budget_cap_usd: float = 10.00
    cost_per_1k_tokens_usd: float = 0.002
    default_output_tokens: int = 512

    def evaluate(
        self,
        *,
        estimated_input_tokens: int,
        estimated_output_tokens: int | None = None,
    ) -> AIBudgetGuardResult:
        output_tokens = int(estimated_output_tokens if estimated_output_tokens is not None else self.default_output_tokens)
        input_tokens = max(0, int(estimated_input_tokens))
        total_tokens = input_tokens + max(0, output_tokens)
        estimated_cost = round((total_tokens / 1000.0) * self.cost_per_1k_tokens_usd, 6)
        fail_reason = ""
        if estimated_cost > self.session_budget_cap_usd:
            fail_reason = "session_budget_exceeded"
        elif estimated_cost > self.monthly_budget_cap_usd:
            fail_reason = "monthly_budget_exceeded"
        return AIBudgetGuardResult(
            provider_name=self.provider_name,
            model_name=self.model_name,
            estimated_input_tokens=input_tokens,
            estimated_output_tokens=max(0, output_tokens),
            estimated_cost_usd=estimated_cost,
            session_budget_cap_usd=float(self.session_budget_cap_usd),
            monthly_budget_cap_usd=float(self.monthly_budget_cap_usd),
            budget_allowed=not bool(fail_reason),
            budget_fail_reason=fail_reason,
        )


def budget_guard_to_public_dict(result: AIBudgetGuardResult) -> dict:
    return asdict(result)


__all__ = ["AIBudgetGuard", "AIBudgetGuardResult", "budget_guard_to_public_dict"]
