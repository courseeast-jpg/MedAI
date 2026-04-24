"""Deterministic connector routing and fallback orchestration."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable

from execution.connectors.gemini_connector import GeminiConnector
from execution.connectors.phi3_connector import Phi3Connector
from execution.connectors.spacy_connector import SpacyConnector
from execution.metrics import PipelineMetrics


OCR_ARTIFACT_RE = re.compile(r"(\ufffd|[|]{3,}|_{4,}|\b(?:l|I){8,}\b)")
CONNECTOR_ORDER = ("spacy", "phi3", "gemini")


@dataclass
class RoutedExtraction:
    extractor_route: str
    extractor_actual: str
    results: list[dict[str, Any]]
    fallback_used: bool = False
    failure_count: int = 0
    events: list[dict[str, Any]] = field(default_factory=list)
    decision_reason: str = ""
    route_score: float = 0.0


class ExecutionRouter:
    """Small routing layer that orchestrates connectors before consensus."""

    def __init__(
        self,
        *,
        spacy_connector: SpacyConnector,
        gemini_connector_factory: Callable[[str], GeminiConnector],
        phi3_connector: Phi3Connector,
        metrics: PipelineMetrics,
        spacy_fast_path_char_limit: int = 3000,
        preferred_confidence_threshold: float = 0.78,
        max_latency_ms: float = 180.0,
        min_success_rate: float = 0.75,
    ):
        self.spacy_connector = spacy_connector
        self.gemini_connector_factory = gemini_connector_factory
        self.phi3_connector = phi3_connector
        self.metrics = metrics
        self.spacy_fast_path_char_limit = spacy_fast_path_char_limit
        self.preferred_confidence_threshold = preferred_confidence_threshold
        self.max_latency_ms = max_latency_ms
        self.min_success_rate = min_success_rate

    def select_route(self, text: str) -> str:
        return self._select_candidate_routes(text)[0]

    def execute(self, text: str, *, specialty: str = "general") -> RoutedExtraction:
        gemini_connector = self.gemini_connector_factory(specialty)
        connectors = {
            "spacy": self.spacy_connector,
            "phi3": self.phi3_connector,
            "gemini": gemini_connector,
        }
        candidate_routes = self._select_candidate_routes(text)
        route_scores = {
            connector_name: self._score_route(
                connector_name,
                text=text,
                is_primary=(connector_name == candidate_routes[0]),
            )
            for connector_name in CONNECTOR_ORDER
        }
        intended_route = self._choose_connector(candidate_routes, route_scores)
        primary_score = route_scores[intended_route]
        decision_reason = self._decision_reason(intended_route, primary_score, candidate_routes)

        spacy_result = self.spacy_connector.extract(text, specialty=specialty)
        results: list[dict[str, Any]] = []
        if spacy_result.get("entities"):
            results.append(spacy_result)

        events: list[dict[str, Any]] = [{
            "action": "route_selected",
            "reason": decision_reason,
            "route_score": primary_score,
        }]
        failure_count = 0
        attempted = {intended_route}
        current_connector = intended_route
        pending_fallback_notes: list[str] = []

        while True:
            try:
                result = self._load_connector_result(
                    connector_name=current_connector,
                    connectors=connectors,
                    spacy_result=spacy_result,
                    existing_results=results,
                    text=text,
                    specialty=specialty,
                )
                if pending_fallback_notes:
                    result["notes"] = list(result.get("notes", [])) + pending_fallback_notes
                    pending_fallback_notes = []
                actual_extractor = str(result.get("actual_extractor", result.get("extractor", current_connector)))
                if actual_extractor != current_connector:
                    self._append_result(results, result)
                    events.append({
                        "action": "route_actual_mismatch",
                        "reason": f"intended={current_connector} actual={actual_extractor}",
                    })
                    return RoutedExtraction(
                        extractor_route=intended_route,
                        extractor_actual=actual_extractor,
                        results=results,
                        fallback_used=len(attempted) > 1,
                        failure_count=failure_count,
                        events=events,
                        decision_reason=decision_reason,
                        route_score=primary_score,
                    )
                degradation_reason = self._degradation_reason(result, current_connector)
                if (
                    current_connector == "spacy"
                    and intended_route == "spacy"
                    and degradation_reason is not None
                    and degradation_reason.startswith("confidence_too_low")
                ):
                    degradation_reason = None
                if degradation_reason is None:
                    self._append_result(results, result)
                    return RoutedExtraction(
                        extractor_route=intended_route,
                        extractor_actual=actual_extractor,
                        results=results,
                        fallback_used=len(attempted) > 1,
                        failure_count=failure_count,
                        events=events,
                        decision_reason=decision_reason,
                        route_score=primary_score,
                    )
                next_connector = self._next_fallback(current_connector, attempted, route_scores)
                if next_connector is None:
                    self._append_result(results, result)
                    return RoutedExtraction(
                        extractor_route=intended_route,
                        extractor_actual=actual_extractor,
                        results=results,
                        fallback_used=len(attempted) > 1,
                        failure_count=failure_count,
                        events=events + [{
                            "action": "degradation_tolerated",
                            "reason": degradation_reason,
                            "connector": current_connector,
                        }],
                        decision_reason=decision_reason,
                        route_score=primary_score,
                    )
                result["notes"] = list(result.get("notes", [])) + [f"router_degraded={degradation_reason}"]
                self._append_result(results, result)
                events.append({
                    "action": "fallback_invoked",
                    "reason": degradation_reason,
                    "from": current_connector,
                    "to": next_connector,
                    "error_code": "degraded",
                    "configured_gemini_fallback": current_connector == "gemini" and gemini_connector.is_configured,
                })
                attempted.add(next_connector)
                current_connector = next_connector
                continue
            except TimeoutError as exc:
                failure_code = "timeout"
                failure_message = str(exc) or "connector_timeout"
            except Exception as exc:  # noqa: BLE001 - deterministic classification wrapper
                failure_code = self._classify_error(exc)
                failure_message = str(exc) or "connector_failure"

            failure_count += 1
            self.metrics.record_connector_result(
                connector=current_connector,
                latency_ms=0.0,
                confidence=0.0,
                success=False,
            )
            next_connector = self._next_fallback(current_connector, attempted, route_scores)
            if next_connector is None:
                raise RuntimeError(f"{current_connector} failed with no fallback available: {failure_message}") from None
            events.append({
                "action": "fallback_invoked",
                "reason": failure_message,
                "from": current_connector,
                "to": next_connector,
                "error_code": failure_code,
                "configured_gemini_fallback": current_connector == "gemini" and gemini_connector.is_configured,
            })
            pending_fallback_notes = [f"router_fallback={current_connector}:{failure_code}"]
            if current_connector == "gemini" and gemini_connector.is_configured:
                pending_fallback_notes.append("gemini_configured_fallback=true")
            attempted.add(next_connector)
            current_connector = next_connector

    def _select_candidate_routes(self, text: str) -> list[str]:
        if len(text) < self.spacy_fast_path_char_limit and not self._has_ocr_artifacts(text):
            return ["spacy", "phi3", "gemini"]
        return ["gemini", "phi3", "spacy"]

    def _choose_connector(self, candidate_routes: list[str], route_scores: dict[str, float]) -> str:
        eligible = []
        for connector_name in candidate_routes:
            profile = self._effective_profile(connector_name, is_primary=(connector_name == candidate_routes[0]))
            if (
                profile["confidence"] >= self.preferred_confidence_threshold
                and profile["latency_ms"] <= self.max_latency_ms
                and profile["success_rate"] >= self.min_success_rate
            ):
                eligible.append(connector_name)
        if eligible:
            return min(
                eligible,
                key=lambda name: (
                    self.metrics.connector_profile(name)["cost_estimate"],
                    -route_scores[name],
                    candidate_routes.index(name),
                ),
            )
        return max(candidate_routes, key=lambda name: (route_scores[name], -candidate_routes.index(name)))

    def _score_route(self, connector_name: str, *, text: str, is_primary: bool) -> float:
        profile = self._effective_profile(connector_name, is_primary=is_primary)
        confidence_score = min(profile["confidence"], 1.0)
        reliability_score = min(profile["success_rate"], 1.0)
        latency_score = max(0.0, 1.0 - (profile["latency_ms"] / max(self.max_latency_ms * 2, 1.0)))
        cost_score = max(0.0, 1.0 - (profile["cost_estimate"] / 0.03))
        base_bonus = 0.05 if is_primary else 0.0
        complexity_bonus = 0.05 if connector_name == "gemini" and len(text) >= self.spacy_fast_path_char_limit else 0.0
        return round(
            (confidence_score * 0.4)
            + (reliability_score * 0.25)
            + (latency_score * 0.15)
            + (cost_score * 0.15)
            + base_bonus
            + complexity_bonus,
            3,
        )

    def _decision_reason(self, connector_name: str, route_score: float, candidate_routes: list[str]) -> str:
        profile = self._effective_profile(connector_name, is_primary=(connector_name == candidate_routes[0]))
        return (
            f"selected={connector_name} score={route_score:.3f} "
            f"cost={profile['cost_estimate']:.5f} "
            f"confidence={profile['confidence']:.3f} "
            f"latency_ms={profile['latency_ms']:.3f} "
            f"success_rate={profile['success_rate']:.3f} "
            f"candidates={','.join(candidate_routes)}"
        )

    def _degradation_reason(self, result: dict[str, Any], connector_name: str) -> str | None:
        confidence = float(result.get("confidence", 0.0))
        latency_ms = float(result.get("latency_ms", 0.0))
        profile = self.metrics.connector_profile(connector_name)
        if confidence < self.preferred_confidence_threshold:
            return f"confidence_too_low:{confidence:.3f}"
        if latency_ms > self.max_latency_ms:
            return f"latency_too_high:{latency_ms:.3f}"
        if profile["success_rate"] < self.min_success_rate:
            return f"reliability_too_low:{profile['success_rate']:.3f}"
        return None

    def _next_fallback(self, current: str, attempted: set[str], route_scores: dict[str, float]) -> str | None:
        remaining = [name for name in CONNECTOR_ORDER if name not in attempted and name != current]
        if not remaining:
            return None
        return max(remaining, key=lambda name: route_scores[name])

    def _effective_profile(self, connector_name: str, *, is_primary: bool) -> dict[str, float]:
        profile = self.metrics.connector_profile(connector_name)
        adjusted_confidence = float(profile["avg_confidence"])
        if connector_name == "spacy" and not is_primary:
            adjusted_confidence -= 0.18
        elif connector_name == "phi3" and is_primary:
            adjusted_confidence += 0.04
        elif connector_name == "gemini" and is_primary:
            adjusted_confidence += 0.03
        return {
            "confidence": max(0.0, min(adjusted_confidence, 1.0)),
            "latency_ms": float(profile["avg_latency_ms"]),
            "success_rate": float(profile["success_rate"]),
            "cost_estimate": float(profile["cost_estimate"]),
        }

    def _load_connector_result(
        self,
        *,
        connector_name: str,
        connectors: dict[str, Any],
        spacy_result: dict[str, Any],
        existing_results: list[dict[str, Any]],
        text: str,
        specialty: str,
    ) -> dict[str, Any]:
        if connector_name == "spacy":
            existing_spacy = next(
                (
                    item for item in existing_results
                    if str(item.get("actual_extractor", item.get("extractor", ""))) == "spacy"
                ),
                None,
            )
            if existing_spacy is not None:
                return existing_spacy
            return spacy_result
        return connectors[connector_name].extract(text, specialty=specialty)

    def _append_result(self, results: list[dict[str, Any]], result: dict[str, Any]) -> None:
        if result not in results:
            results.append(result)

    def _has_ocr_artifacts(self, text: str) -> bool:
        if OCR_ARTIFACT_RE.search(text):
            return True
        if not text:
            return False
        non_alnum = sum(1 for char in text if not char.isalnum() and not char.isspace())
        return (non_alnum / max(len(text), 1)) > 0.18

    def _classify_error(self, exc: Exception) -> str:
        if "timeout" in str(exc).lower():
            return "timeout"
        return "connector_error"
