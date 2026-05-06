"""CKA-B10: System scaffold — wires all CKA B01-B09 components for runtime.

CKASystemScaffold is NOT a production-autonomous system.
It is a research/operator scaffold with all safety constraints enforced:
- safe_mode=True by default
- allow_active_write=False, invariant — raises ValueError if True attempted
- EXTERNAL_APIS_ENABLED=False enforced — raises ValueError if True in config
- No clinical advice, no diagnosis, no medication orders issued
- HITL release freeze remains closed
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from clinical_knowledge.config import CKAConfig
from clinical_knowledge.connectors.registry import ConnectorRegistry
from clinical_knowledge.preflight import CKAPreflightReport, run_cka_preflight
from clinical_knowledge.store import MKBStore

_SCAFFOLD_VERSION = "CKA-B10"
_PRODUCTION_AUTONOMOUS = False   # Never True in CKA architecture


@dataclass
class CKASystemScaffold:
    """Runtime scaffold wiring all CKA B01-B09 components.

    Build via CKASystemScaffold.build() — do not instantiate directly
    unless you supply a pre-validated store, registry, and config.

    Invariants:
    - allow_active_write is always False
    - External APIs are always blocked
    - production_autonomous is always False
    """

    store: MKBStore
    registry: ConnectorRegistry
    config: CKAConfig
    safe_mode: bool = True
    allow_active_write: bool = False
    _last_preflight: Optional[CKAPreflightReport] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.allow_active_write:
            raise ValueError(
                "CKASystemScaffold: allow_active_write=True is not permitted. "
                "All writes require HITL operator review."
            )
        if self.config.EXTERNAL_APIS_ENABLED:
            raise ValueError(
                "CKASystemScaffold: EXTERNAL_APIS_ENABLED=True is not permitted. "
                "All connectors must be local stubs only."
            )

    @classmethod
    def build(
        cls,
        safe_mode: bool = True,
        db_path: str = ":memory:",
    ) -> "CKASystemScaffold":
        """Build a scaffold with safe defaults.

        - In-memory SQLite store by default (no disk persistence)
        - Default connector registry (3 enabled stubs, 1 disabled)
        - Default CKAConfig (external APIs off, graph off, LLM off)
        """
        store = MKBStore(db_path)
        registry = ConnectorRegistry.default()
        config = CKAConfig()
        return cls(
            store=store,
            registry=registry,
            config=config,
            safe_mode=safe_mode,
            allow_active_write=False,
        )

    # ------------------------------------------------------------------
    # Preflight
    # ------------------------------------------------------------------

    def preflight(self) -> CKAPreflightReport:
        """Run all CKA B01-B09 preflight checks. Result is cached."""
        report = run_cka_preflight(safe_mode=self.safe_mode)
        self._last_preflight = report
        return report

    def is_ready(self) -> bool:
        """True if preflight passes (no FAIL checks). Runs preflight if not cached."""
        if self._last_preflight is None:
            self._last_preflight = self.preflight()
        return self._last_preflight.passed

    def reset_preflight(self) -> None:
        """Clear cached preflight so next is_ready() call re-runs all checks."""
        self._last_preflight = None

    # ------------------------------------------------------------------
    # Public summary
    # ------------------------------------------------------------------

    def safe_public_summary(self) -> dict:
        """Return public scaffold state — no private data, no PHI, no secrets."""
        return {
            "scaffold_version": _SCAFFOLD_VERSION,
            "safe_mode": self.safe_mode,
            "allow_active_write": self.allow_active_write,
            "production_autonomous": _PRODUCTION_AUTONOMOUS,
            "external_api_used": False,
            "store_initialized": self.store is not None,
            "registry_total_connectors": len(self.registry.list_all()),
            "registry_enabled_connectors": len(self.registry.list_enabled()),
            "config_external_apis_enabled": self.config.EXTERNAL_APIS_ENABLED,
            "config_enrich_promote": self.config.ENRICH_PROMOTE,
            "preflight_run": self._last_preflight is not None,
            "preflight_passed": (
                self._last_preflight.passed if self._last_preflight is not None else None
            ),
        }
