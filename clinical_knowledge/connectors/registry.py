"""Connector registry for CKA-B08.

ConnectorRegistry:
- register/list/get/disable connector specs
- rejects non-synthetic external connector specs
- default registry includes 4 stubs (generic_stub disabled)
"""
from __future__ import annotations

from typing import Dict, List, Optional

from clinical_knowledge.connectors.models import (
    ConnectorCapability,
    ConnectorKind,
    ConnectorSpec,
)


class ConnectorRegistryError(ValueError):
    pass


class ConnectorRegistry:
    """Registry of connector specs. Enforces B08 safety rules."""

    def __init__(self) -> None:
        self._specs: Dict[str, ConnectorSpec] = {}

    # ------------------------------------------------------------------
    def register(self, spec: ConnectorSpec) -> None:
        """Register a connector spec. Rejects unsafe external specs."""
        self._validate_spec(spec)
        self._specs[spec.name] = spec

    def _validate_spec(self, spec: ConnectorSpec) -> None:
        if spec.allow_external:
            raise ConnectorRegistryError(
                f"Connector '{spec.name}': allow_external=True is not permitted in CKA-B08. "
                "All connectors must be local stubs."
            )
        if not spec.synthetic_only:
            raise ConnectorRegistryError(
                f"Connector '{spec.name}': synthetic_only=False is not permitted in CKA-B08. "
                "All connectors must use synthetic data only."
            )

    def get(self, name: str) -> Optional[ConnectorSpec]:
        return self._specs.get(name)

    def list_enabled(self) -> List[ConnectorSpec]:
        return [s for s in self._specs.values() if s.enabled]

    def list_all(self) -> List[ConnectorSpec]:
        return list(self._specs.values())

    def disable(self, name: str) -> None:
        spec = self._specs.get(name)
        if spec:
            # Rebuild with enabled=False (dataclass is mutable)
            spec.enabled = False

    def enable(self, name: str) -> None:
        spec = self._specs.get(name)
        if spec:
            spec.enabled = True

    # ------------------------------------------------------------------
    @classmethod
    def default(cls) -> "ConnectorRegistry":
        """Build the default B08 registry with 4 stub connectors."""
        reg = cls()
        reg.register(ConnectorSpec(
            name="dxgpt_stub",
            kind=ConnectorKind.DXGPT_STUB,
            enabled=True,
            capabilities=[
                ConnectorCapability.DIAGNOSIS_SUPPORT,
                ConnectorCapability.CITATION_SUPPORT,
                ConnectorCapability.STRUCTURED_FACT_OUTPUT,
            ],
            timeout_seconds=5.0,
            allow_external=False,
            synthetic_only=True,
        ))
        reg.register(ConnectorSpec(
            name="sage_epilepsy_stub",
            kind=ConnectorKind.SAGE_EPILEPSY_STUB,
            enabled=True,
            capabilities=[
                ConnectorCapability.EPILEPSY_SUPPORT,
                ConnectorCapability.STRUCTURED_FACT_OUTPUT,
            ],
            timeout_seconds=5.0,
            allow_external=False,
            synthetic_only=True,
        ))
        reg.register(ConnectorSpec(
            name="patientnotes_ddi_stub",
            kind=ConnectorKind.PATIENTNOTES_DDI_STUB,
            enabled=True,
            capabilities=[
                ConnectorCapability.MEDICATION_SAFETY,
                ConnectorCapability.STRUCTURED_FACT_OUTPUT,
            ],
            timeout_seconds=5.0,
            allow_external=False,
            synthetic_only=True,
        ))
        reg.register(ConnectorSpec(
            name="generic_stub",
            kind=ConnectorKind.GENERIC_STUB,
            enabled=False,   # disabled by default
            capabilities=[
                ConnectorCapability.STRUCTURED_FACT_OUTPUT,
            ],
            timeout_seconds=5.0,
            allow_external=False,
            synthetic_only=True,
        ))
        return reg
