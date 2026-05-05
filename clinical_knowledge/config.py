"""CKA feature flags and configuration (CKA-B01).

All flags default to safe/off. No environment secrets required.
External APIs disabled by default.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class CKAConfig:
    # Graph and LLM features — off by default
    ENABLE_GRAPH: bool = False
    ENABLE_LOCAL_LLM: bool = False
    ENRICH_PROMOTE: bool = False

    # Connector whitelist — only stub connectors active
    ACTIVE_CONNECTORS: List[str] = field(default_factory=lambda: ["dxgpt_stub"])

    # Ingestion features — off by default
    ENABLE_WEB_INGESTION: bool = False
    ENABLE_EPUB: bool = False
    ENABLE_YOUTUBE: bool = False

    # Safety thresholds
    SAFE_MODE_THRESHOLD: float = 0.4

    # Privacy / API boundaries
    MEDAI_LOCAL_ONLY: bool = True
    EXTERNAL_APIS_ENABLED: bool = False

    def __post_init__(self) -> None:
        # Respect runtime env overrides (string → bool)
        if os.environ.get("MEDAI_LOCAL_ONLY", "").lower() == "false":
            self.MEDAI_LOCAL_ONLY = False
        if os.environ.get("EXTERNAL_APIS_ENABLED", "").lower() == "true":
            self.EXTERNAL_APIS_ENABLED = True

    def as_dict(self) -> dict:
        return {
            "ENABLE_GRAPH": self.ENABLE_GRAPH,
            "ENABLE_LOCAL_LLM": self.ENABLE_LOCAL_LLM,
            "ENRICH_PROMOTE": self.ENRICH_PROMOTE,
            "ACTIVE_CONNECTORS": list(self.ACTIVE_CONNECTORS),
            "ENABLE_WEB_INGESTION": self.ENABLE_WEB_INGESTION,
            "ENABLE_EPUB": self.ENABLE_EPUB,
            "ENABLE_YOUTUBE": self.ENABLE_YOUTUBE,
            "SAFE_MODE_THRESHOLD": self.SAFE_MODE_THRESHOLD,
            "MEDAI_LOCAL_ONLY": self.MEDAI_LOCAL_ONLY,
            "EXTERNAL_APIS_ENABLED": self.EXTERNAL_APIS_ENABLED,
        }


# Module-level default instance
DEFAULT_CONFIG = CKAConfig()
