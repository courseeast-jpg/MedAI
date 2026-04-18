"""
MedAI v1.1 — Specialty Plugin Registry
Auto-discovers all specialty plugins from config.yaml files.
Adding a new specialty = create directory + config.yaml + plugin.py.
Zero changes to core pipeline required.
"""
from pathlib import Path
from typing import Optional
import yaml
from loguru import logger

from app.config import SPECIALTIES_DIR


class SpecialtyPlugin:
    """Base class for all specialty plugins."""

    def __init__(self, config: dict):
        self.specialty = config["specialty"]
        self.display_name = config["display_name"]
        self.config = config
        self.keywords = config.get("keywords", [])
        self.task_types = config.get("task_types", [])
        self.curated_sources = config.get("curated_sources", [])
        self.is_stub = config.get("status") == "stub"

    def matches_query(self, query: str) -> float:
        """Returns match score 0.0–1.0 for this specialty given a query."""
        query_lower = query.lower()
        matches = sum(1 for kw in self.keywords if kw in query_lower)
        if not self.keywords:
            return 0.0
        return min(1.0, matches / max(1, len(self.keywords) * 0.3))

    def get_primary_connector(self) -> str:
        return self.config.get("external_apis", {}).get("primary", "dxgpt")

    def get_extraction_prompt(self) -> Optional[str]:
        return self.config.get("extraction_prompt")

    def get_curated_sources(self) -> list:
        return self.curated_sources


class SpecialtyRegistry:
    """Loads and manages all specialty plugins."""

    def __init__(self, specialties_dir: Path = SPECIALTIES_DIR):
        self.plugins: dict[str, SpecialtyPlugin] = {}
        self._load_all(specialties_dir)

    def _load_all(self, base_dir: Path):
        if not base_dir.exists():
            logger.warning(f"Specialties directory not found: {base_dir}")
            return

        for config_path in base_dir.glob("*/config.yaml"):
            try:
                with open(config_path) as f:
                    config = yaml.safe_load(f)
                plugin = SpecialtyPlugin(config)
                self.plugins[plugin.specialty] = plugin
                status = "STUB" if plugin.is_stub else "active"
                logger.info(f"Specialty loaded: {plugin.display_name} [{status}]")
            except Exception as e:
                logger.error(f"Failed to load specialty from {config_path}: {e}")

        logger.info(f"Registry: {len(self.plugins)} specialties loaded")

    def detect_specialty(self, query: str) -> tuple[str, float]:
        """Returns (specialty_name, confidence) for a query."""
        scores = {
            name: plugin.matches_query(query)
            for name, plugin in self.plugins.items()
            if not plugin.is_stub
        }
        if not scores or max(scores.values()) == 0.0:
            return "general", 0.4

        best = max(scores, key=scores.get)
        return best, min(0.95, scores[best] + 0.30)

    def get(self, specialty: str) -> Optional[SpecialtyPlugin]:
        return self.plugins.get(specialty)

    def active_specialties(self) -> list[str]:
        return [name for name, p in self.plugins.items() if not p.is_stub]

    def all_specialties(self) -> list[str]:
        return list(self.plugins.keys())
