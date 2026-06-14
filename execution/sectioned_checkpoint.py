"""Private section-aware checkpoint for 17C-R2-R13.

All files are outside the repository. Public reports should only reference
counts, doc hashes, section names, and redacted labels.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

SECTION_CHECKPOINT_DIR = Path(os.path.expandvars(
    r"%LOCALAPPDATA%\MedAI_Private\ai_extraction_17C_R2_R13_section_checkpoint"))
SECTION_STATE_FILE = "section_checkpoint_state_private.json"
SECTION_COMPLETED_FILE = "section_checkpoint_completed_private.json"
SECTION_FAILED_FILE = "section_checkpoint_failed_private.json"


def _base(base: str | Path | None = None) -> Path:
    return Path(base) if base is not None else SECTION_CHECKPOINT_DIR


def load_completed_sections(base: str | Path | None = None) -> dict[str, list[str]]:
    path = _base(base) / SECTION_COMPLETED_FILE
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(k): [str(x) for x in v] for k, v in data.items() if isinstance(v, list)}


def mark_section_completed(doc_id: str, section: str, base: str | Path | None = None) -> None:
    b = _base(base)
    b.mkdir(parents=True, exist_ok=True)
    data = load_completed_sections(base)
    sections = data.setdefault(str(doc_id), [])
    if section not in sections:
        sections.append(section)
    (b / SECTION_COMPLETED_FILE).write_text(json.dumps(data, indent=2, ensure_ascii=True), encoding="utf-8")


def mark_section_failed(doc_id: str, section: str, category: str, base: str | Path | None = None) -> None:
    b = _base(base)
    b.mkdir(parents=True, exist_ok=True)
    payload = {"doc_id": str(doc_id), "section": str(section), "failure_category": str(category), "resolved": False}
    (b / SECTION_FAILED_FILE).write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def clear_failed_section(base: str | Path | None = None) -> None:
    try:
        (_base(base) / SECTION_FAILED_FILE).unlink()
    except OSError:
        pass


def write_state(payload: dict[str, Any], base: str | Path | None = None) -> None:
    b = _base(base)
    b.mkdir(parents=True, exist_ok=True)
    (b / SECTION_STATE_FILE).write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


__all__ = [
    "SECTION_CHECKPOINT_DIR",
    "load_completed_sections",
    "mark_section_completed",
    "mark_section_failed",
    "clear_failed_section",
    "write_state",
]
