"""Phase 44 — Local OCR capability detection.

Detects whether the host environment provides a working Tesseract binary
and which trained-data languages are installed. No package installs, no
network calls. Failures degrade gracefully into a structured "unavailable"
report so downstream code can fall back safely.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from typing import Any


_DEFAULT_WINDOWS_PATHS = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
)


@dataclass(frozen=True)
class OcrCapabilityReport:
    tesseract_available: bool
    tesseract_path: str | None
    available_languages: list[str]
    russian_available: bool
    english_available: bool
    warnings: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def detect_tesseract_available(*, override_path: str | None = None) -> str | None:
    """Return the absolute path to a working tesseract binary, or None.

    Resolution order:
      1. ``override_path`` argument (caller-supplied)
      2. ``TESSERACT_CMD`` environment variable
      3. ``shutil.which("tesseract")``
      4. Common Windows install locations
    """
    candidates: list[str] = []
    if override_path:
        candidates.append(override_path)
    env_path = os.environ.get("TESSERACT_CMD")
    if env_path:
        candidates.append(env_path)
    on_path = shutil.which("tesseract")
    if on_path:
        candidates.append(on_path)
    if os.name == "nt":
        candidates.extend(_DEFAULT_WINDOWS_PATHS)

    for candidate in candidates:
        if not candidate:
            continue
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
        # On Windows, os.access may report False even for usable .exe files;
        # fall back to a simple existence check.
        if os.name == "nt" and os.path.isfile(candidate):
            return candidate
    return None


def detect_tesseract_languages(
    *,
    binary_path: str | None = None,
    timeout_seconds: float = 10.0,
) -> tuple[list[str], list[str]]:
    """Return (languages, warnings). Empty list + warnings if detection fails."""
    path = binary_path or detect_tesseract_available()
    if not path:
        return [], ["tesseract_binary_not_found"]
    try:
        result = subprocess.run(
            [path, "--list-langs"],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except (FileNotFoundError, OSError) as exc:
        return [], [f"tesseract_invocation_error:{exc.__class__.__name__}"]
    except subprocess.TimeoutExpired:
        return [], ["tesseract_list_langs_timeout"]

    output = (result.stdout or "") + "\n" + (result.stderr or "")
    languages: list[str] = []
    for raw in output.splitlines():
        line = raw.strip()
        if not line or line.startswith("List of available languages"):
            continue
        if "/" in line or "\\" in line:
            # script directory entries like "script\Cyrillic" — keep as-is
            languages.append(line)
            continue
        # Real lang codes are short ASCII tokens; filter banner noise
        if all(c.isalnum() or c == "_" for c in line) and 2 <= len(line) <= 16:
            languages.append(line)
    warnings: list[str] = []
    if not languages and result.returncode != 0:
        warnings.append(f"tesseract_list_langs_returncode_{result.returncode}")
    return languages, warnings


def has_russian_ocr_support(languages: list[str] | None = None) -> bool:
    langs = languages if languages is not None else detect_tesseract_languages()[0]
    return "rus" in {lang.lower() for lang in langs}


def get_ocr_capability_report(*, override_path: str | None = None) -> OcrCapabilityReport:
    path = detect_tesseract_available(override_path=override_path)
    if not path:
        return OcrCapabilityReport(
            tesseract_available=False,
            tesseract_path=None,
            available_languages=[],
            russian_available=False,
            english_available=False,
            warnings=["tesseract_binary_not_found", "cyrillic_ocr_unavailable"],
            metadata={"resolution_attempts": _resolution_summary(override_path)},
        )

    languages, warnings = detect_tesseract_languages(binary_path=path)
    russian = "rus" in {lang.lower() for lang in languages}
    english = "eng" in {lang.lower() for lang in languages}
    enriched_warnings = list(warnings)
    if not russian:
        enriched_warnings.append("cyrillic_ocr_unavailable")
    if not english:
        enriched_warnings.append("english_ocr_unavailable")

    return OcrCapabilityReport(
        tesseract_available=True,
        tesseract_path=path,
        available_languages=languages,
        russian_available=russian,
        english_available=english,
        warnings=enriched_warnings,
        metadata={"resolution_attempts": _resolution_summary(override_path)},
    )


def _resolution_summary(override_path: str | None) -> dict[str, Any]:
    return {
        "override_path_provided": bool(override_path),
        "env_TESSERACT_CMD_set": "TESSERACT_CMD" in os.environ,
        "platform": os.name,
    }
