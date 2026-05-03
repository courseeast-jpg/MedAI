"""Local image OCR helper for Phase56 blind-audit inputs."""

from __future__ import annotations

import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ocr_layout.ocr_capabilities import OcrCapabilityReport, get_ocr_capability_report


SUPPORTED_IMAGE_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".webp"}
_FRAME_LIMIT_DEFAULT = 8
_TESSERACT_TIMEOUT_SECONDS = 60.0


@dataclass(frozen=True)
class ImageOcrResult:
    text: str
    frame_count: int
    ocr_engine: str
    language_hint: str
    warnings: list[str]
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def page_count(self) -> int:
        return self.frame_count

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["page_count"] = self.page_count
        return payload


def extract_image_text(
    source_path: Path | str,
    *,
    capability: OcrCapabilityReport | None = None,
    frame_limit: int = _FRAME_LIMIT_DEFAULT,
) -> ImageOcrResult:
    """OCR an image file locally with Tesseract.

    The function never raises for caller-visible file/OCR failures. It returns
    structured warnings and empty text so callers can route safely to review.
    """
    path = Path(source_path)
    cap = capability if capability is not None else get_ocr_capability_report()
    warnings: list[str] = []
    metadata: dict[str, Any] = {
        "source": "phase56_image_ocr",
        "frame_limit": frame_limit,
        "tesseract_available": cap.tesseract_available,
        "russian_available": cap.russian_available,
        "english_available": cap.english_available,
    }

    if not path.exists():
        return _failed("image_ocr_source_not_found", metadata=metadata)
    if path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
        return _failed("image_ocr_unsupported_extension", metadata=metadata)
    if not cap.tesseract_available:
        return _failed("tesseract_binary_not_found", metadata=metadata, warnings=["image_ocr_unavailable"])

    language_hint = _language_hint(cap)
    metadata["language_hint"] = language_hint
    metadata["ocr_attempted"] = True

    try:
        from PIL import Image, ImageOps, ImageSequence
    except Exception as exc:  # noqa: BLE001
        return _failed("pillow_unavailable", metadata=metadata, error=f"{exc.__class__.__name__}:{exc}")

    try:
        image = Image.open(path)
    except Exception as exc:  # noqa: BLE001
        return _failed("image_open_failed", metadata=metadata, error=f"{exc.__class__.__name__}:{exc}")

    frame_texts: list[str] = []
    per_frame: list[dict[str, Any]] = []
    frames_seen = 0
    try:
        with tempfile.TemporaryDirectory(prefix="phase56_image_ocr_") as tmp:
            tmp_dir = Path(tmp)
            for index, frame in enumerate(ImageSequence.Iterator(image), start=1):
                frames_seen = index
                if index > frame_limit:
                    warnings.append("image_ocr_frame_limit_truncated")
                    break
                frame_audit: dict[str, Any] = {"frame": index}
                try:
                    normalized = ImageOps.exif_transpose(frame)
                    if normalized.mode not in {"RGB", "L"}:
                        normalized = normalized.convert("RGB")
                    image_path = tmp_dir / f"frame_{index:03d}.png"
                    normalized.save(image_path)
                except Exception as exc:  # noqa: BLE001
                    frame_audit["normalize_error"] = f"{exc.__class__.__name__}:{exc}"
                    per_frame.append(frame_audit)
                    _append_unique(warnings, "image_frame_normalize_failed")
                    continue

                text, frame_warnings, frame_run = _run_tesseract_on_image(
                    binary_path=cap.tesseract_path or "tesseract",
                    image_path=image_path,
                    languages=language_hint,
                )
                frame_audit.update(frame_run)
                for warning in frame_warnings:
                    _append_unique(warnings, warning)
                if text:
                    frame_texts.append(f"[Frame {index}]\n{text}")
                per_frame.append(frame_audit)
    finally:
        try:
            image.close()
        except Exception:  # noqa: BLE001
            pass

    text = "\n".join(frame_texts).strip()
    if not text:
        _append_unique(warnings, "image_ocr_empty_output")
    metadata["frames_seen"] = frames_seen
    metadata["frames_processed"] = len(per_frame)
    metadata["per_frame"] = per_frame
    return ImageOcrResult(
        text=text,
        frame_count=frames_seen,
        ocr_engine="tesseract",
        language_hint=language_hint,
        warnings=warnings,
        error=None,
        metadata=metadata,
    )


def _language_hint(capability: OcrCapabilityReport) -> str:
    if capability.russian_available:
        return "rus+eng" if capability.english_available else "rus"
    return "eng"


def _run_tesseract_on_image(*, binary_path: str, image_path: Path, languages: str) -> tuple[str, list[str], dict[str, Any]]:
    try:
        result = subprocess.run(
            [binary_path, str(image_path), "stdout", "-l", languages],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=_TESSERACT_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return "", ["image_ocr_timeout"], {"timeout": True}
    except (FileNotFoundError, OSError) as exc:
        return "", ["image_ocr_invocation_error"], {"error": exc.__class__.__name__}

    warnings: list[str] = []
    if result.returncode != 0:
        warnings.append(f"tesseract_returncode_{result.returncode}")
    return (result.stdout or "").strip(), warnings, {"returncode": result.returncode, "output_chars": len(result.stdout or "")}


def _failed(
    warning: str,
    *,
    metadata: dict[str, Any],
    warnings: list[str] | None = None,
    error: str | None = None,
) -> ImageOcrResult:
    all_warnings = list(warnings or [])
    _append_unique(all_warnings, warning)
    return ImageOcrResult(
        text="",
        frame_count=0,
        ocr_engine="tesseract",
        language_hint="unknown",
        warnings=all_warnings,
        error=error or warning,
        metadata=metadata,
    )


def _append_unique(values: list[str], value: str) -> None:
    if value not in values:
        values.append(value)
