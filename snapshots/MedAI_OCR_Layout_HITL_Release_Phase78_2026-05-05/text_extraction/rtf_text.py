from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


SUPPORTED_RTF_EXTENSIONS = {".rtf"}

_DESTINATION_CONTROLS = {
    "annotation",
    "author",
    "colortbl",
    "comment",
    "datastore",
    "fonttbl",
    "footer",
    "footerf",
    "footerl",
    "footerr",
    "header",
    "headerf",
    "headerl",
    "headerr",
    "info",
    "object",
    "pict",
    "revtbl",
    "stylesheet",
    "themedata",
    "xmlnstbl",
}

_TEXT_BREAK_CONTROLS = {
    "line": "\n",
    "par": "\n",
    "page": "\n",
    "sect": "\n",
    "tab": "\t",
}


@dataclass(frozen=True)
class RtfTextResult:
    text: str
    parser_name: str = "local_rtf_text_parser"
    warnings: list[str] | None = None
    error: str | None = None
    metadata: dict[str, int | str | bool] | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def extract_rtf_text(path: Path) -> RtfTextResult:
    """Extract plain text from an RTF file without external services.

    The parser intentionally handles only local deterministic RTF text
    recovery. It strips common formatting groups and returns text in memory;
    callers are responsible for PHI-safe reporting.
    """
    warnings: list[str] = []
    try:
        raw = path.read_bytes()
    except OSError as exc:
        return RtfTextResult(
            text="",
            warnings=["rtf_read_failed"],
            error=f"rtf_read_failed:{exc.__class__.__name__}",
            metadata={"byte_length": 0, "is_rtf_header": False},
        )

    source = _decode_rtf_bytes(raw, warnings)
    is_rtf_header = source.lstrip().startswith("{\\rtf")
    if not is_rtf_header:
        warnings.append("rtf_header_missing")

    try:
        text = rtf_to_text(source)
    except Exception as exc:  # pragma: no cover - defensive safety net
        return RtfTextResult(
            text="",
            warnings=warnings + ["rtf_parse_failed"],
            error=f"rtf_parse_failed:{exc.__class__.__name__}",
            metadata={"byte_length": len(raw), "is_rtf_header": is_rtf_header},
        )

    text = _normalize_text(text)
    if not text:
        warnings.append("rtf_empty_text")
    return RtfTextResult(
        text=text,
        warnings=warnings,
        error=None,
        metadata={
            "byte_length": len(raw),
            "is_rtf_header": is_rtf_header,
            "text_length": len(text),
        },
    )


def rtf_to_text(source: str) -> str:
    output: list[str] = []
    stack: list[tuple[bool, bool]] = []
    skip_group = False
    group_start = False
    uc_skip = 1
    pending_unicode_skip = 0
    index = 0
    length = len(source)

    while index < length:
        char = source[index]
        if pending_unicode_skip > 0:
            pending_unicode_skip -= 1
            index += 1
            continue
        if char == "{":
            stack.append((skip_group, group_start))
            group_start = True
            index += 1
            continue
        if char == "}":
            if stack:
                skip_group, group_start = stack.pop()
            else:
                skip_group = False
                group_start = False
            index += 1
            continue
        if char == "\\":
            control, parameter, has_parameter, next_index = _read_control(source, index + 1)
            index = next_index
            if control == "*":
                if group_start:
                    skip_group = True
                continue
            if control in _DESTINATION_CONTROLS and group_start:
                skip_group = True
                continue
            group_start = False
            if skip_group:
                continue
            if control == "'":
                if parameter is not None:
                    output.append(chr(parameter))
                continue
            if control == "u" and has_parameter and parameter is not None:
                output.append(chr(parameter if parameter >= 0 else parameter + 65536))
                pending_unicode_skip = max(0, uc_skip)
                continue
            if control == "uc" and has_parameter and parameter is not None:
                uc_skip = max(0, parameter)
                continue
            if control in _TEXT_BREAK_CONTROLS:
                output.append(_TEXT_BREAK_CONTROLS[control])
                continue
            if control in {"~"}:
                output.append(" ")
                continue
            if control in {"-", "_"}:
                output.append("-")
                continue
            continue
        group_start = False
        if not skip_group:
            output.append(char)
        index += 1
    return "".join(output)


def _read_control(source: str, index: int) -> tuple[str, int | None, bool, int]:
    if index >= len(source):
        return "", None, False, index
    char = source[index]
    if char == "'":
        hex_text = source[index + 1 : index + 3]
        try:
            return "'", int(hex_text, 16), True, min(len(source), index + 3)
        except ValueError:
            return "'", None, False, min(len(source), index + 3)
    if not char.isalpha():
        return char, None, False, index + 1

    start = index
    while index < len(source) and source[index].isalpha():
        index += 1
    control = source[start:index]

    sign = 1
    if index < len(source) and source[index] == "-":
        sign = -1
        index += 1
    number_start = index
    while index < len(source) and source[index].isdigit():
        index += 1
    has_parameter = index > number_start
    parameter = sign * int(source[number_start:index]) if has_parameter else None
    if index < len(source) and source[index] == " ":
        index += 1
    return control, parameter, has_parameter, index


def _decode_rtf_bytes(raw: bytes, warnings: list[str]) -> str:
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    warnings.append("rtf_decode_replacement_used")
    return raw.decode("utf-8", errors="replace")


def _normalize_text(text: str) -> str:
    lines = [" ".join(line.split()) for line in text.replace("\r", "\n").split("\n")]
    collapsed: list[str] = []
    previous_blank = False
    for line in lines:
        if not line:
            if not previous_blank:
                collapsed.append("")
            previous_blank = True
        else:
            collapsed.append(line)
            previous_blank = False
    return "\n".join(collapsed).strip()
