"""Flash-specific output-contract hardening (17C-R2-R17).

gemini-2.5-flash often returns JSON that the strict full-schema validator rejects
(truncated_or_invalid_json). These small, pure helpers make Flash section output
recoverable WITHOUT weakening the clinical schema or promoting anything to MKB:

  - normalize_section_name : map benign section-name variants to the allowed enum
  - salvage_flash_json     : strip fences, extract first balanced JSON, close a
                             parse-safe truncated tail; reject if unrecoverable
  - minimal_section_object : review-bound minimal section package (empty items)
  - is_minimal_section     : validate a minimal review-bound section object

No provider/network/MKB calls. Output is review-bound only.
"""
from __future__ import annotations

import json
from typing import Any

from execution.sectioned_extraction import SECTION_NAMES

ALLOWED_SECTIONS = tuple(SECTION_NAMES)
_MINIMAL_KEYS = ("section", "items", "needs_review", "warnings")


def normalize_section_name(name: Any, allowed: "tuple[str, ...]" = ALLOWED_SECTIONS) -> "str | None":
    """Return the canonical allowed section name for a benign variant (case, spaces vs
    underscores, exact singular/plural), else None."""
    if not isinstance(name, str) or not name.strip():
        return None
    canon = {s.lower(): s for s in allowed}
    n = name.strip().lower().replace(" ", "_").replace("-", "_")
    while "__" in n:
        n = n.replace("__", "_")
    if n in canon:
        return canon[n]
    for s in allowed:
        sl = s.lower()
        if n == sl or n + "s" == sl or n == sl + "s" or n.rstrip("s") == sl.rstrip("s"):
            return s
    return None


def _extract_first_balanced(s: str) -> "tuple[Any | None, str]":
    """Extract the first balanced JSON object/array, tolerating a parse-safe truncated
    tail (append the missing closers only if the result then parses)."""
    start = None
    for i, ch in enumerate(s):
        if ch in "{[":
            start = i
            break
    if start is None:
        return None, "no_json_start"
    opener = s[start]
    closer = "}" if opener == "{" else "]"
    depth = 0
    in_str = False
    esc = False
    end = None
    stack = []
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
            depth += 1
        elif ch in "}]":
            if depth:
                depth -= 1
                stack.pop()
                if depth == 0:
                    end = i
                    break
    if end is not None:
        frag = s[start:end + 1]
        try:
            return json.loads(frag), "balanced"
        except ValueError:
            return None, "balanced_but_invalid"
    # Truncated: close remaining openers if that yields valid JSON.
    if 0 < depth <= 200 and not in_str:
        frag = s[start:] + "".join(reversed(stack))
        try:
            return json.loads(frag), "closed_truncated_tail"
        except ValueError:
            return None, "truncated_not_recoverable"
    return None, "unrecoverable"


def salvage_flash_json(text: str) -> "tuple[Any | None, str]":
    """Strip markdown fences, then parse / extract first balanced JSON. Returns
    (obj, reason) or (None, reason)."""
    s = (text or "").strip()
    if s == "":
        return None, "empty_response"
    if s.startswith("```"):
        lines = s.split("\n")
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        s = "\n".join(lines).strip()
    try:
        return json.loads(s), "direct"
    except ValueError:
        pass
    return _extract_first_balanced(s)


def minimal_section_object(section: str) -> dict[str, Any]:
    """Review-bound minimal section package (no clinical items; flagged for review)."""
    return {"section": section, "items": [], "needs_review": True, "warnings": []}


def is_minimal_section(obj: Any, section: str) -> bool:
    """True if obj is a valid minimal/review-bound section object for `section`."""
    if not isinstance(obj, dict):
        return False
    if normalize_section_name(obj.get("section"), ALLOWED_SECTIONS) != section:
        return False
    if not isinstance(obj.get("items"), list):
        return False
    if not isinstance(obj.get("warnings", []), list):
        return False
    return "needs_review" in obj


__all__ = [
    "ALLOWED_SECTIONS", "normalize_section_name", "salvage_flash_json",
    "minimal_section_object", "is_minimal_section",
]
