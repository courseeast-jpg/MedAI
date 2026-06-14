"""Safe strict-JSON response normalizer for AI extraction responses.

A provider may wrap a single JSON object in a Markdown fence (```json ... ```), which is
otherwise valid JSON. This normalizer strips ONE clean surrounding fence and then
requires the remainder to be exactly one JSON object. It deliberately does NOT:
  - extract a JSON substring out of prose-wrapped text,
  - accept multiple JSON objects or trailing text,
  - accept truncated/partial JSON,
  - infer or repair missing content.

It never weakens the schema; callers still apply their full schema + privacy validation
to the returned object.
"""
from __future__ import annotations

import json
from typing import Any


def normalize_one_json_object(text: str) -> tuple[Any | None, str]:
    """Return (obj, "ok") for a single clean JSON object (optionally fenced), else
    (None, reason) where reason is a strict-JSON failure class."""
    s = (text or "").strip()
    if s == "":
        return None, "empty_response"

    # Strip exactly one surrounding Markdown fence if present: a first line starting with
    # ``` (optionally ```json) and a final line that is exactly ```.
    if s.startswith("```"):
        lines = s.split("\n")
        if len(lines) >= 2 and lines[0].startswith("```") and lines[-1].strip() == "```":
            s = "\n".join(lines[1:-1]).strip()
            if s == "":
                return None, "empty_response"
        else:
            return None, "markdown_fence_unterminated"

    # The remainder must be exactly one JSON object — no prose, no extra objects.
    try:
        obj = json.loads(s)
    except json.JSONDecodeError as exc:
        msg = str(exc)
        if "Extra data" in msg:
            return None, "multiple_json_objects_or_trailing_text"
        if not s.startswith("{"):
            return None, "prose_wrapped_or_non_json"
        return None, "truncated_or_invalid_json"
    if not isinstance(obj, dict):
        return None, "not_json_object"
    return obj, "ok"


__all__ = ["normalize_one_json_object"]
