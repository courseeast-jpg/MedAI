"""CKA-TERM-01G safe performance buckets."""
from __future__ import annotations


def elapsed_seconds_safe_bucket(seconds: float) -> str:
    if seconds < 0.25:
        return "lt_250ms"
    if seconds < 1.0:
        return "lt_1s"
    if seconds < 5.0:
        return "lt_5s"
    if seconds < 30.0:
        return "lt_30s"
    return "gte_30s"
