"""CKA-TERM-01A — license-acknowledgment template helpers.

Generates a *template* JSON file with `operator_acknowledged=false` and
empty `acknowledged_systems`. NEVER creates a true acknowledgment
automatically — the operator must edit the template themselves and
rename it to the real `LICENSE_ACK_PRIVATE.json` (which is gitignored
and never staged).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


_TEMPLATE_FILENAME = "LICENSE_ACK_PRIVATE.template.json"
_REAL_ACK_FILENAME = "LICENSE_ACK_PRIVATE.json"


def template_payload() -> dict:
    """The exact contents of the template JSON. operator_acknowledged is
    always False; the operator must change it themselves AND rename the
    file to LICENSE_ACK_PRIVATE.json.
    """
    return {
        "operator_acknowledged": False,
        "acknowledged_systems": [],
        "_notes": (
            "Operator: rename this file to LICENSE_ACK_PRIVATE.json "
            "and set operator_acknowledged=true and acknowledged_systems "
            "to the systems you have legal local access to. Do NOT commit "
            "this file. The default .gitignore already excludes "
            "LICENSE_ACK_PRIVATE* and *TERMINOLOGY_PRIVATE*."
        ),
    }


@dataclass
class TemplateWriteResult:
    """Public-report-safe result of writing the template."""

    template_created: bool = False
    template_already_present: bool = False
    real_ack_present: bool = False
    real_ack_created: bool = False   # ALWAYS False in this block

    def safe_public_summary(self) -> dict:
        return {
            "template_created": self.template_created,
            "template_already_present": self.template_already_present,
            "real_ack_present": self.real_ack_present,
            "real_ack_created": False,   # invariant
        }


def write_ack_template(
    target_dir: Path,
    *,
    overwrite_template: bool = False,
) -> TemplateWriteResult:
    """Write `LICENSE_ACK_PRIVATE.template.json` into `target_dir`.

    Never creates `LICENSE_ACK_PRIVATE.json`. Never sets
    `operator_acknowledged=true`. Refuses to overwrite an existing
    template unless `overwrite_template=True`.
    """
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    template_path = target_dir / _TEMPLATE_FILENAME
    real_path = target_dir / _REAL_ACK_FILENAME

    result = TemplateWriteResult(
        real_ack_present=real_path.exists(),
        real_ack_created=False,
    )

    if template_path.exists() and not overwrite_template:
        result.template_already_present = True
        return result

    template_path.write_text(
        json.dumps(template_payload(), indent=2),
        encoding="utf-8",
    )
    result.template_created = True
    return result


def template_filename() -> str:
    return _TEMPLATE_FILENAME


def real_ack_filename() -> str:
    return _REAL_ACK_FILENAME
