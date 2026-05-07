"""CKA-TERM-01A — operator-facing readiness checker.

Combines the TERM-01 inventory with the license-gate state and tells
the operator exactly which systems still need acknowledgment.

No network. No file content reading. No license bypass.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from clinical_knowledge.terminology import (    # noqa: E402
    compute_readiness,
    inventory_terminology_data_dir,
)


def main() -> int:
    inv = inventory_terminology_data_dir(repo_root=REPO_ROOT)
    rd = compute_readiness(repo_root=REPO_ROOT)
    out = {
        "inventory": inv.safe_public_summary(),
        "readiness": rd.safe_public_summary(),
        "operator_action_required": (
            "Add the following systems to LICENSE_ACK_PRIVATE.json's "
            "acknowledged_systems list to make them import_ready: "
            f"{rd.pending_acknowledgments}"
            if rd.pending_acknowledgments
            else "No pending acknowledgments. Either no licensed files "
                 "are present or all present systems are already "
                 "acknowledged."
        ),
        "next_code_action_after_manual_files": (
            "CKA-TERM-02 controlled local terminology import"
        ),
    }
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
