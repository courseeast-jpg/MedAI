"""CKA-TERM-01 — operator-facing inventory CLI for terminology_data/.

Prints a public-safe summary of what's present locally; never reads
file contents, never opens vendor APIs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from clinical_knowledge.terminology import inventory_terminology_data_dir    # noqa: E402


def main() -> int:
    inv = inventory_terminology_data_dir()
    print(json.dumps(inv.safe_public_summary(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
