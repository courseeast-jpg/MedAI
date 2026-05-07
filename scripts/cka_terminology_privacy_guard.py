"""Run CKA-TERM-01H terminology privacy guard checks."""
from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> int:
    from clinical_knowledge.terminology.privacy_regression import run_privacy_regression_checks
    from clinical_knowledge.terminology.staging_guard import check_terminology_staging

    privacy = run_privacy_regression_checks().safe_public_summary()
    staging = check_terminology_staging(repo_root=ROOT).safe_public_summary()
    payload = {
        "block_id": "CKA-TERM-01H",
        "privacy_guard_ready": True,
        "privacy_regression": privacy,
        "actual_staging_guard": staging,
        "external_api_used": False,
        "no_real_import_performed": True,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if privacy["raw_path_leak_blocked"] and privacy["license_text_leak_blocked"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
