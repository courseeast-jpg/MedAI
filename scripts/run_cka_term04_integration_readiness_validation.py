"""Validate CKA-TERM-04 design-only integration readiness artifacts."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REPORT_DIR = ROOT / "reports" / "cka_term04_integration_readiness_blueprint"
REPORT_JSON = REPORT_DIR / "cka_term04_integration_readiness_report.json"
REQUIRED_FILES = [
    "CKA_TERM04_INTEGRATION_READINESS_BLUEPRINT.md",
    "CKA_TERM04_SAFETY_CONTRACT.md",
    "CKA_TERM04_FUTURE_BLOCK_PLAN.md",
    "cka_term04_integration_readiness_report.json",
    "cka_term04_integration_readiness_report.md",
]
REQUIRED_PHRASES = [
    "TERM-05 synthetic read-only terminology adapter",
    "TERM-06 private-store read-only adapter validation",
    "TERM-07 UI-only terminology lookup panel",
    "TERM-08 hypothesis-only coding annotation pilot",
    "B07-TERM opt-in integration",
    "Unknown terms must return unmapped",
    "Ambiguous terms must return ambiguous/manual-review",
    "must not generate clinical advice",
    "must not clear or downgrade DDI status",
]


def main() -> int:
    from clinical_knowledge.privacy.report_privacy import check_public_report_payload

    missing = [name for name in REQUIRED_FILES if not (REPORT_DIR / name).exists()]
    combined = "\n".join(
        (REPORT_DIR / name).read_text(encoding="utf-8")
        for name in REQUIRED_FILES
        if (REPORT_DIR / name).exists()
    )
    phrase_missing = [phrase for phrase in REQUIRED_PHRASES if phrase not in combined]
    payload = json.loads(REPORT_JSON.read_text(encoding="utf-8")) if REPORT_JSON.exists() else {}
    privacy_checks = []
    for name in REQUIRED_FILES:
        path = REPORT_DIR / name
        if not path.exists():
            continue
        item = json.loads(path.read_text(encoding="utf-8")) if path.suffix == ".json" else {name: path.read_text(encoding="utf-8")}
        privacy_checks.append((name, check_public_report_payload(item)))
    privacy_passed = all(check.passed for _, check in privacy_checks)
    staged = _git_staged_paths()
    blocked_staged = [
        path for path in staged
        if path.startswith("terminology_data/")
        or path.startswith("data/terminology/")
        or "LICENSE_ACK_PRIVATE" in path
        or path.lower().endswith((".rrf", ".csv", ".zip", ".sqlite", ".db", ".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"))
    ]
    required_flags = {
        "external_api_used": False,
        "clinical_decision_logic_changed": False,
        "b07_integration_changed": False,
        "b07_integration_enabled": False,
        "ocr_extractor_safety_gates_changed": False,
        "terminology_data_staged": False,
        "terminology_data_committed": False,
        "local_db_index_committed": False,
    }
    flag_failures = {
        key: payload.get(key)
        for key, expected in required_flags.items()
        if payload.get(key) is not expected
    }
    passed = not missing and not phrase_missing and privacy_passed and not blocked_staged and not flag_failures
    result = {
        "block_id": "CKA-TERM-04",
        "conclusion": "cka_term04_integration_readiness_blueprint_ready" if passed else "cka_term04_integration_readiness_blueprint_blocked",
        "required_files_present": not missing,
        "missing_files": missing,
        "required_phrases_present": not phrase_missing,
        "missing_phrases": phrase_missing,
        "privacy_report_clean": privacy_passed,
        "blocked_staged_files": blocked_staged,
        "flag_failures": flag_failures,
        "external_api_used": False,
        "clinical_logic_changed": False,
        "b07_integration_changed": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if passed else 1


def _git_staged_paths() -> list[str]:
    proc = subprocess.run(["git", "diff", "--cached", "--name-only"], cwd=ROOT, text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        return []
    return [line.strip().replace("\\", "/") for line in proc.stdout.splitlines() if line.strip()]


if __name__ == "__main__":
    raise SystemExit(main())
