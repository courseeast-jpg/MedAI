"""Validate B07-TERM opt-in planning artifacts."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REPORT_DIR = ROOT / "reports" / "b07_term_opt_in_planning"
REPORT_JSON = REPORT_DIR / "b07_term_opt_in_planning_report.json"
REQUIRED_FILES = [
    "B07_TERM_OPT_IN_INTEGRATION_PLAN.md",
    "B07_TERM_SAFETY_CONTRACT.md",
    "B07_TERM_ROLLBACK_PLAN.md",
    "B07_TERM_IMPLEMENTATION_PROMPT.md",
    "b07_term_opt_in_planning_report.json",
    "b07_term_opt_in_planning_report.md",
]
REQUIRED_PHRASES = [
    "MEDAI_B07_TERMINOLOGY_OPT_IN=false",
    "MEDAI_TERMINOLOGY_LOOKUP_ENABLED=false",
    "MEDAI_TERMINOLOGY_HYPOTHESIS_ANNOTATION=false",
    "MEDAI_TERMINOLOGY_READ_ONLY=true",
    "MEDAI_TERMINOLOGY_ALLOW_WRITES=false",
    "Unknown terms must remain unmapped",
    "Ambiguous terms must remain ambiguous",
    "No code hallucination",
    "No accepted clinical fact",
    "No hypothesis promotion",
    "No DDI status clearing",
    "No external API",
]
FORBIDDEN_IMPLEMENTATION_FILES = [
    ROOT / "clinical_knowledge" / "terminology" / "b07_term_integration.py",
]
APPROVED_B07_TERM01_FILE = ROOT / "clinical_knowledge" / "terminology" / "b07_term_opt_in.py"
TERM01_REPORT_JSON = ROOT / "reports" / "b07_term01_opt_in_integration" / "b07_term01_opt_in_integration_report.json"


def main() -> int:
    from clinical_knowledge.privacy.report_privacy import check_public_report_payload

    missing_files = [name for name in REQUIRED_FILES if not (REPORT_DIR / name).exists()]
    combined = "\n".join(
        (REPORT_DIR / name).read_text(encoding="utf-8")
        for name in REQUIRED_FILES
        if (REPORT_DIR / name).exists()
    )
    missing_phrases = [phrase for phrase in REQUIRED_PHRASES if phrase not in combined]
    implementation_files_created = [str(path.relative_to(ROOT)).replace("\\", "/") for path in FORBIDDEN_IMPLEMENTATION_FILES if path.exists()]
    approved_term01_present = _approved_b07_term01_present()
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
        or path.lower().endswith((".rrf", ".csv", ".zip", ".sqlite", ".db", ".key", ".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"))
    ]
    required_flags = {
        "design_only": True,
        "b07_runtime_behavior_changed": False,
        "b07_integration_implemented": False,
        "accepted_clinical_facts_created": False,
        "hypothesis_promotion_allowed": False,
        "ddi_status_clearing_allowed": False,
        "external_api_used": False,
        "ocr_extractor_safety_gates_changed": False,
        "implementation_files_created": False,
    }
    flag_failures = {
        key: payload.get(key)
        for key, expected in required_flags.items()
        if payload.get(key) is not expected
    }
    passed = (
        not missing_files
        and not missing_phrases
        and not implementation_files_created
        and approved_term01_present
        and privacy_passed
        and not blocked_staged
        and not flag_failures
    )
    result = {
        "block_id": "B07-TERM-PLAN-01",
        "conclusion": "b07_term_opt_in_planning_ready" if passed else "b07_term_opt_in_planning_blocked",
        "missing_files": missing_files,
        "missing_phrases": missing_phrases,
        "implementation_files_created": implementation_files_created,
        "approved_b07_term01_opt_in_present": approved_term01_present,
        "privacy_report_clean": privacy_passed,
        "blocked_staged_files": blocked_staged,
        "flag_failures": flag_failures,
        "external_api_used": False,
        "b07_runtime_behavior_changed": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if passed else 1


def _git_staged_paths() -> list[str]:
    proc = subprocess.run(["git", "diff", "--cached", "--name-only"], cwd=ROOT, text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        return []
    return [line.strip().replace("\\", "/") for line in proc.stdout.splitlines() if line.strip()]


def _approved_b07_term01_present() -> bool:
    if not APPROVED_B07_TERM01_FILE.exists() or not TERM01_REPORT_JSON.exists():
        return False
    try:
        payload = json.loads(TERM01_REPORT_JSON.read_text(encoding="utf-8"))
    except Exception:
        return False
    return (
        payload.get("conclusion") == "b07_term01_opt_in_integration_ready"
        and payload.get("feature_flags_default_off") is True
        and payload.get("writes_active_fact") is False
        and payload.get("promotes_hypothesis") is False
        and payload.get("clears_ddi_status") is False
        and payload.get("external_api_used") is False
    )


if __name__ == "__main__":
    raise SystemExit(main())
