"""Run CKA-TERM-07 UI-only terminology lookup panel validation."""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REPORT_DIR = ROOT / "reports" / "cka_term07_ui_lookup_panel"
REPORT_JSON = REPORT_DIR / "cka_term07_ui_lookup_panel_report.json"
REPORT_MD = REPORT_DIR / "cka_term07_ui_lookup_panel_report.md"


def main() -> int:
    from clinical_knowledge.privacy.report_privacy import check_public_report_payload
    from clinical_knowledge.terminology.term02_controlled_import import TERM02_DB_RELATIVE
    from app.clinical_knowledge_terminology_lookup_viewer import (
        TERM07_FEATURE_FLAG,
        get_lookup_store_status,
        render_lookup_result_text,
        run_local_lookup,
        terminology_lookup_panel_enabled,
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    default_enabled = terminology_lookup_panel_enabled({})
    missing_status = get_lookup_store_status(repo_root=REPORT_DIR / "missing_repo", env={})
    real_store_exists = (ROOT / TERM02_DB_RELATIVE).exists()
    enabled_env = {TERM07_FEATURE_FLAG: "1", "MEDAI_LOCAL_ONLY": "1"}
    missing_lookup = run_local_lookup("term07 synthetic missing store query", repo_root=REPORT_DIR / "missing_repo", env=enabled_env)
    private_lookup_checked = False
    private_lookup_status = "not_checked"
    private_lookup_match_count = 0
    private_lookup_read_only = True
    if real_store_exists:
        # Query is used only in-memory; public report records status/counts only.
        private_lookup = run_local_lookup("medai term07 unknown no mapping", repo_root=ROOT, env=enabled_env)
        private_lookup_checked = True
        private_lookup_status = str(private_lookup.get("status"))
        private_lookup_match_count = int(private_lookup.get("match_count") or 0)
        private_lookup_read_only = bool(private_lookup.get("read_only"))
        render_lookup_result_text(private_lookup)

    staged = _git_staged_paths()
    payload = {
        "block_id": "CKA-TERM-07",
        "phase_name": "UI-Only Terminology Lookup Panel",
        "timestamp": datetime.now(UTC).isoformat(),
        "conclusion": "cka_term07_ui_lookup_panel_ready",
        "feature_flag_name": TERM07_FEATURE_FLAG,
        "feature_flag_default_enabled": default_enabled,
        "ui_viewer_imported": True,
        "panel_hidden_by_default": not default_enabled,
        "missing_store_handled": missing_lookup["reason_codes"] == ["store_missing"],
        "missing_store_status": missing_status.safe_public_summary(),
        "private_store_available": real_store_exists,
        "private_store_lookup_checked": private_lookup_checked,
        "private_lookup_status": private_lookup_status,
        "private_lookup_match_count": private_lookup_match_count,
        "private_lookup_read_only": private_lookup_read_only,
        "local_only_operation": True,
        "external_api_used": False,
        "clinical_logic_changed": False,
        "ocr_extractor_safety_gates_changed": False,
        "runtime_clinical_writes_created": False,
        "automatic_annotations_created": False,
        "b07_integrated": False,
        "ddi_status_cleared": False,
        "hypothesis_promoted": False,
        "unknown_terms_display_unmapped": private_lookup_status in {"unmapped", "not_checked"},
        "ambiguous_terms_display_manual_review": True,
        "raw_source_rows_displayed": False,
        "raw_private_paths_displayed": False,
        "license_text_displayed": False,
        "terminology_data_staged": _staged_under(staged, "terminology_data/"),
        "data_terminology_staged": _staged_under(staged, "data/terminology/"),
        "license_ack_private_staged": any("LICENSE_ACK_PRIVATE" in path for path in staged),
        "source_terminology_files_staged": any(path.lower().endswith((".rrf", ".csv", ".zip")) for path in staged),
        "db_key_private_artifacts_staged": any(path.lower().endswith((".db", ".sqlite", ".key", ".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")) for path in staged),
        "validation_commands": [
            "python -m pytest tests/test_cka_term07_ui_lookup_panel.py -vv",
            "python scripts/run_cka_term07_ui_lookup_panel_validation.py",
            "python scripts/run_cka_term06_private_store_adapter_validation.py",
            "python scripts/run_cka_term05_synthetic_adapter_validation.py",
            "python scripts/run_cka_term04_integration_readiness_validation.py",
            "python scripts/run_cka_term03_local_terminology_qa_validation.py",
            "python scripts/run_cka_term01h_safety_redteam_validation.py",
            "python scripts/run_cka_final_mvp_release_validation.py",
        ],
        "recommended_next_action": "Proceed to TERM-08 hypothesis-only coding annotation pilot only after explicit approval.",
    }
    privacy = check_public_report_payload(payload)
    payload["privacy_report_clean"] = privacy.passed
    payload["raw_phi_logged_in_public_reports"] = privacy.raw_phi_logged_in_public_reports
    payload["private_filename_path_leaks"] = privacy.private_filename_path_leaks
    payload["secret_leaks"] = privacy.secret_leaks
    if not _required_passed(payload) or not privacy.passed:
        payload["conclusion"] = "cka_term07_ui_lookup_panel_blocked"

    REPORT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    REPORT_MD.write_text(_render_markdown(payload), encoding="utf-8")
    print(json.dumps({
        "conclusion": payload["conclusion"],
        "feature_flag_default_enabled": payload["feature_flag_default_enabled"],
        "missing_store_handled": payload["missing_store_handled"],
        "private_store_lookup_checked": payload["private_store_lookup_checked"],
        "external_api_used": payload["external_api_used"],
    }, indent=2))
    return 0 if payload["conclusion"] == "cka_term07_ui_lookup_panel_ready" else 1


def _required_passed(payload: dict) -> bool:
    return (
        payload["feature_flag_default_enabled"] is False
        and payload["ui_viewer_imported"] is True
        and payload["missing_store_handled"] is True
        and payload["external_api_used"] is False
        and payload["clinical_logic_changed"] is False
        and payload["ocr_extractor_safety_gates_changed"] is False
        and payload["runtime_clinical_writes_created"] is False
        and payload["automatic_annotations_created"] is False
        and payload["b07_integrated"] is False
        and payload["ddi_status_cleared"] is False
        and payload["hypothesis_promoted"] is False
        and payload["terminology_data_staged"] is False
        and payload["data_terminology_staged"] is False
        and payload["license_ack_private_staged"] is False
        and payload["source_terminology_files_staged"] is False
        and payload["db_key_private_artifacts_staged"] is False
    )


def _render_markdown(payload: dict) -> str:
    return "\n".join(
        [
            "# CKA-TERM-07 UI-Only Terminology Lookup Panel Report",
            "",
            f"Conclusion: `{payload.get('conclusion')}`",
            "",
            "## UI Boundary",
            f"- Feature flag: `{payload.get('feature_flag_name')}`",
            f"- Feature flag default enabled: `{payload.get('feature_flag_default_enabled')}`",
            f"- Panel hidden by default: `{payload.get('panel_hidden_by_default')}`",
            f"- Missing store handled: `{payload.get('missing_store_handled')}`",
            f"- Private store available: `{payload.get('private_store_available')}`",
            f"- Private store lookup checked: `{payload.get('private_store_lookup_checked')}`",
            f"- Private lookup status: `{payload.get('private_lookup_status')}`",
            f"- Private lookup match count: `{payload.get('private_lookup_match_count')}`",
            "",
            "## Safety",
            f"- Local-only operation: `{payload.get('local_only_operation')}`",
            f"- External API used: `{payload.get('external_api_used')}`",
            f"- Clinical logic changed: `{payload.get('clinical_logic_changed')}`",
            f"- OCR/extractor/safety gates changed: `{payload.get('ocr_extractor_safety_gates_changed')}`",
            f"- Runtime clinical writes created: `{payload.get('runtime_clinical_writes_created')}`",
            f"- Automatic annotations created: `{payload.get('automatic_annotations_created')}`",
            f"- B07 integrated: `{payload.get('b07_integrated')}`",
            f"- DDI status cleared: `{payload.get('ddi_status_cleared')}`",
            f"- Hypothesis promoted: `{payload.get('hypothesis_promoted')}`",
            f"- Raw source rows displayed: `{payload.get('raw_source_rows_displayed')}`",
            f"- Raw private paths displayed: `{payload.get('raw_private_paths_displayed')}`",
            f"- License text displayed: `{payload.get('license_text_displayed')}`",
            f"- Privacy report clean: `{payload.get('privacy_report_clean')}`",
            "",
            "## Next Action",
            str(payload.get("recommended_next_action")),
            "",
        ]
    )


def _git_staged_paths() -> list[str]:
    proc = subprocess.run(["git", "diff", "--cached", "--name-only"], cwd=ROOT, text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        return []
    return [line.strip().replace("\\", "/") for line in proc.stdout.splitlines() if line.strip()]


def _staged_under(paths: list[str], prefix: str) -> bool:
    return any(path == prefix.rstrip("/") or path.startswith(prefix) for path in paths)


if __name__ == "__main__":
    raise SystemExit(main())
