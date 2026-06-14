#!/usr/bin/env python3
"""MEDAI-VERTEX-CREDENTIALS-ENVIRONMENT-PREFLIGHT-17C-R1.

Diagnose the local Vertex credential environment for this worktree after 17C failed at
credential resolution. Local-only diagnosis plus an optional token-only refresh check.

This block sends NO medical payload, makes NO Vertex model call, uploads NO file, opens
NO MKB DB, and never prints credential contents, access/refresh tokens, or private keys.
Public reports carry presence/absence booleans, sanitized paths, and status only.
"""
from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET_PROJECT = "sot-knowledge-ocr"

PRIVATE_OUT = Path(os.path.expandvars(r"%LOCALAPPDATA%\MedAI_Private\vertex_env_17C_R1"))
PRIVATE_HELPER = PRIVATE_OUT / "vertex_env_setter_PRIVATE.ps1"
PRIVATE_HELPER_LABEL = r"C:\Users\S1\AppData\Local\MedAI_Private\vertex_env_17C_R1\vertex_env_setter_PRIVATE.ps1"

REPORT_DIR = REPO_ROOT / "reports" / "medai_vertex_credentials_environment_preflight_17c_r1"

# Standard ADC / gcloud config locations (no contents are ever read).
ADC_CANDIDATES = [
    Path(os.path.expandvars(r"%APPDATA%\gcloud\application_default_credentials.json")),
    Path(os.path.expandvars(r"%USERPROFILE%\.config\gcloud\application_default_credentials.json")),
]
SHALLOW_SEARCH_ROOTS = [
    Path(os.path.expandvars(r"%APPDATA%\gcloud")),
    Path(os.path.expandvars(r"%USERPROFILE%\.config\gcloud")),
    Path(r"G:\Codex"),
]


def _sanitize_path(p: str | None) -> str:
    """Return a home-redacted path label (never the literal user home)."""
    if not p:
        return ""
    s = str(p)
    appdata = os.environ.get("APPDATA", "")
    userprofile = os.environ.get("USERPROFILE", "")
    localappdata = os.environ.get("LOCALAPPDATA", "")
    if localappdata and s.startswith(localappdata):
        return s.replace(localappdata, "%LOCALAPPDATA%", 1)
    if appdata and s.startswith(appdata):
        return s.replace(appdata, "%APPDATA%", 1)
    if userprofile and s.startswith(userprofile):
        return s.replace(userprofile, "%USERPROFILE%", 1)
    # Redact any drive-rooted absolute path generically.
    return re.sub(r"^[A-Za-z]:\\\\?[^\\]+\\[^\\]+", "<redacted-path-prefix>", s)


def _google_auth_importable() -> bool:
    try:
        import importlib.util
        return importlib.util.find_spec("google.auth") is not None
    except Exception:
        return False


def _gcloud_on_path() -> bool:
    return shutil.which("gcloud") is not None


def _find_adc() -> tuple[bool, str]:
    # Prefer GOOGLE_APPLICATION_CREDENTIALS if set and present.
    gac = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if gac and Path(gac).is_file():
        return True, _sanitize_path(gac)
    for c in ADC_CANDIDATES:
        if c.is_file():
            return True, _sanitize_path(str(c))
    # Shallow, top-level-only search for an ADC file (no deep recursion, no content read).
    for root in SHALLOW_SEARCH_ROOTS:
        try:
            if not root.is_dir():
                continue
            for entry in list(root.iterdir())[:200]:
                if entry.is_file() and entry.name == "application_default_credentials.json":
                    return True, _sanitize_path(str(entry))
        except OSError:
            continue
    return False, ""


def _detect_project() -> str:
    for var in ("GOOGLE_CLOUD_PROJECT", "GCLOUD_PROJECT", "MEDAI_VERTEX_PROJECT_ID"):
        v = os.environ.get(var)
        if v:
            return v.strip()
    # gcloud config (if present, read only the project line, not credentials).
    cfg = Path(os.path.expandvars(r"%APPDATA%\gcloud\configurations\config_default"))
    try:
        if cfg.is_file():
            for line in cfg.read_text(encoding="utf-8", errors="ignore").splitlines():
                m = re.match(r"\s*project\s*=\s*(\S+)", line)
                if m:
                    return m.group(1).strip()
    except OSError:
        pass
    return "none"


def _token_refresh_check() -> str:
    """Optional Level-2 token-only check. Never prints the token. No model call.
    Returns: not_checked | pass | login_required | fail."""
    try:
        import google.auth  # type: ignore
        import google.auth.transport.requests  # type: ignore
        from google.auth.exceptions import DefaultCredentialsError  # type: ignore
    except Exception:
        return "not_checked"
    try:
        creds, _proj = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    except Exception as exc:  # DefaultCredentialsError or similar
        name = type(exc).__name__
        return "login_required" if "DefaultCredentials" in name else "fail"
    try:
        req = google.auth.transport.requests.Request()
        creds.refresh(req)
        return "pass" if str(getattr(creds, "token", "") or "").strip() else "fail"
    except Exception as exc:  # noqa: BLE001
        low = str(exc).lower()
        if "reauth" in low or "invalid_grant" in low or "login" in low or "expired" in low:
            return "login_required"
        return "fail"


def run() -> dict[str, Any]:
    google_auth = _google_auth_importable()
    gcloud = _gcloud_on_path()
    adc_found, adc_sanitized = _find_adc()
    gac_set = bool(os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")) and Path(
        os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
    ).is_file()
    project = _detect_project()

    token_result = "not_checked"
    token_checked = False
    if google_auth and adc_found:
        token_checked = True
        token_result = _token_refresh_check()

    # Resolve the concrete ADC path for the private helper (kept private, never reported raw).
    adc_real = ""
    gac = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if gac and Path(gac).is_file():
        adc_real = gac
    else:
        for c in ADC_CANDIDATES:
            if c.is_file():
                adc_real = str(c)
                break

    helper_created = False
    if adc_found and adc_real:
        try:
            PRIVATE_OUT.mkdir(parents=True, exist_ok=True)
            PRIVATE_HELPER.write_text(
                "# Private 17C credential environment setter (NO credential contents inside).\n"
                "# Sets the ADC path + project for re-running 17C in this shell. Do not commit.\n"
                f'$env:GOOGLE_APPLICATION_CREDENTIALS = "{adc_real}"\n'
                f'$env:GOOGLE_CLOUD_PROJECT = "{TARGET_PROJECT}"\n'
                'Write-Output "Vertex env set for project sot-knowledge-ocr (ADC path configured)."\n',
                encoding="utf-8",
            )
            (PRIVATE_OUT / "credential_path_private.txt").write_text(adc_real + "\n", encoding="utf-8")
            (PRIVATE_OUT / "vertex_env_status_private.json").write_text(
                json.dumps({"adc_path": adc_real, "project": TARGET_PROJECT,
                            "token_refresh_result": token_result}, indent=2), encoding="utf-8")
            helper_created = True
        except OSError:
            helper_created = False

    ready = bool(adc_found and token_result == "pass")

    if token_result == "login_required":
        result = "LOGIN_REQUIRED"
    elif token_result == "pass" and ready:
        result = "PASS"
    elif not adc_found or not google_auth:
        result = "FAIL"
    else:
        result = "FAIL"

    summary = {
        "block": "MEDAI-VERTEX-CREDENTIALS-ENVIRONMENT-PREFLIGHT-17C-R1",
        "local_only_preflight": True,
        "medical_payload_sent": False,
        "vertex_model_call_made": False,
        "provider_content_call_made": False,
        "billing_api_call_made": False,
        "mkb_db_opened": False,
        "mkb_write": False,
        "credentials_printed": False,
        "tokens_printed": False,
        "credential_file_committed": False,
        "credential_file_copied_to_repo": False,
        "gcloud_on_path": gcloud,
        "google_auth_importable": google_auth,
        "adc_file_found": adc_found,
        "adc_file_path_sanitized": adc_sanitized,
        "google_application_credentials_set": gac_set,
        "project_detected": project if project in (TARGET_PROJECT,) else (project if project != "none" else "none"),
        "token_refresh_checked": token_checked,
        "token_refresh_result": token_result,
        "private_env_helper_created": helper_created,
        "private_env_helper_path": PRIVATE_HELPER_LABEL,
        "ready_to_rerun_17c": ready,
        "preflight_result": result,
        "safety_result": "passed",
    }
    return summary


def _matrix(s: dict[str, Any]) -> str:
    keys = ["google_auth_importable", "gcloud_on_path", "adc_file_found",
            "adc_file_path_sanitized", "google_application_credentials_set", "project_detected",
            "token_refresh_checked", "token_refresh_result", "private_env_helper_created",
            "ready_to_rerun_17c", "preflight_result"]
    return "\n".join(["# 17C-R1 credential environment matrix", "",
                      "| Check | Value |", "| --- | --- |",
                      *[f"| {k} | `{s[k]}` |" for k in keys], "",
                      "Presence/absence and sanitized paths only. No credential contents, tokens,",
                      "or private keys are printed or committed.", ""])


def _next_steps(s: dict[str, Any]) -> str:
    if s["preflight_result"] == "PASS":
        body = (
            "Credentials resolve and a token refresh succeeded. To re-run 17C (separate, "
            "still-authorized step):\n\n"
            "1. In a NEW shell, dot-source the private helper to set the ADC path + project:\n"
            "   . $env:LOCALAPPDATA\\MedAI_Private\\vertex_env_17C_R1\\vertex_env_setter_PRIVATE.ps1\n"
            "2. Confirm `MEDAI_AI_FIRST_CORPUS_SMALL_LIVE_BATCH_17C_APPROVED` is unset before start.\n"
            "3. Re-run the 17C script. It enforces the $0.05 cap, stop-on-first-failure, and the\n"
            "   gate lifecycle. (Re-running 17C is NOT performed by this block.)\n"
        )
    elif s["preflight_result"] == "LOGIN_REQUIRED":
        body = (
            "Credentials require interactive login/reauth (no secret was printed; no model call\n"
            "was made). A human must run a gcloud/ADC login out-of-band, for example:\n"
            "   gcloud auth application-default login --project sot-knowledge-ocr\n"
            "Then re-run this preflight. 17C is NOT re-run here.\n"
        )
    else:
        body = (
            "Credentials are not usable in this worktree. Either install/configure gcloud, or\n"
            "place a valid Application Default Credentials file at the standard ADC path with the\n"
            "cloud-platform scope and billing enabled for project sot-knowledge-ocr, then re-run\n"
            "this preflight. 17C is NOT re-run here.\n"
        )
    return f"# Next steps to re-run 17C\n\nResult: **{s['preflight_result']}**\n\n{body}"


def _implementation(s: dict[str, Any]) -> str:
    lines = [f"# {s['block']}", "", f"## Result: **{s['preflight_result']}**", "", "## Findings", ""]
    for k in ("google_auth_importable", "gcloud_on_path", "adc_file_found", "adc_file_path_sanitized",
              "google_application_credentials_set", "project_detected", "token_refresh_checked",
              "token_refresh_result", "private_env_helper_created", "ready_to_rerun_17c"):
        lines.append(f"- {k}: `{s[k]}`")
    lines += ["", "## Safety", "",
              "- No medical payload sent; no Vertex model call; no provider content call; no billing call.",
              "- No MKB DB opened or written. No credential contents, tokens, or private keys printed.",
              "- No credential file copied into or committed to the repo.",
              "- A private environment helper (no credential contents) may be created outside the repo.",
              ""]
    return "\n".join(lines)


def main() -> int:
    s = run()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    matrix_md = _matrix(s)
    next_md = _next_steps(s)
    impl_md = _implementation(s)

    # Defense-in-depth: ensure no token-like / credential-like content in public blobs.
    public_blob = "\n".join([json.dumps(s), matrix_md, next_md, impl_md])
    secret_markers = ("ya29.", "AIza", "-----BEGIN", "refresh_token", "private_key", "client_secret")
    if any(m in public_blob for m in secret_markers):
        s["safety_result"] = "blocked"
        s["credentials_printed"] = True  # would indicate a leak; force visible failure

    (REPORT_DIR / "summary.json").write_text(json.dumps(s, indent=2), encoding="utf-8")
    (REPORT_DIR / "implementation_report.md").write_text(_implementation(s), encoding="utf-8")
    (REPORT_DIR / "credential_environment_matrix.md").write_text(matrix_md, encoding="utf-8")
    (REPORT_DIR / "next_steps_to_rerun_17c.md").write_text(next_md, encoding="utf-8")

    ok = s["safety_result"] == "passed" and not s["credentials_printed"] and not s["tokens_printed"]
    print(f"medai_vertex_credentials_environment_preflight_17c_r1_{s['preflight_result'].lower()}"
          if ok else "medai_vertex_credentials_environment_preflight_17c_r1_unsafe")
    print(json.dumps({k: s[k] for k in (
        "preflight_result", "google_auth_importable", "gcloud_on_path", "adc_file_found",
        "google_application_credentials_set", "project_detected", "token_refresh_checked",
        "token_refresh_result", "private_env_helper_created", "ready_to_rerun_17c",
        "medical_payload_sent", "vertex_model_call_made", "credentials_printed",
        "tokens_printed", "safety_result")}, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
