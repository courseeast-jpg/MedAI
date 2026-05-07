"""CKA-TERM-01F one-command manual return readiness pack.

Runs safe local readiness checks only. It does not import terminology data,
download terminology files, or create a real LICENSE_ACK_PRIVATE.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from clinical_knowledge.terminology.manual_return_pack import run_manual_return_pack  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run safe terminology manual-return readiness checks.")
    parser.add_argument("--prepare", action="store_true", help="Also prepare local gitignored intake folders/template.")
    parser.add_argument("--skip-readiness", action="store_true")
    parser.add_argument("--skip-dry-run", action="store_true")
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_manual_return_pack(
        repo_root=REPO_ROOT,
        run_prepare=args.prepare,
        run_readiness=not args.skip_readiness,
        run_dry_run=not args.skip_dry_run,
        run_preflight=not args.skip_preflight,
    )
    payload = result.safe_public_summary()
    payload["term02_not_started"] = True
    payload["real_license_ack_created"] = False
    payload["no_real_download_performed"] = True
    payload["no_real_terminology_import_performed"] = True
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print("CKA-TERM-01F manual return readiness pack")
        print(f"term02_preflight_allowed: {payload['term02_preflight_allowed']}")
        print(f"term02_preflight_reason_codes: {', '.join(payload['term02_preflight_reason_codes']) or 'none'}")
        print("real_import_performed: False")
        print("external_api_used: False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
