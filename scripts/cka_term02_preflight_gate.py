"""CKA-TERM-01F TERM-02 preflight gate CLI.

Read-only. Does not import, download, or create license acknowledgments.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from clinical_knowledge.terminology.term02_preflight_gate import run_term02_preflight_gate  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check whether TERM-02 may start.")
    parser.add_argument("--terminology-root", default="terminology_data")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.terminology_root)
    if not root.is_absolute():
        root = REPO_ROOT / root
    result = run_term02_preflight_gate(repo_root=REPO_ROOT, terminology_root=root)
    payload = result.safe_public_summary()
    payload["term02_may_start"] = result.allowed
    payload["real_import_performed"] = False
    payload["external_api_used"] = False
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print("CKA TERM-02 preflight gate")
        print(f"term02_may_start: {result.allowed}")
        print(f"reason_codes: {', '.join(result.reason_codes) if result.reason_codes else 'none'}")
        print("real_import_performed: False")
        print("external_api_used: False")
    return 0 if result.allowed else 2


if __name__ == "__main__":
    raise SystemExit(main())
