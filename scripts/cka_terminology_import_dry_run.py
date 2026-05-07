"""CKA-TERM-01B terminology import dry-run CLI.

This command plans capacity only. It does not import terminology data, does
not create a production terminology index, and does not create a private
license acknowledgement.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from clinical_knowledge.terminology import (  # noqa: E402
    build_import_limits,
    run_terminology_import_dry_run,
)


def _resolve_repo_root(terminology_root: str) -> Path:
    root = Path(terminology_root)
    if not root.is_absolute():
        root = REPO_ROOT / root
    if root.name == "terminology_data":
        return root.parent
    return REPO_ROOT


def _message_for_plan(result: dict) -> str:
    plan = result["plan"]
    if plan["estimated_files"] == 0:
        return "local terminology files required"
    if plan["systems_blocked_license"]:
        systems = ", ".join(plan["systems_blocked_license"])
        return f"systems needing acknowledgment: {systems}"
    if plan["systems_import_ready"]:
        systems = ", ".join(plan["systems_import_ready"])
        return (
            f"import-ready dry run for: {systems}. TERM-02 is required "
            "for actual import."
        )
    return "dry-run plan completed; no import performed"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plan a local terminology import without importing data.",
    )
    parser.add_argument(
        "--terminology-root",
        default="terminology_data",
        help="Local terminology_data root to inspect.",
    )
    parser.add_argument(
        "--max-rows-per-file",
        type=int,
        default=None,
        help="Planning cap per source file.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Planning chunk size.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable dry-run output.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    limits = build_import_limits(
        max_rows_per_file=args.max_rows_per_file,
        chunk_size=args.chunk_size,
    )
    result = run_terminology_import_dry_run(
        repo_root=_resolve_repo_root(args.terminology_root),
        limits=limits,
    )
    result["message"] = _message_for_plan(result)
    result["dry_run_only"] = True
    result["real_files_imported"] = False
    result["production_index_created"] = False

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print("CKA-TERM-01B terminology import dry run")
        print(f"message: {result['message']}")
        print(f"dry_run: {result['plan']['dry_run']}")
        print(f"import_allowed: {result['plan']['import_allowed']}")
        print("real_files_imported: False")
        print("production_index_created: False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
