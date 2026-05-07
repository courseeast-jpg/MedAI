"""Run CKA-TERM-01D synthetic terminology QA."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from clinical_knowledge.terminology import run_synthetic_terminology_qa  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run synthetic terminology golden lookup QA.",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON output.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    report = run_synthetic_terminology_qa().safe_public_summary()
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        metrics = report["metrics"]
        print("CKA-TERM-01D terminology QA")
        print(f"total_cases: {metrics['total_cases']}")
        print(f"passed_cases: {metrics['passed_cases']}")
        print(f"failed_cases: {metrics['failed_cases']}")
        print(f"b07_boundary_passed: {metrics['b07_boundary_passed']}")
        print("real_terminology_imported: False")
        print("external_api_used: False")
    return 0 if report["metrics"]["failed_cases"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
