"""CKA-TERM-01G synthetic terminology scale test CLI."""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from clinical_knowledge.terminology.import_limits import build_import_limits  # noqa: E402
from clinical_knowledge.terminology.import_resume import simulate_chunked_import_with_resume  # noqa: E402
from clinical_knowledge.terminology.import_scale import build_scale_fixtures, parse_scale_fixture  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run synthetic scale/resume terminology import simulation.")
    parser.add_argument("--rows", type=int, default=120)
    parser.add_argument("--max-rows-per-file", type=int, default=50)
    parser.add_argument("--chunk-size", type=int, default=10)
    parser.add_argument("--interrupt-after-chunks", type=int, default=2)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    limits = build_import_limits(max_rows_per_file=args.max_rows_per_file, chunk_size=args.chunk_size)
    with tempfile.TemporaryDirectory(prefix="medai_term01g_scale_") as tmp:
        fixtures = build_scale_fixtures(Path(tmp), rows_per_system=args.rows)
        metrics = []
        for fixture in fixtures:
            parsed = parse_scale_fixture(fixture, max_rows=limits.max_rows_per_file_default)
            result = simulate_chunked_import_with_resume(
                parsed,
                limits=limits,
                interrupt_after_chunks=args.interrupt_after_chunks,
            )
            metrics.append(result.safe_public_summary())
    payload = {
        "block_id": "CKA-TERM-01G",
        "synthetic_scale_test_completed": True,
        "systems_tested": [m["system"] for m in metrics],
        "metrics": metrics,
        "no_real_terminology_import_performed": True,
        "no_real_terminology_files_used": True,
        "external_api_used": False,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
