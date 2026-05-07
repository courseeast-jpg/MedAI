"""CKA-TERM-01A — operator-facing intake preparation.

Creates the gitignored `terminology_data/` tree and drops a license
acknowledgment TEMPLATE. NEVER creates the real ack file. NEVER
downloads anything. NEVER copies files unless the operator explicitly
passes `--copy-approved` with a `--scan` directory. NEVER extracts
ZIPs unless the operator explicitly passes `--extract-approved`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from clinical_knowledge.terminology import (    # noqa: E402
    copy_classified_files,
    optional_local_scan,
    prepare_intake_folders,
    safe_extract_zip,
)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cka_terminology_prepare_intake",
        description=(
            "Prepare local terminology_data/ folders + license-ack template. "
            "Never downloads, never bypasses licensing, never commits files."
        ),
    )
    p.add_argument("--scan", default=None,
                   help="Optional local folder to scan for classified terminology files. Off by default.")
    p.add_argument("--recurse", action="store_true",
                   help="Recurse into subfolders during --scan (default off).")
    p.add_argument("--copy-approved", action="store_true",
                   help="Copy recognized files from --scan into terminology_data/<system>/. "
                        "Refused without an explicit --scan path.")
    p.add_argument("--extract-approved", action="store_true",
                   help="Extract recognized ZIPs from --scan into terminology_data/<system>/. "
                        "zip-slip protection always on.")
    p.add_argument("--overwrite-template", action="store_true",
                   help="Re-write the license-ack template if one already exists.")
    return p


def main(argv=None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    out: dict = {
        "preparation": {},
        "scan": {},
        "copy": {},
        "extract": {},
        "real_terminology_downloaded": False,
        "real_terminology_imported": False,
        "license_gate_bypassed": False,
    }

    prep = prepare_intake_folders(
        repo_root=REPO_ROOT,
        write_template=True,
        overwrite_template=args.overwrite_template,
    )
    out["preparation"] = prep.safe_public_summary()

    scan_dir = Path(args.scan).resolve() if args.scan else None
    scan = optional_local_scan(
        scan_dir=scan_dir,
        enabled=bool(args.scan),
        recurse=args.recurse,
    )
    out["scan"] = scan.safe_public_summary()

    # Copy is gated behind both --scan AND --copy-approved.
    files_for_copy = []
    archives_for_extract = []
    if args.scan and (args.copy_approved or args.extract_approved):
        from clinical_knowledge.terminology import classify_filename
        target = scan_dir
        if target and target.exists() and target.is_dir():
            walker = target.rglob("*") if args.recurse else target.iterdir()
            for child in walker:
                if not child.is_file():
                    continue
                c = classify_filename(child.name)
                if c.system is None:
                    continue
                if c.is_zip:
                    archives_for_extract.append(child)
                else:
                    files_for_copy.append(child)

    copy_result = copy_classified_files(
        files_for_copy,
        repo_root=REPO_ROOT,
        copy_approved=bool(args.copy_approved),
    )
    out["copy"] = copy_result.safe_public_summary()

    extract_result = safe_extract_zip(
        archives_for_extract,
        repo_root=REPO_ROOT,
        extract_approved=bool(args.extract_approved),
    )
    out["extract"] = extract_result.safe_public_summary()

    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
