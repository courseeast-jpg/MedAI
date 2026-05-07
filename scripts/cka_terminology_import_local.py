"""CKA-TERM-01 — operator-facing local-import CLI (TERM-01 stub).

In TERM-01, this CLI does NOT perform a large licensed-import. It
prints the inventory + license-gate state so the operator can confirm
readiness. The actual licensed import is deferred to a future block
(CKA-TERM-02 controlled local import), which will require a separate
operator approval cycle.

Refuses any `--key` / `--encryption-key` argument. Defensive only:
this script does not accept or use encryption keys at all.
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
    TerminologySystem,
    inventory_terminology_data_dir,
    license_acknowledged_for,
)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cka_terminology_import_local",
        description=(
            "Inspect local terminology readiness. TERM-01 does NOT perform a "
            "real licensed-import; that is deferred to TERM-02."
        ),
    )
    p.add_argument("--system", choices=[
        TerminologySystem.UMLS.value, TerminologySystem.SNOMED_CT.value,
        TerminologySystem.RXNORM.value, TerminologySystem.LOINC.value, "all",
    ], default="all", help="Terminology system to inspect.")
    # Defensive: refuse key flags even though this script does not use a key.
    p.add_argument("--key", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--encryption-key", action="store_true", help=argparse.SUPPRESS)
    return p


def main(argv=None) -> int:
    raw = argv if argv is not None else sys.argv[1:]
    for token in raw:
        if token.startswith("--key=") or token.startswith("--encryption-key=") \
                or token in ("--key", "--encryption-key"):
            print("ERROR: command_line_key_not_accepted", file=sys.stderr)
            return 2
    parser = _build_parser()
    args = parser.parse_args(raw)

    inv = inventory_terminology_data_dir()
    out = {
        "inventory": inv.safe_public_summary(),
        "license_state": {},
        "import_action": "none_in_term01",
        "next_recommended_block": "CKA-TERM-02 controlled local import",
    }
    systems = (
        [TerminologySystem(args.system)]
        if args.system != "all"
        else [TerminologySystem.UMLS, TerminologySystem.SNOMED_CT,
              TerminologySystem.RXNORM, TerminologySystem.LOINC]
    )
    for s in systems:
        out["license_state"][s.value] = license_acknowledged_for(s)

    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
