"""Generate the solver-correctness baseline JSON.

Usage:
    python -m tests.baselines.generate --output tests/baselines/2026-05-02-yourdfpy-solver-baseline.json

The output is committed to git. Regeneration is allowed when a step's
spec justifies it (per-step decision recorded in that step's design doc).
Existing baseline files stay committed for traceability.
"""
from __future__ import annotations
import argparse
import json
import sys

from tests.baselines.cases import CASES
from tests.baselines.runner import run_case


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--label", required=True, help="baseline_label written into the JSON")
    args = ap.parse_args()

    results = [run_case(c) for c in CASES]
    unsolved = [r["name"] for r in results if not r["solved"]]
    if unsolved:
        print(f"WARN: unsolved cases (still saved): {unsolved}", file=sys.stderr)

    payload = {
        "schema_version": 1,
        "baseline_label": args.label,
        "cases": results,
    }
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    print(f"wrote {args.output} ({len(results)} cases)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
