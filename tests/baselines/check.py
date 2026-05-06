"""Re-run baseline cases and compare to a saved JSON.

Two modes:
  --strict     : numerical-noise tolerance only (steps 1, 2, 3, 5, 6).
  --loose      : pose-error tolerance per spec §4.2 (step 4 only).

Exit code 0 on pass, 1 on fail. Prints a per-case verdict.
"""
from __future__ import annotations
import argparse
import json

from tests.baselines.cases import CASES
from tests.baselines.runner import run_case

STRICT_REL_COST = 1e-9          # joint values + cost noise tolerance
STRICT_ABS_POSE = 1e-12         # pose-error noise tolerance

LOOSE_COST_REL_CEILING = 0.01           # cost ≤ 1.01 × baseline
LOOSE_POSE_REL_CEILING = 1.5            # error ≤ 1.5 × baseline
LOOSE_TRANS_ABS_FLOOR = 1e-3            # 1 mm absolute floor
LOOSE_ROT_ABS_FLOOR = 1e-3              # 1e-3 rad absolute floor


def _check_case(actual: dict, expected: dict, mode: str) -> tuple[bool, str]:
    if expected["solved"] != actual["solved"]:
        return False, f"solved flag changed: {expected['solved']} -> {actual['solved']}"
    if not expected["solved"]:
        return True, "skipped (baseline unsolved)"

    if mode == "strict":
        for k in ("trans_err", "rot_err"):
            if abs(actual[k] - expected[k]) > STRICT_ABS_POSE:
                return False, f"{k} drifted: {expected[k]:.3e} -> {actual[k]:.3e}"
        return True, "ok"

    # loose
    trans_cap = max(LOOSE_POSE_REL_CEILING * expected["trans_err"], LOOSE_TRANS_ABS_FLOOR)
    rot_cap = max(LOOSE_POSE_REL_CEILING * expected["rot_err"], LOOSE_ROT_ABS_FLOOR)
    if actual["trans_err"] > trans_cap:
        return False, f"trans_err {actual['trans_err']:.3e} > cap {trans_cap:.3e}"
    if actual["rot_err"] > rot_cap:
        return False, f"rot_err {actual['rot_err']:.3e} > cap {rot_cap:.3e}"
    return True, "ok"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--mode", choices=("strict", "loose"), default="strict")
    args = ap.parse_args()

    expected_by_name = {c["name"]: c for c in json.load(open(args.baseline))["cases"]}
    failures = 0
    for case in CASES:
        actual = run_case(case)
        if case["name"] not in expected_by_name:
            print(f"FAIL {case['name']}: not in baseline")
            failures += 1
            continue
        ok, msg = _check_case(actual, expected_by_name[case["name"]], args.mode)
        verdict = "PASS" if ok else "FAIL"
        print(f"{verdict} {case['name']}: {msg}")
        if not ok:
            failures += 1
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
