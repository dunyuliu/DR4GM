#!/usr/bin/env python3
"""
compare_e2e_summary.py -- diff two ensemble-summary npz files produced by
extract_ensemble_summary.py, for test_system/run_e2e_bundle.sh.

Same precision convention as diff_gm_metrics.py: rel 1e-6 (the summary is
derived from float32 station velocities, so float64 bit-identity is not
expected run-to-run on different machines/BLAS builds).

NaN-safe by construction: the pass condition is `worst_rel <= TOL`, not
`not (worst_rel > TOL)` -- a NaN relative error must FAIL, and `nan <= TOL`
is already False, so no special-casing is needed as long as the comparison
is written this way round.
"""
import sys

import numpy as np

TOL = 1e-6


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <ref.npz> <new.npz>", file=sys.stderr)
        return 2
    ref_path, new_path = sys.argv[1], sys.argv[2]
    ref = np.load(ref_path)
    new = np.load(new_path)

    ref_keys, new_keys = set(ref.files), set(new.files)
    problems = []
    worst_rel = 0.0
    hard_fail = False  # structural mismatches that make a rel-error meaningless

    if ref_keys != new_keys:
        missing = sorted(ref_keys - new_keys)
        extra = sorted(new_keys - ref_keys)
        if missing:
            problems.append(f"MISSING keys in new output: {missing}")
        if extra:
            problems.append(f"UNEXPECTED new keys: {extra}")
        hard_fail = True

    for k in sorted(ref_keys & new_keys):
        a, b = ref[k], new[k]
        if a.dtype.kind not in "fc":
            if not np.array_equal(a, b):
                problems.append(f"{k}: non-numeric mismatch (e.g. code set changed)")
                hard_fail = True
            continue
        if a.shape != b.shape:
            problems.append(f"{k}: shape mismatch {a.shape} vs {b.shape}")
            hard_fail = True
            continue
        diff = np.abs(a - b)
        denom = np.maximum(np.abs(a), np.abs(b))
        rel = np.where(denom > 0, diff / denom, 0.0)
        max_rel = rel.max() if rel.size else 0.0
        if not np.isfinite(max_rel):
            problems.append(f"{k}: non-finite relative error (NaN/Inf present)")
            hard_fail = True
            continue
        worst_rel = max(worst_rel, max_rel)
        if max_rel > TOL:
            problems.append(f"{k}: max_rel={max_rel:.3e} (tol={TOL:.0e})")

    worst_rel = float("nan") if hard_fail else worst_rel
    passed = worst_rel <= TOL  # False when worst_rel is NaN -- fails closed.
    print(f"E2E summary diff: worst_rel={worst_rel:.3e} tol={TOL:.0e} -> "
          f"{'PASS' if passed else 'FAIL'}")
    for p in problems:
        print(f"  {p}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
