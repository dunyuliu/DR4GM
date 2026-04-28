#!/usr/bin/env python3
"""
diff_gm_metrics.py — diff two ground_motion_metrics.npz files element-wise.

Usage: python diff_gm_metrics.py <ref.npz> <new.npz> [--input <processed_stations.npz>]

Reports max absolute and max relative error for every numeric array key
present in the reference. Tolerance is precision-aware: float32 input
velocities yield CAV at float32 epsilon (~1e-7 rel), so PASS threshold is
1e-6 rel when --input is float32 and 1e-12 rel otherwise.
"""
import sys
import numpy as np

args = sys.argv[1:]
input_npz = None
if "--input" in args:
    j = args.index("--input")
    input_npz = args[j + 1]
    args = args[:j] + args[j + 2:]
if len(args) != 2:
    print("Usage: diff_gm_metrics.py <ref.npz> <new.npz> [--input <processed_stations.npz>]")
    sys.exit(2)

ref_path, new_path = args
ref = np.load(ref_path)
new = np.load(new_path)

TOL = 1e-12
if input_npz is not None:
    ips = np.load(input_npz)
    for vk in ("vel_strike", "vel_x", "vel_normal", "vel_y"):
        if vk in ips.files and ips[vk].dtype == np.float32:
            TOL = 1e-6
            break

worst_rel = 0.0
lines = []
for k in ref.files:
    if k not in new.files:
        lines.append(f"  MISSING in new: {k}")
        worst_rel = float("inf")
        continue
    a = np.asarray(ref[k])
    b = np.asarray(new[k])
    if a.dtype.kind not in "fc" and b.dtype.kind not in "fc":
        if not np.array_equal(a, b):
            lines.append(f"  {k}: non-numeric mismatch")
            worst_rel = float("inf")
        continue
    if a.shape != b.shape:
        lines.append(f"  {k}: shape mismatch {a.shape} vs {b.shape}")
        worst_rel = float("inf")
        continue
    diff = np.abs(a - b)
    max_abs = diff.max() if diff.size else 0.0
    denom = np.maximum(np.abs(a), np.abs(b))
    rel = np.where(denom > 0, diff / denom, 0.0)
    max_rel = rel.max() if rel.size else 0.0
    worst_rel = max(worst_rel, max_rel)
    lines.append(f"  {k}: max_abs={max_abs:.3e} max_rel={max_rel:.3e}")

verdict = "PASS" if worst_rel <= TOL else "FAIL"
print(f"DIFF {verdict}  worst_rel={worst_rel:.3e}  (tol={TOL:.0e})")
for line in lines:
    print(line)
sys.exit(0 if worst_rel <= TOL else 1)
