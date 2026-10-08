#!/usr/bin/env python3
"""
extract_ensemble_summary.py — test-only harness for the bundle-based e2e test
(tests/run_e2e_bundle.sh).

Figs 13-19 are PNGs; visualize_ensemble_stats.py never writes an npz of the
binned curves it plots. To get a small, diffable numeric oracle for those
figures without reimplementing their math (which would test our copy, not
the pipeline), this script imports the real plotting module and calls its
own internal per-code aggregation functions (_extract_distance_curve,
_group_geomean, _group_arithmean_xlog, _group_logstd) on the same
gm_statistics.npz files regen_ensemble_figures.sh reads, reproducing the
exact numeric curves that end up in Figures 13A/13C and 17B.

Usage:
    python3 extract_ensemble_summary.py <production_runs_dir> <out.npz>
"""
import sys
from pathlib import Path

import numpy as np

UTILS = Path(__file__).resolve().parent.parent / "src" / "utils"
sys.path.insert(0, str(UTILS))
import visualize_ensemble_stats as ves  # noqa: E402

# Kept in sync with ALL_SCENARIOS in ../regen_ensemble_figures.sh (seissol/2
# intentionally excluded there too — it never reached Mw 7).
ALL_SCENARIOS = [
    "eqdyna/0001.A.100m", "eqdyna/0001.B.100m", "eqdyna/0001.C.100m",
    "fd3d/ncent.sd4", "fd3d/ncent.sd8", "fd3d/nleft.sd4", "fd3d/nleft.sd8",
    "fd3d/nright.sd4", "fd3d/nright.sd8",
    "mafe/1", "mafe/2", "mafe/3",
    "seissol/1", "seissol/3", "seissol/4", "seissol/5",
    "waveqlab3d/a24", "waveqlab3d/c24", "waveqlab3d/d24",
    "sord/1/sord_scenario",
    "specfem3d/1", "specfem3d/2", "specfem3d/3",
]

# Figs 13A/13C use RSA at T=3.0s and T=0.333s; also PGA (Fig 13-ish family)
# and CAV (Fig 19A). Figs 17/18 use the T=1.0s inter-event tau.
METRICS_FOR_DISTANCE = ["PGA", "CAV", "RSA_T_1_000"]
TAU_PERIOD = 1.0


def per_code_distance_groupmeans(scenarios, metric):
    """Reproduce the per-code bold group-mean curve plotted in
    plot_gm_metrics_vs_distance for one metric. Returns {code: (x, y)}."""
    is_acc = ves._is_acc_metric(metric)
    per_code_means = {}
    max_dist_km = 0.0
    for s in scenarios:
        try:
            data = ves.load_gm_statistics(s.gm_file)
        except FileNotFoundError:
            continue
        mean_key = f"{metric}_mean"
        if mean_key not in data.files:
            continue
        curve = ves._extract_distance_curve(
            data, mean_key, f"{metric}_count", f"{metric}_std",
            convert_to_g=is_acc)
        if curve is None:
            continue
        distances_km, means_sorted, _stds = curve
        if metric == "CAV":
            means_sorted = means_sorted / 981.0
        max_dist_km = max(max_dist_km, distances_km.max())
        code = ves._code_of(s.label)
        per_code_means.setdefault(code, []).append((distances_km, means_sorted))

    out = {}
    if max_dist_km <= 0:
        return out
    common_rjb = np.geomspace(1.0, max_dist_km, 60)
    for code in sorted(per_code_means):
        avg_x, avg_y = ves._group_geomean(per_code_means[code], common_rjb)
        if avg_x is not None and len(avg_x) >= 2:
            out[code] = (avg_x, avg_y)
    return out


def epistemic_tau_at_period(scenarios, period_s):
    """Reproduce the 'epistemic tau across groups' solid curve from
    plot_inter_event_std_vs_distance at one period. Returns (x, y) or None."""
    per_code_curves = {}
    for s in scenarios:
        try:
            data = ves.load_gm_statistics(s.gm_file)
        except FileNotFoundError:
            continue
        mean_key, _ = ves._match_rsa_period_key(data, period_s, tol=0.06)
        if mean_key is None:
            continue
        curve = ves._extract_distance_curve(
            data, mean_key, mean_key.replace("_mean", "_count"),
            convert_to_g=True)
        if curve is None:
            continue
        rjb_km, means, _ = curve
        ok = (rjb_km > 0) & (means > 0) & np.isfinite(means)
        if ok.sum() < 2:
            continue
        per_code_curves.setdefault(ves._code_of(s.label), []).append(
            (rjb_km[ok], means[ok]))

    if not per_code_curves:
        return None
    rjb_max = max(c[0].max() for code in per_code_curves for c in per_code_curves[code])
    common_rjb = np.geomspace(0.1, rjb_max, 120)

    per_code_groupmean = []
    for code in sorted(per_code_curves):
        avg_x, avg_y = ves._group_geomean(per_code_curves[code], common_rjb)
        if avg_x is not None and len(avg_x) >= 2:
            per_code_groupmean.append((avg_x, avg_y))
    if len(per_code_groupmean) < 2:
        return None
    return ves._group_logstd(per_code_groupmean, common_rjb, min_n=3)


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <production_runs_dir> <out.npz>", file=sys.stderr)
        sys.exit(2)
    prod_dir, out_path = sys.argv[1], sys.argv[2]

    scenarios = [ves.resolve_scenario_entry(s, input_dir=prod_dir) for s in ALL_SCENARIOS]

    save = {}
    for metric in METRICS_FOR_DISTANCE:
        groupmeans = per_code_distance_groupmeans(scenarios, metric)
        codes = sorted(groupmeans)
        save[f"{metric}__codes"] = np.array(codes)
        for code in codes:
            x, y = groupmeans[code]
            save[f"{metric}__{code}__x"] = x.astype(np.float64)
            save[f"{metric}__{code}__y"] = y.astype(np.float64)

    tau = epistemic_tau_at_period(scenarios, TAU_PERIOD)
    if tau is not None and tau[0] is not None:
        save["tau_T1__x"] = tau[0].astype(np.float64)
        save["tau_T1__y"] = tau[1].astype(np.float64)

    np.savez_compressed(out_path, **save)
    print(f"Wrote {out_path} ({len(save)} arrays)")


if __name__ == "__main__":
    main()
