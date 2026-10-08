#!/usr/bin/env python3
"""
plot_pergroup_ens_figure12.py — manuscript Figure 12 builder.

For each modeling group (=simulation code in DR4GM), produces one panel
showing SARotD50 at a target oscillator period vs Joyner-Boore distance:

  - per-station scatter (subset, ~1000 stations per simulation)
  - per-simulation median trend (dashed)
  - group-average trend (bold solid; geometric mean across that code's sims)
  - NGA-West2 GMM mean range (shaded) and ±1σ envelope (dashed)

Inputs per scenario:
    <input_dir>/<code>/<scenario>/ground_motion_metrics.npz
    <input_dir>/<code>/<scenario>/geometry.npz
    <input_dir>/<code>/<scenario>/gm_statistics.npz

Outputs:
    <output_dir>/SA_T<period>s_per_group_<code>.png   (one per code present)

Usage:
    python plot_pergroup_ens_figure12.py \
        --input-dir results/production_runs \
        --output-dir results/production_runs/ensemble_per_group \
        --period 1.0 --magnitude 7.0 --vs30 760 \
        eqdyna/0001.A.100m fd3d/ncent.sd4 mafe/1 ...

Or pass scenarios via JSON:
    python plot_pergroup_ens_figure12.py \
        --input-dir results/production_runs \
        --output-dir results/production_runs/ensemble_per_group \
        --scenarios-file scenarios.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from gm_stats import rjb_distances_m

try:
    from openquake_engine_gmpe import get_nga_west2_gmpe_predictions_cached as get_nga_west2_gmpe_predictions
    GMPE_AVAILABLE = True
except ImportError as exc:
    print(f"WARNING: NGA-West2 GMPE module not available: {exc}", file=sys.stderr)
    GMPE_AVAILABLE = False

from code_style import CODE_COLORS, CODE_DISPLAY_NAMES, code_of as _code_of, gmm_envelope as _gmm_envelope

CM_S2_PER_G = 981.0
SCATTER_PER_SIM = 1000


def _rjb_km(locations_m: np.ndarray, fault_start_m, fault_end_m) -> np.ndarray:
    """Thin wrapper: returns Rjb in km (no floor — scatter plot wants true distance)."""
    return rjb_distances_m(locations_m, fault_start_m, fault_end_m) / 1000.0


def _rsa_key_for_period(stats: np.lib.npyio.NpzFile, period_s: float):
    """Locate the RSA mean key in gm_statistics.npz matching the requested period."""
    target = float(period_s)
    best = None
    best_diff = np.inf
    for key in stats.files:
        if not (key.startswith('RSA_T_') and key.endswith('_mean')):
            continue
        try:
            stem = key[len('RSA_T_'):-len('_mean')]
            value = float(stem.replace('_', '.'))
        except ValueError:
            continue
        diff = abs(value - target)
        if diff < best_diff:
            best, best_diff = key, diff
    if best is None or best_diff > 1e-3:
        raise KeyError(f"No RSA_T_* mean key matches period={period_s:g}s")
    return best


def _load_scenario(input_dir: Path, label: str, period_s: float, rng: np.random.Generator):
    """Load per-scenario scatter + binned curve for one panel.

    Prefers per-station ``ground_motion_metrics.npz`` (gives the scatter cloud).
    Falls back to pre-binned ``gm_statistics.npz`` when the per-station file
    is unavailable (e.g. SORD shipped stats only) — scatter is empty in that
    case but the binned median curve still plots.
    """
    sd = input_dir / label
    gm_path = sd / 'ground_motion_metrics.npz'
    stats_path = sd / 'gm_statistics.npz'
    geom_path = sd / 'geometry.npz'
    if not geom_path.exists():
        raise FileNotFoundError(f"Missing {geom_path}")
    if not gm_path.exists() and not stats_path.exists():
        raise FileNotFoundError(f"Missing both {gm_path} and {stats_path}")

    geom = np.load(geom_path, allow_pickle=True)

    if gm_path.exists():
        gm = np.load(gm_path)
        periods = gm['periods']
        period_idx = int(np.argmin(np.abs(periods - period_s)))
        if abs(periods[period_idx] - period_s) > 1e-3:
            raise KeyError(f"{label}: period {period_s}s not in metrics (closest {periods[period_idx]})")
        sa_g = gm['SA'][:, period_idx] / CM_S2_PER_G
        locations = gm['locations']
        # Compute binned geomean directly from per-station data at 2 km resolution.
        # Using gm_statistics.npz (500 m bins) creates aliasing artifacts when the
        # station grid spacing is 1 km and single-station bins produce outliers.
        rjb_all_km = _rjb_km(locations, geom['fault_trace_start'], geom['fault_trace_end'])

        all_valid = (sa_g > 0) & np.isfinite(sa_g) & np.isfinite(rjb_all_km)
        rjb_v = rjb_all_km[all_valid]
        sa_v = sa_g[all_valid]

        rjb_scatter_km = rjb_v
        sa_g_scatter = sa_v
        if len(rjb_scatter_km) > SCATTER_PER_SIM:
            idx = rng.choice(len(rjb_scatter_km), SCATTER_PER_SIM, replace=False)
            rjb_scatter_km = rjb_scatter_km[idx]
            sa_g_scatter = sa_g_scatter[idx]

        bin_size_km = 2.0
        bin_edges = np.arange(0.0, rjb_v.max() + bin_size_km, bin_size_km)
        rjb_bin_km, sa_bin_g = [], []
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            mask = (rjb_v >= lo) & (rjb_v < hi)
            if mask.sum() < 5:
                continue
            rjb_bin_km.append(0.5 * (lo + hi))
            sa_bin_g.append(np.exp(np.mean(np.log(sa_v[mask]))))
        rjb_bin_km = np.array(rjb_bin_km)
        sa_bin_g = np.array(sa_bin_g)
    else:
        # Fallback: stats-only (e.g. SORD). No per-station scatter; just the
        # binned median line from gm_statistics.npz. ``rjb_distance_bins`` is
        # already bin-center values in meters; ``RSA_T_*_mean`` is in cm/s².
        stats = np.load(stats_path)
        rjb_centers_km = stats['rjb_distance_bins'].astype(float) / 1000.0
        target = float(period_s)
        best_key, best_diff = None, np.inf
        for key in stats.files:
            if not (key.startswith('RSA_T_') and key.endswith('_mean')):
                continue
            try:
                val = float(key[len('RSA_T_'):-len('_mean')].replace('_', '.'))
            except ValueError:
                continue
            if abs(val - target) < best_diff:
                best_key, best_diff = key, abs(val - target)
        if best_key is None or best_diff > 1e-3:
            raise KeyError(f"{label}: RSA_T_{period_s}s not in {stats_path}")
        sa_mean_g = stats[best_key].astype(float) / CM_S2_PER_G  # cm/s² → g
        count_key = best_key.replace('_mean', '_count')
        counts = stats[count_key] if count_key in stats.files else np.ones_like(sa_mean_g)
        valid = (sa_mean_g > 0) & np.isfinite(sa_mean_g) & (counts > 0)
        rjb_bin_km = rjb_centers_km[valid]
        sa_bin_g = sa_mean_g[valid]
        rjb_scatter_km = np.array([])
        sa_g_scatter = np.array([])

    return {
        'label': label,
        'rjb_km_scatter': rjb_scatter_km,
        'sa_g_scatter': sa_g_scatter,
        'rjb_km_bin': rjb_bin_km,
        'sa_g_bin': sa_bin_g,
    }


def _group_geomean(scenarios, common_rjb_km):
    """Geometric mean of per-scenario binned trends, interpolated in log-log
    onto a common rjb grid. Bins outside a scenario's own range are excluded
    from that scenario's contribution (no extrapolation)."""
    log_x_common = np.log(common_rjb_km)
    log_y_stack = []
    for sd in scenarios:
        x = sd['rjb_km_bin']
        y = sd['sa_g_bin']
        if len(x) < 2:
            continue
        log_x = np.log(x)
        log_y = np.log(y)
        order = np.argsort(log_x)
        log_x, log_y = log_x[order], log_y[order]
        log_y_at = np.interp(log_x_common, log_x, log_y, left=np.nan, right=np.nan)
        log_y_stack.append(log_y_at)
    if not log_y_stack:
        return None, None
    arr = np.vstack(log_y_stack)
    with np.errstate(invalid='ignore'):
        avg_log_y = np.nanmean(arr, axis=0)
    keep = np.isfinite(avg_log_y)
    return common_rjb_km[keep], np.exp(avg_log_y[keep])


def _add_gmpe(ax, period_s, rjb_min_km, rjb_max_km, magnitude, vs30):
    if not GMPE_AVAILABLE:
        return
    dist_km = np.geomspace(max(rjb_min_km, 0.5), max(rjb_max_km, 1.0) * 1.05, 80)
    try:
        results = get_nga_west2_gmpe_predictions(dist_km, magnitude, [period_s], vs30)
    except Exception as exc:
        print(f"WARNING: GMPE call failed: {exc}", file=sys.stderr)
        return
    per_period = results.get(period_s)
    if per_period is None:
        return
    # ±τ envelope: inter-event std brackets each event's median (FORMULAS.md §7.1).
    means, upper, lower = _gmm_envelope(per_period, stat_key='tau')
    if means is None:
        return
    ax.fill_between(dist_km, means.min(axis=0), means.max(axis=0),
                    color='black', alpha=0.20, label='GMM mean range', zorder=0)
    ax.plot(dist_km, upper, '--', color='black', linewidth=1.4, alpha=0.7,
            label='GMM ±1τ (inter-event)', zorder=0)
    ax.plot(dist_km, lower, '--', color='black', linewidth=1.4, alpha=0.7, zorder=0)


def plot_one_group(code, scenarios, period_s, magnitude, vs30, output_dir,
                   add_gmpe, ylim=None, xlim=None):
    color = CODE_COLORS.get(code, 'tab:blue')
    fig, ax = plt.subplots(figsize=(8, 6))

    rjb_lo = np.inf
    rjb_hi = -np.inf
    for sd in scenarios:
        ax.scatter(sd['rjb_km_scatter'], sd['sa_g_scatter'],
                   s=10, alpha=0.15, color=color, edgecolors='none', zorder=1)
        ax.plot(sd['rjb_km_bin'], sd['sa_g_bin'], '--',
                color=color, alpha=0.75, linewidth=2.0, zorder=2)
        if len(sd['rjb_km_bin']):
            rjb_lo = min(rjb_lo, float(np.min(sd['rjb_km_bin'])))
            rjb_hi = max(rjb_hi, float(np.max(sd['rjb_km_bin'])))

    if not np.isfinite(rjb_lo) or not np.isfinite(rjb_hi) or rjb_hi <= rjb_lo:
        plt.close(fig)
        raise RuntimeError(f"{code}: no valid Rjb range across {len(scenarios)} scenarios")

    common_rjb = np.geomspace(max(rjb_lo, 0.1), rjb_hi, 60)
    avg_x, avg_y = _group_geomean(scenarios, common_rjb)
    display_name = CODE_DISPLAY_NAMES.get(code, code)
    if avg_x is not None:
        ax.plot(avg_x, avg_y, '-', color=color, linewidth=3.5,
                label=f'{display_name} group mean ({len(scenarios)} sims)', zorder=4)

    if add_gmpe:
        gmpe_lo = xlim[0] if xlim is not None else rjb_lo
        gmpe_hi = xlim[1] if xlim is not None else rjb_hi
        _add_gmpe(ax, period_s, gmpe_lo, gmpe_hi, magnitude, vs30)

    ax.set_xscale('log')
    ax.set_yscale('log')
    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])
    else:
        ax.set_xlim(max(rjb_lo, 0.5), 18.0)
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])
    ax.set_xlabel('Rupture Distance (km)', fontsize=13, fontweight='bold')
    ax.set_ylabel(f'SA(T={period_s:g}s) (g)', fontsize=13, fontweight='bold')
    ax.set_title(display_name, fontsize=14, fontweight='bold')
    ax.grid(True, which='both', alpha=0.25)
    ax.tick_params(labelsize=11)
    ax.legend(fontsize=10, loc='lower left', framealpha=0.9)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"SA_T{period_s:.3f}s_per_group_{code}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return out_path


def _load_labels_from_files(paths):
    out = []
    for p in paths:
        with open(p) as fh:
            data = json.load(fh)
        if isinstance(data, list):
            out.extend(data)
        elif isinstance(data, dict) and 'scenarios' in data:
            out.extend(data['scenarios'])
        else:
            raise ValueError(f"{p}: expected list or dict with 'scenarios' key")
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input-dir', required=True,
                        help='Directory containing <code>/<scenario>/ subdirs')
    parser.add_argument('--output-dir', required=True,
                        help='Directory to write per-group panels into')
    parser.add_argument('--period', type=float, default=1.0,
                        help='Spectral period in seconds (default: 1.0)')
    parser.add_argument('--magnitude', type=float, default=7.0,
                        help='Mw for GMM predictions (default: 7.0)')
    parser.add_argument('--vs30', type=float, default=760.0,
                        help='Vs30 in m/s for GMM predictions (default: 760)')
    parser.add_argument('--no-gmpe', action='store_true',
                        help='Skip GMM band overlay')
    parser.add_argument('--seed', type=int, default=0,
                        help='RNG seed for station subsampling (default: 0)')
    parser.add_argument('--ylim', type=float, nargs=2, default=None, metavar=('YMIN', 'YMAX'),
                        help='Shared y-axis SA(T) limits in g across all panels '
                             '(e.g. --ylim 0.003 2.0)')
    parser.add_argument('--xlim', type=float, nargs=2, default=None, metavar=('XMIN', 'XMAX'),
                        help='Shared x-axis Rjb limits in km across all panels '
                             '(e.g. --xlim 0.5 40)')
    parser.add_argument('--scenarios-file', action='append', default=[],
                        help='JSON file with scenarios (may repeat)')
    parser.add_argument('scenarios', nargs='*',
                        help='Scenario labels of form <code>/<scenario>')
    args = parser.parse_args(argv)

    labels = list(args.scenarios)
    labels.extend(_load_labels_from_files(args.scenarios_file))
    if not labels:
        parser.error('No scenarios provided (positional or via --scenarios-file)')

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    rng = np.random.default_rng(args.seed)

    groups: dict = {}
    for label in labels:
        groups.setdefault(_code_of(label), []).append(label)

    written = []
    for code in sorted(groups):
        per_code_scenarios = []
        for label in groups[code]:
            per_code_scenarios.append(_load_scenario(input_dir, label, args.period, rng))
        ylim = tuple(args.ylim) if args.ylim else None
        xlim = tuple(args.xlim) if args.xlim else None
        out = plot_one_group(code, per_code_scenarios, args.period,
                             args.magnitude, args.vs30, output_dir,
                             add_gmpe=not args.no_gmpe,
                             ylim=ylim, xlim=xlim)
        print(f"wrote {out}")
        written.append(out)

    if not written:
        sys.exit("No panels written")


if __name__ == '__main__':
    main()
