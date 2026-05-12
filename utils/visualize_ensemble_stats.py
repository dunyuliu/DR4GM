#!/usr/bin/env python3
"""
Simple plotting script for ground motion scenarios - plots raw values as-is
Usage: python plot_gm_scenarios.py [scenario1 scenario2 ...] [--scenarios-file scenarios.json]
You may provide multiple JSON files via repeated --scenarios-file flags.
Each JSON file should either be a list of scenario paths or contain a
"scenarios" key whose value is that list. Use --min-mean to drop bins with
means at or below a threshold (default keeps all positive values).

GMPE Support:
This script supports GMPE (Ground Motion Prediction Equation) comparisons using OpenQuake.
To enable GMPE plotting, install OpenQuake:
  pip install openquake.engine
  
GMPE options:
  --add-gmpe        : Add GMPE comparison curves and GMM φ band on intra-event std plots
  --magnitude M     : Set magnitude for GMPE predictions (default: 7.0)
  --vs30 VS30       : Set Vs30 for GMPE predictions in m/s (default: 760)
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

# Try to import OpenQuake Engine GMPE functions - requires OpenQuake Engine
# Install with: pip install openquake.engine
try:
    from openquake_engine_gmpe import get_nga_west2_gmpe_predictions, get_cav_gmm_predictions
    PLOT_GMPE_AVAILABLE = True
    print("OpenQuake Engine GMPE functions successfully loaded")
except ImportError as e:
    print(f"WARNING: OpenQuake Engine GMPE functions not available: {e}")
    print("GMPE plotting disabled. To enable: pip install openquake.engine")
    PLOT_GMPE_AVAILABLE = False


def load_gm_statistics(gm_file):
    """Load gm_statistics.npz file for a scenario"""
    gm_file = Path(gm_file)
    if not gm_file.exists():
        raise FileNotFoundError(f"GM statistics file not found: {gm_file}")
    return np.load(gm_file)


_CODE_COLORS = {
    'eqdyna':     'tab:blue',
    'fd3d':       'tab:orange',
    'mafe':       'tab:green',
    'seissol':    'tab:red',
    'waveqlab3d': 'tab:purple',
    'specfem3d':  'tab:brown',
}

_CODE_AUTHOR_LABELS = {
    'waveqlab3d': 'Withers',
    'seissol':    'Ulrich/Gabriel',
    'sord':       'Wang',
    'eqdyna':     'Liu/Duan',
    'mafe':       'Ma',
    'specfem3d':  'Oral/Ampuero/Asimaki',
    'fd3d':       'Gallovič/Valentová',
}


def _code_of(label):
    return label.split('/', 1)[0] if '/' in label else label


def _code_color(label):
    return _CODE_COLORS.get(_code_of(label), 'tab:gray')


def _is_acc_metric(metric):
    return metric == 'PGA' or metric.startswith('RSA_T_')


def _rsa_units(stats):
    return str(stats['RSA_units']) if 'RSA_units' in stats.files else 'g'


def _valid_bin_mask(counts, min_frac=0.05):
    """Bins with count >= max(1, min_frac * counts.max()) — drops sparse far-edge tail."""
    counts = np.asarray(counts)
    if counts.size == 0:
        return counts.astype(bool)
    threshold = max(1, int(np.ceil(min_frac * counts.max())))
    return counts >= threshold


def _interp_log(x, y, x_target, log_y):
    """Interpolate y vs log(x) onto log(x_target). If ``log_y`` is True, both
    y and the interpolated result are taken in ln space. Returns NaN outside
    the input x range. Inputs assumed positive (x always; y when log_y)."""
    log_x = np.log(np.asarray(x, dtype=float))
    y_arr = np.log(y) if log_y else np.asarray(y, dtype=float)
    order = np.argsort(log_x)
    return np.interp(np.log(x_target), log_x[order], y_arr[order],
                     left=np.nan, right=np.nan)


def _stack_curves(curves, x_target, log_y):
    """Stack interpolated curves as rows; drops any per-curve with <2 valid pts."""
    stack = []
    for x, y in curves:
        x = np.asarray(x); y = np.asarray(y)
        pos = (y > 0) if log_y else np.ones_like(y, dtype=bool)
        ok = (x > 0) & np.isfinite(x) & pos & np.isfinite(y)
        if ok.sum() < 2:
            continue
        stack.append(_interp_log(x[ok], y[ok], x_target, log_y))
    return np.vstack(stack) if stack else None


def _nan_reduce(func, arr, **kw):
    with np.errstate(invalid='ignore'), warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return func(arr, axis=0, **kw)


def _group_geomean(curves, x_target):
    """Geometric mean across curves at each x_target."""
    arr = _stack_curves(curves, x_target, log_y=True)
    if arr is None:
        return None, None
    avg = _nan_reduce(np.nanmean, arr)
    keep = np.isfinite(avg)
    return x_target[keep], np.exp(avg[keep])


def _group_arithmean_xlog(curves, x_target):
    """Arithmetic mean across curves at each x_target (log-x interpolation)."""
    arr = _stack_curves(curves, x_target, log_y=False)
    if arr is None:
        return None, None
    avg = _nan_reduce(np.nanmean, arr)
    keep = np.isfinite(avg)
    return x_target[keep], avg[keep]


def _group_logstd(curves, x_target, min_n=2):
    """Inter-curve sample std in ln(y) units (ddof=1) — used for inter-event tau."""
    min_n = max(min_n, 2)
    arr = _stack_curves(curves, x_target, log_y=True)
    if arr is None or arr.shape[0] < min_n:
        return None, None
    n_finite = np.sum(np.isfinite(arr), axis=0)
    std = _nan_reduce(np.nanstd, arr, ddof=1)
    keep = (n_finite >= min_n) & np.isfinite(std)
    return x_target[keep], std[keep]


def _iter_rsa_period_keys(stats):
    """Yield (mean_key, period_s) for every RSA_T_*_mean in the npz."""
    prefix, suffix = 'RSA_T_', '_mean'
    for k in stats.files:
        if not (k.startswith(prefix) and k.endswith(suffix)):
            continue
        try:
            yield k, float(k[len(prefix):-len(suffix)].replace('_', '.'))
        except ValueError:
            continue


def _match_rsa_period_key(stats, period_s, tol=0.06):
    """Return ('RSA_T_<...>_mean', period_value) closest to ``period_s`` within
    ``tol`` seconds, or (None, None) if no key matches."""
    target = float(period_s)
    best_key, best_val, best_d = None, None, np.inf
    for k, val in _iter_rsa_period_keys(stats):
        d = abs(val - target)
        if d < best_d:
            best_key, best_val, best_d = k, val, d
    if best_key is None or best_d > tol:
        return None, None
    return best_key, best_val


def _extract_distance_curve(stats, mean_key, count_key, std_key=None,
                            convert_to_g=False):
    """Read (rjb_km, mean[, std]) from a gm_statistics.npz, applying the
    standard valid-bin mask and optional cm/s² → g conversion. Returns
    None if no valid bins remain."""
    valid = _valid_bin_mask(stats[count_key])
    if not valid.any():
        return None
    rjb_km = stats['rjb_distance_bins'].astype(float)[valid] / 1000.0
    means = stats[mean_key].astype(float)[valid]
    if convert_to_g and _rsa_units(stats) == 'cm/s²':
        means = means / 981.0
    stds = stats[std_key].astype(float)[valid] if std_key is not None else None
    order = np.argsort(rjb_km)
    return (rjb_km[order], means[order],
            None if stds is None else stds[order])


def _extract_periods_at_rjb(stats, target_rjb_km, want_std=False):
    """At the bin closest to ``target_rjb_km``, return arrays
    (periods_s, sa_g[, std]) sorted by period for every RSA_T_* with count > 0."""
    rjb_km = stats['rjb_distance_bins'].astype(float) / 1000.0
    idx = int(np.argmin(np.abs(rjb_km - target_rjb_km)))
    is_cm = _rsa_units(stats) == 'cm/s²'
    periods, sa_g, stds = [], [], []
    for mean_key, T in _iter_rsa_period_keys(stats):
        if stats[mean_key.replace('_mean', '_count')][idx] <= 0:
            continue
        v = float(stats[mean_key][idx])
        if is_cm:
            v = v / 981.0
        if v <= 0 or not np.isfinite(v):
            continue
        periods.append(T)
        sa_g.append(v)
        if want_std:
            stds.append(float(stats[mean_key.replace('_mean', '_std')][idx]))
    if not periods:
        return None
    order = np.argsort(periods)
    p_arr = np.asarray(periods)[order]
    s_arr = np.asarray(sa_g)[order]
    if want_std:
        return p_arr, s_arr, np.asarray(stds)[order]
    return p_arr, s_arr


_METRIC_UNITS = {'PGA': 'g', 'PGV': 'cm/s', 'PGD': 'cm', 'CAV': 'g·s'}


def _metric_unit(metric):
    if metric in _METRIC_UNITS:
        return _METRIC_UNITS[metric]
    return 'g' if metric.startswith('RSA_T_') else ''


def _compute_global_yrange(scenarios, mean_key, count_key, distance_range,
                           convert_to_g):
    """Aggregate valid means across scenarios → (y_min, y_max) with 10%
    log-space padding. Returns None if no data."""
    all_means = []
    for s in scenarios:
        try:
            data = load_gm_statistics(s.gm_file)
        except Exception:
            continue
        if mean_key not in data.files:
            continue
        valid = _valid_bin_mask(data[count_key])
        distances = data['rjb_distance_bins'][valid]
        means = data[mean_key][valid]
        if convert_to_g and _rsa_units(data) == 'cm/s²':
            means = means / 981.0
        if distance_range is not None:
            keep = (distances >= distance_range[0]) & (distances <= distance_range[1])
            means = means[keep]
        if means.size:
            all_means.append(means)
    if not all_means:
        return None
    flat = np.concatenate(all_means)
    log_lo, log_hi = np.log10(flat.min()), np.log10(flat.max())
    pad = (log_hi - log_lo) * 0.1
    return 10 ** (log_lo - pad), 10 ** (log_hi + pad)


@dataclass
class ScenarioInfo:
    source: str
    base_path: Path
    gm_file: Path
    label: str


def resolve_scenario_entry(raw_scenario, input_dir='results'):
    """Normalize a scenario argument to a ScenarioInfo"""
    scenario_path = Path(raw_scenario).expanduser()
    input_dir_path = Path(input_dir)

    if scenario_path.is_file():
        if scenario_path.name != 'gm_statistics.npz':
            raise ValueError(f"Scenario file must be gm_statistics.npz: {raw_scenario}")
        base_path = scenario_path.parent
        gm_file = scenario_path
    else:
        base_path = scenario_path
        gm_file = base_path / 'gm_statistics.npz'

    if not base_path.is_absolute():
        base_path = input_dir_path / base_path
        gm_file = base_path / 'gm_statistics.npz'
    else:
        gm_file = base_path / 'gm_statistics.npz'

    try:
        label = str(base_path.relative_to(input_dir_path))
    except ValueError:
        label = str(base_path)

    return ScenarioInfo(source=str(raw_scenario), base_path=base_path, gm_file=gm_file, label=label)

def plot_gm_metrics_vs_distance(scenarios, output_dir="results", distance_range=None, min_mean=0.0, add_gmpe=True, magnitude=7.0, vs30=760.0):
    """Plot GM metrics vs distance - raw values as stored"""
    
    os.makedirs(output_dir, exist_ok=True)

    metrics = ['PGA', 'PGV', 'PGD', 'CAV']
    global_ranges = {}
    for metric in metrics:
        rng = _compute_global_yrange(
            scenarios, f'{metric}_mean', f'{metric}_count',
            distance_range, convert_to_g=_is_acc_metric(metric))
        if rng is not None:
            if metric == 'CAV':
                rng = (rng[0] / 981.0, rng[1] / 981.0)
            global_ranges[metric] = rng
    
    # Create individual plots for each metric
    for metric in metrics:
        fig_mean, ax_mean = plt.subplots(figsize=(8, 6))
        fig_cov, ax_cov = plt.subplots(figsize=(8, 6))

        max_dist_km = 0.0
        per_code_means = {}
        per_code_stds = {}
        is_acc = _is_acc_metric(metric)
        for scenario_info in scenarios:
            try:
                data = load_gm_statistics(scenario_info.gm_file)
            except Exception as e:
                print(f"Warning: Could not load {scenario_info.source}: {e}")
                continue
            if f'{metric}_mean' not in data.files:
                print(f"Info: Scenario {scenario_info.label} does not have {metric}, skipping")
                continue
            curve = _extract_distance_curve(
                data, f'{metric}_mean', f'{metric}_count', f'{metric}_std',
                convert_to_g=is_acc)
            if curve is None:
                print(f"Warning: Scenario {scenario_info.label} has no {metric} data; skipping")
                continue
            distances_km, means_sorted, stds_sorted = curve
            if metric == 'CAV':
                means_sorted = means_sorted / 981.0
            max_dist_km = max(max_dist_km, distances_km.max())

            code = _code_of(scenario_info.label)
            color = _code_color(scenario_info.label)
            ax_mean.plot(distances_km, means_sorted, color=color,
                         linestyle='--', alpha=0.35, linewidth=1.0,
                         label='_nolegend_')
            ax_cov.plot(distances_km, stds_sorted, color=color,
                        linestyle='--', alpha=0.35, linewidth=1.0,
                        label='_nolegend_')
            per_code_means.setdefault(code, []).append((distances_km, means_sorted))
            per_code_stds.setdefault(code, []).append((distances_km, stds_sorted))

        # Per-code group-mean bold solid line (geometric mean across that code's sims)
        if max_dist_km > 0:
            common_rjb = np.geomspace(1.0, max_dist_km, 60)
            for code in sorted(per_code_means):
                color = _CODE_COLORS.get(code, 'tab:gray')
                avg_x, avg_y = _group_geomean(per_code_means[code], common_rjb)
                if avg_x is not None and avg_y is not None and len(avg_x) >= 2:
                    ax_mean.plot(avg_x, avg_y, color=color, linewidth=3.0,
                                 label=f'{code} ({len(per_code_means[code])})')
                std_x, std_y = _group_arithmean_xlog(per_code_stds[code], common_rjb)
                if std_x is not None and std_y is not None and len(std_x) >= 2:
                    ax_cov.plot(std_x, std_y, color=color, linewidth=3.0,
                                label=f'{code} ({len(per_code_stds[code])})')

        # Add GMPE curves for PGA and RSA metrics only
        if add_gmpe and PLOT_GMPE_AVAILABLE and (metric == 'PGA' or metric.startswith('RSA_T_')):
            try:
                # Create distance array for smooth GMPE curves — cap at max
                # observed Rjb across the plotted scenarios so the GMPE doesn't
                # extend past where the simulation data ends.
                if distance_range:
                    dist_min, dist_max = distance_range[0] / 1000.0, distance_range[1] / 1000.0
                else:
                    dist_min = 1.0
                    dist_max = max_dist_km if max_dist_km > dist_min else 100.0
                gmpe_distances = np.logspace(np.log10(dist_min), np.log10(dist_max), 50)
                
                # Extract period from metric name
                if metric == 'PGA':
                    periods = [0.01]  # Use special flag for direct PGA prediction
                    period = 0.01
                else:
                    # Extract period from metric name (e.g., RSA_T_1_000 -> 1.0)
                    period_str = metric.replace('RSA_T_', '').replace('_', '.')
                    period = float(period_str)
                    periods = [period]
                
                # Get GMPE predictions from external openquake_engine_gmpe module
                gmpe_results = get_nga_west2_gmpe_predictions(gmpe_distances, magnitude, periods, vs30)
                
                if period in gmpe_results:
                    data = gmpe_results[period]
                    gmpe_distances_km = data['distances']
                    ax_mean.loglog(gmpe_distances_km, data['NGA_AVG']['mean'], 'black', linewidth=3, label=f'NGA-West2 Avg (M{magnitude})')
                    
                    # Black shaded area: range of 4 GMPE means
                    all_means = []
                    for gmpe in ['ASK', 'BSSA', 'CB', 'CY']:
                        all_means.append(data[gmpe]['mean'])
                    
                    means_upper = np.max(all_means, axis=0)
                    means_lower = np.min(all_means, axis=0)
                    
                    ax_mean.fill_between(gmpe_distances_km, means_lower, means_upper, 
                                       color='black', alpha=0.2, label='GMPE Mean Range')
                    
                    # Black dashed lines: ±1σ envelope from all 4 GMPEs
                    all_upper = []
                    all_lower = []
                    for gmpe in ['ASK', 'BSSA', 'CB', 'CY']:
                        upper = data[gmpe]['mean'] * np.exp(data[gmpe]['std'])
                        lower = data[gmpe]['mean'] * np.exp(-data[gmpe]['std'])
                        all_upper.append(upper)
                        all_lower.append(lower)
                    
                    envelope_upper = np.max(all_upper, axis=0)
                    envelope_lower = np.min(all_lower, axis=0)
                    
                    ax_mean.loglog(gmpe_distances_km, envelope_upper, '--', color='black',
                                   linewidth=2, alpha=0.7, label='GMPE ±1σ envelope')
                    ax_mean.loglog(gmpe_distances_km, envelope_lower, '--', color='black', linewidth=2, alpha=0.7)

                    # φ band on std plot (only for acceleration metrics)
                    if is_acc:
                        all_phi = [data[g]['phi'] for g in ('ASK', 'BSSA', 'CB', 'CY') if 'phi' in data.get(g, {})]
                        if all_phi:
                            phi_lo = np.min(all_phi, axis=0)
                            phi_hi = np.max(all_phi, axis=0)
                            ax_cov.fill_between(gmpe_distances_km, phi_lo, phi_hi,
                                                color='grey', alpha=0.4,
                                                label='GMM φ range', zorder=0)

                    print(f"Added GMPE curve for {metric} (M{magnitude}, Vs30={vs30})")
                    
            except Exception as e:
                print(f"Warning: Could not add GMPE curves for {metric}: {e}")
        elif add_gmpe and metric == 'CAV' and PLOT_GMPE_AVAILABLE:
            try:
                if distance_range:
                    dist_min, dist_max = distance_range[0] / 1000.0, distance_range[1] / 1000.0
                else:
                    dist_min = 1.0
                    dist_max = max_dist_km if max_dist_km > dist_min else 100.0
                gmpe_distances_km = np.logspace(np.log10(dist_min), np.log10(dist_max), 50)
                cav_pred = get_cav_gmm_predictions(gmpe_distances_km, magnitude, vs30)
                cb_mean_g_s = cav_pred['CB']['mean']
                cb_std_ln = cav_pred['CB']['std']
                cb_phi = cav_pred['CB']['phi']
                ax_mean.loglog(gmpe_distances_km, cb_mean_g_s, color='black',
                               linewidth=2.5, label=f'CB14 CAV (M{magnitude})')
                ax_mean.loglog(gmpe_distances_km, cb_mean_g_s * np.exp(cb_std_ln),
                               '--', color='black', linewidth=1.5, alpha=0.7,
                               label='CB14 ±1σ')
                ax_mean.loglog(gmpe_distances_km, cb_mean_g_s * np.exp(-cb_std_ln),
                               '--', color='black', linewidth=1.5, alpha=0.7)
                ax_cov.plot(gmpe_distances_km, cb_phi, color='black',
                            linewidth=2.5, label='CB14 φ')
            except Exception as e:
                print(f"Warning: Could not add CAV GMM curve: {e}")
        elif add_gmpe and metric in ['PGV', 'PGD']:
            print(f"Note: {metric} doesn't have direct NGA-West2 GMPE support")
            

        unit = _metric_unit(metric)
        ax_mean.set_xlabel('Distance (km)', fontsize=14, fontweight='bold')
        ax_mean.set_ylabel(f'{metric} ({unit})' if unit else metric,
                           fontsize=14, fontweight='bold')
        ax_mean.set_yscale('log')
        ax_mean.set_xscale('log')
        ax_mean.grid(True, which='both', alpha=0.25)
        ax_mean.legend(fontsize=9, loc='lower left', framealpha=0.9, ncol=1)
        ax_mean.tick_params(axis='both', which='major', labelsize=12)

        ax_cov.set_xlabel('Distance (km)', fontsize=14, fontweight='bold')
        std_label = f'{metric} Intra-event Std Dev φ (ln units)'
        ax_cov.set_ylabel(std_label, fontsize=14, fontweight='bold')
        ax_cov.set_ylim(0, 1)
        ax_cov.grid(True, which='both', alpha=0.25)
        ax_cov.legend(fontsize=9, loc='upper left', framealpha=0.9, ncol=1)
        ax_cov.tick_params(axis='both', which='major', labelsize=12)

        if distance_range:
            ax_mean.set_xlim((distance_range[0]/1000, distance_range[1]/1000))
            ax_cov.set_xlim((distance_range[0]/1000, distance_range[1]/1000))
        else:
            ax_mean.set_xlim((1.0, 18.0))
            ax_cov.set_xlim((0, 18.0))

        if metric in global_ranges and metric != 'CAV':
            ax_mean.set_ylim(global_ranges[metric])

        fig_mean.tight_layout()
        fig_mean.savefig(f"{output_dir}/{metric}_vs_distance.png", dpi=300, bbox_inches='tight')
        plt.close(fig_mean)

        fig_cov.tight_layout()
        fig_cov.savefig(f"{output_dir}/{metric}_std_vs_distance.png", dpi=300, bbox_inches='tight')
        plt.close(fig_cov)

def plot_response_spectra_vs_distance(scenarios, periods=None, output_dir="results", distance_range=None, add_gmpe=True, magnitude=7.0, vs30=760.0):
    """Plot response spectra vs distance - raw values as stored"""

    if periods is None:
        # Auto-detect all available RSA periods from first scenario
        if scenarios:
            try:
                data = np.load(scenarios[0].gm_file)
                rsa_keys = [k for k in data.keys() if k.startswith('RSA_T_') and k.endswith('_mean')]
                periods = []
                for key in sorted(rsa_keys):
                    period_str = key.replace('RSA_T_', '').replace('_mean', '')
                    periods.append(period_str)
                print(f"Auto-detected {len(periods)} RSA periods: {periods}")
            except Exception as e:
                print(f"Could not auto-detect periods, using default: {e}")
                periods = ['0_100', '0_250', '0_500', '1_000', '2_000', '5_000']
        else:
            periods = ['0_100', '0_250', '0_500', '1_000', '2_000', '5_000']
    
    os.makedirs(output_dir, exist_ok=True)
    period_ranges = {}
    for period in periods:
        rng = _compute_global_yrange(
            scenarios, f'RSA_T_{period}_mean', f'RSA_T_{period}_count',
            distance_range, convert_to_g=True)
        if rng is not None:
            period_ranges[period] = rng
    
    # Create individual plots for each period
    for period in periods:
        fig_mean, ax_mean = plt.subplots(figsize=(8, 6))
        fig_cov, ax_cov = plt.subplots(figsize=(8, 6))
        period_label = period.replace('_', '.')
        gmpe_phi_plotted = False

        max_dist_km = 0.0
        per_code_means = {}
        per_code_stds = {}
        for scenario_info in scenarios:
            try:
                data = load_gm_statistics(scenario_info.gm_file)
            except Exception as e:
                print(f"Warning: Could not load {scenario_info.source} for period {period}: {e}")
                continue
            mean_key = f'RSA_T_{period}_mean'
            if mean_key not in data.files:
                print(f"Info: Scenario {scenario_info.label} does not have RSA_T_{period_label}s, skipping")
                continue
            curve = _extract_distance_curve(
                data, mean_key, mean_key.replace('_mean', '_count'),
                mean_key.replace('_mean', '_std'), convert_to_g=True)
            if curve is None:
                print(f"Warning: Scenario {scenario_info.label} has no RSA_T={period_label}s data; skipping")
                continue
            distances_km, means_sorted, stds_sorted = curve
            max_dist_km = max(max_dist_km, distances_km.max())

            code = _code_of(scenario_info.label)
            color = _code_color(scenario_info.label)
            ax_mean.plot(distances_km, means_sorted, color=color,
                         linestyle='--', alpha=0.35, linewidth=1.0,
                         label='_nolegend_')
            ax_cov.plot(distances_km, stds_sorted, color=color,
                        linestyle='--', alpha=0.35, linewidth=1.0,
                        label='_nolegend_')
            per_code_means.setdefault(code, []).append((distances_km, means_sorted))
            per_code_stds.setdefault(code, []).append((distances_km, stds_sorted))

        if max_dist_km > 0:
            common_rjb = np.geomspace(1.0, max_dist_km, 60)
            for code in sorted(per_code_means):
                color = _CODE_COLORS.get(code, 'tab:gray')
                avg_x, avg_y = _group_geomean(per_code_means[code], common_rjb)
                if avg_x is not None and avg_y is not None and len(avg_x) >= 2:
                    ax_mean.plot(avg_x, avg_y, color=color, linewidth=3.0,
                                 label=f'{code} ({len(per_code_means[code])})')
                std_x, std_y = _group_arithmean_xlog(per_code_stds[code], common_rjb)
                if std_x is not None and std_y is not None and len(std_x) >= 2:
                    ax_cov.plot(std_x, std_y, color=color, linewidth=3.0,
                                label=f'{code} ({len(per_code_stds[code])})')
        
        # Add GMPE curves for RSA periods
        if add_gmpe and PLOT_GMPE_AVAILABLE:
            try:
                # Cap GMPE distance at max observed Rjb across plotted scenarios
                if distance_range:
                    dist_min, dist_max = distance_range[0] / 1000.0, distance_range[1] / 1000.0
                else:
                    dist_min = 1.0
                    dist_max = max_dist_km if max_dist_km > dist_min else 100.0
                gmpe_distances = np.logspace(np.log10(dist_min), np.log10(dist_max), 50)
                
                # Convert period format (e.g., "1_000" -> 1.0)
                period_value = float(period.replace('_', '.'))
                periods = [period_value]
                
                # Get enhanced GMPE predictions
                gmpe_results = get_nga_west2_gmpe_predictions(gmpe_distances, magnitude, periods, vs30)
                
                if period_value in gmpe_results:
                    data = gmpe_results[period_value]
                    gmpe_distances_km = data['distances']
                    ax_mean.loglog(gmpe_distances_km, data['NGA_AVG']['mean'], 'black', linewidth=3, label=f'NGA-West2 Avg (M{magnitude})')
                    
                    # Black shaded area: range of 4 GMPE means
                    all_means = []
                    for gmpe in ['ASK', 'BSSA', 'CB', 'CY']:
                        all_means.append(data[gmpe]['mean'])
                    
                    means_upper = np.max(all_means, axis=0)
                    means_lower = np.min(all_means, axis=0)
                    
                    ax_mean.fill_between(gmpe_distances_km, means_lower, means_upper, 
                                       color='black', alpha=0.2, label='GMPE Mean Range')
                    
                    # Black dashed lines: ±1σ envelope from all 4 GMPEs
                    all_upper = []
                    all_lower = []
                    for gmpe in ['ASK', 'BSSA', 'CB', 'CY']:
                        upper = data[gmpe]['mean'] * np.exp(data[gmpe]['std'])
                        lower = data[gmpe]['mean'] * np.exp(-data[gmpe]['std'])
                        all_upper.append(upper)
                        all_lower.append(lower)
                    
                    envelope_upper = np.max(all_upper, axis=0)
                    envelope_lower = np.min(all_lower, axis=0)
                    
                    ax_mean.loglog(gmpe_distances_km, envelope_upper, '--', color='black', linewidth=2, alpha=0.7)
                    ax_mean.loglog(gmpe_distances_km, envelope_lower, '--', color='black', linewidth=2, alpha=0.7)

                    # φ (intra-event std) band on ax_cov
                    all_phi = [data[g]['phi'] for g in ('ASK', 'BSSA', 'CB', 'CY') if 'phi' in data.get(g, {})]
                    if all_phi:
                        phi_lo = np.min(all_phi, axis=0)
                        phi_hi = np.max(all_phi, axis=0)
                        ax_cov.fill_between(gmpe_distances_km, phi_lo, phi_hi,
                                            color='grey', alpha=0.4,
                                            label='GMM φ range', zorder=0)

                    print(f"Added GMPE curve for RSA T={period_label}s (M{magnitude}, Vs30={vs30})")

            except Exception as e:
                print(f"Warning: Could not add GMPE curves for RSA T={period_label}s: {e}")

        ax_mean.set_xlabel('Distance (km)', fontsize=14, fontweight='bold')
        ax_mean.set_ylabel(f'RSA T={period_label}s (g)', fontsize=14, fontweight='bold')
        ax_mean.set_yscale('log')
        ax_mean.set_xscale('log')
        ax_mean.grid(True, which='both', alpha=0.25)
        ax_mean.legend(fontsize=9, loc='lower left', framealpha=0.9, ncol=1)
        ax_mean.tick_params(axis='both', which='major', labelsize=12)

        ax_cov.set_xlabel('Distance (km)', fontsize=14, fontweight='bold')
        ax_cov.set_ylabel(f'RSA T={period_label}s Intra-event Std Dev φ (ln units)', fontsize=14, fontweight='bold')
        ax_cov.set_ylim(0, 1)
        ax_cov.grid(True, which='both', alpha=0.25)
        ax_cov.legend(fontsize=9, loc='upper left', framealpha=0.9, ncol=1)
        ax_cov.tick_params(axis='both', which='major', labelsize=12)

        if distance_range:
            ax_mean.set_xlim((distance_range[0]/1000, distance_range[1]/1000))
            ax_cov.set_xlim((distance_range[0]/1000, distance_range[1]/1000))
        else:
            ax_mean.set_xlim((1.0, 18.0))
            ax_cov.set_xlim((0, 18.0))

        if period in period_ranges:
            ax_mean.set_ylim(period_ranges[period])

        fig_mean.tight_layout()
        fig_mean.savefig(f"{output_dir}/RSA_T{period_label}s_vs_distance.png", dpi=300, bbox_inches='tight')
        plt.close(fig_mean)

        fig_cov.tight_layout()
        fig_cov.savefig(f"{output_dir}/RSA_T{period_label}s_std_vs_distance.png", dpi=300, bbox_inches='tight')
        plt.close(fig_cov)


def load_scenarios_from_json(json_path):
    """Load scenario paths from a JSON file."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Scenarios file not found: {json_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in scenarios file {json_path}: {exc}") from exc

    if isinstance(data, dict):
        key = 'scenarios'
        if key not in data:
            raise ValueError(f"JSON object must contain a '{key}' key with a list of scenario paths")
        data = data[key]

    if not isinstance(data, list) or not all(isinstance(item, str) for item in data):
        raise ValueError("Scenarios JSON must be a list of scenario path strings")

    return data

def plot_response_spectra_vs_periods(scenarios, target_rjb_km, output_dir="results", add_gmpe=True, magnitude=7.0, vs30=760.0):
    """
    Plot RSA metrics as a function of periods for a given Rjb distance
    
    Args:
        scenarios: List of ScenarioInfo objects containing scenario data
        target_rjb_km: Target Rjb distance in kilometers
        output_dir: Output directory for plots
        add_gmpe: Whether to add GMPE comparison curves
        magnitude: Magnitude for GMPE predictions
        vs30: Vs30 for GMPE predictions in m/s
    """
    os.makedirs(output_dir, exist_ok=True)
    scenario_rsa_data = {}
    all_periods = set()
    for scenario_info in scenarios:
        try:
            data = load_gm_statistics(scenario_info.gm_file)
        except Exception as e:
            print(f"Warning: Could not process scenario {scenario_info.label}: {e}")
            continue
        rjb_bins_km = data['rjb_distance_bins'] / 1000.0
        actual_rjb_km = float(rjb_bins_km[int(np.argmin(np.abs(rjb_bins_km - target_rjb_km)))])
        print(f"Scenario {scenario_info.label}: target {target_rjb_km}km → actual {actual_rjb_km:.1f}km")
        extracted = _extract_periods_at_rjb(data, target_rjb_km, want_std=True)
        if extracted is None:
            continue
        periods, sa_g, stds = extracted
        all_periods.update(periods.tolist())
        scenario_rsa_data[scenario_info.label] = {
            'periods': periods, 'sa_g': sa_g, 'stds': stds,
            'actual_rjb_km': actual_rjb_km,
        }

    if not scenario_rsa_data:
        print("Error: No valid RSA data found for any scenario")
        return

    periods_sorted = sorted(all_periods)
    
    # Create two plots: mean and std
    fig_mean, ax_mean = plt.subplots(figsize=(8, 6))
    fig_std, ax_std = plt.subplots(figsize=(8, 6))
    
    per_code_curves = {}   # code -> list of (periods, rsa_mean) per scenario
    per_code_stds = {}     # code -> list of (periods, rsa_std) per scenario
    for scenario_label, scenario_data in scenario_rsa_data.items():
        periods_plot = scenario_data['periods']
        rsa_means_plot = scenario_data['sa_g']
        rsa_stds_plot = scenario_data['stds']
        code = _code_of(scenario_label)
        color = _code_color(scenario_label)

        ax_mean.loglog(periods_plot, rsa_means_plot, '--', color=color,
                       alpha=0.35, linewidth=1.0, label='_nolegend_')
        ax_std.loglog(periods_plot, rsa_stds_plot, '--', color=color,
                      alpha=0.35, linewidth=1.0, label='_nolegend_')

        per_code_curves.setdefault(code, []).append((periods_plot, rsa_means_plot))
        per_code_stds.setdefault(code, []).append((periods_plot, rsa_stds_plot))

    if periods_sorted:
        common_T = np.geomspace(min(periods_sorted), max(periods_sorted), 80)
        for code in sorted(per_code_curves):
            color = _CODE_COLORS.get(code, 'tab:gray')
            avg_x, avg_y = _group_geomean(per_code_curves[code], common_T)
            if avg_x is not None and avg_y is not None and len(avg_x) >= 2:
                ax_mean.loglog(avg_x, avg_y, color=color, linewidth=3.0,
                               label=f'{code} ({len(per_code_curves[code])})')
            std_x, std_y = _group_arithmean_xlog(per_code_stds[code], common_T)
            if std_x is not None and std_y is not None and len(std_x) >= 2:
                ax_std.plot(std_x, std_y, color=color, linewidth=3.0,
                            label=f'{code} ({len(per_code_stds[code])})')
            
    
    # Add GMPE comparison if requested and available
    if add_gmpe and PLOT_GMPE_AVAILABLE:
        try:
            # Use the actual Rjb distance from the first scenario (they should be similar)
            first_scenario = next(iter(scenario_rsa_data.values()))
            gmpe_rjb_km = first_scenario['actual_rjb_km']
            
            # Get GMPE predictions for this distance
            gmpe_distances = np.array([gmpe_rjb_km])
            gmpe_periods_list = sorted(all_periods)
            gmpe_results = get_nga_west2_gmpe_predictions(gmpe_distances, magnitude, gmpe_periods_list, vs30)
            
            # Extract GMPE data for plotting
            gmpe_periods_plot = []
            gmpe_sa_avg = []
            all_gmpe_means = []
            all_gmpe_upper = []
            all_gmpe_lower = []
            
            for period in gmpe_periods_list:
                if period in gmpe_results:
                    data = gmpe_results[period]
                    gmpe_periods_plot.append(period)
                    
                    # Get NGA-West2 average
                    gmpe_sa_avg.append(data['NGA_AVG']['mean'][0])
                    
                    # Collect all GMPE means for shaded area
                    period_means = []
                    period_upper = []
                    period_lower = []
                    
                    for gmpe in ['ASK', 'BSSA', 'CB', 'CY']:
                        mean_val = data[gmpe]['mean'][0]
                        std_val = data[gmpe]['std'][0]
                        period_means.append(mean_val)
                        period_upper.append(mean_val * np.exp(std_val))
                        period_lower.append(mean_val * np.exp(-std_val))
                    
                    all_gmpe_means.append(period_means)
                    all_gmpe_upper.append(period_upper)
                    all_gmpe_lower.append(period_lower)
            
            gmpe_periods_plot = np.array(gmpe_periods_plot)
            gmpe_sa_avg = np.array(gmpe_sa_avg)
            all_gmpe_means = np.array(all_gmpe_means)
            all_gmpe_upper = np.array(all_gmpe_upper)
            all_gmpe_lower = np.array(all_gmpe_lower)
            
            # Plot NGA-West2 average (solid black line) on mean plot
            ax_mean.loglog(gmpe_periods_plot, gmpe_sa_avg, 'black', linewidth=3, 
                          label=f'NGA-West2 Avg (M{magnitude})')
            
            # Black shaded area: range of 4 GMPE means on mean plot
            means_upper = np.max(all_gmpe_means, axis=1)
            means_lower = np.min(all_gmpe_means, axis=1)
            ax_mean.fill_between(gmpe_periods_plot, means_lower, means_upper, 
                                color='black', alpha=0.2, label='GMPE Mean Range')
            
            # Black dashed lines: ±1σ envelope from all 4 GMPEs on mean plot
            envelope_upper = np.max(all_gmpe_upper, axis=1)
            envelope_lower = np.min(all_gmpe_lower, axis=1)
            ax_mean.loglog(gmpe_periods_plot, envelope_upper, '--', color='black', linewidth=2, alpha=0.7)
            ax_mean.loglog(gmpe_periods_plot, envelope_lower, '--', color='black', linewidth=2, alpha=0.7)
            
            # For std plot, get GMPE intra-event phi (not total sigma)
            gmpe_phi_plot = []
            all_gmpe_phis = []

            for period in gmpe_periods_list:
                if period in gmpe_results:
                    data = gmpe_results[period]

                    gmpe_phi_plot.append(data['NGA_AVG']['phi'][0])

                    period_phis = []
                    for gmpe in ['ASK', 'BSSA', 'CB', 'CY']:
                        period_phis.append(data[gmpe]['phi'][0])
                    all_gmpe_phis.append(period_phis)

            gmpe_phi_plot = np.array(gmpe_phi_plot)
            all_gmpe_phis = np.array(all_gmpe_phis)

            # Plot NGA-West2 average phi (solid black line) on std plot
            ax_std.loglog(gmpe_periods_plot, gmpe_phi_plot, 'black', linewidth=3,
                         label=f'NGA-West2 Avg φ (M{magnitude})')

            # Black shaded area: range of 4 GMPE phis on std plot
            phi_upper = np.max(all_gmpe_phis, axis=1)
            phi_lower = np.min(all_gmpe_phis, axis=1)
            ax_std.fill_between(gmpe_periods_plot, phi_lower, phi_upper,
                               color='black', alpha=0.2, label='GMM φ range')
            
        except Exception as e:
            print(f"Warning: Could not add GMPE curves: {e}")
    
    # Formatting for mean plot
    ax_mean.set_xlabel('Period (s)', fontsize=12, fontweight='bold')
    ax_mean.set_ylabel('Response Spectral Acceleration (g)', fontsize=12, fontweight='bold')
    ax_mean.grid(True, which='both', alpha=0.25)
    ax_mean.legend(fontsize=9, loc='lower left', framealpha=0.9, ncol=1)

    # Formatting for std plot
    ax_std.set_xlabel('Period (s)', fontsize=12, fontweight='bold')
    ax_std.set_ylabel('RSA Intra-event Std Dev φ (ln units)', fontsize=12, fontweight='bold')
    ax_std.grid(True, which='both', alpha=0.25)
    ax_std.legend(fontsize=9, loc='upper left', framealpha=0.9, ncol=1)
    
    # Set reasonable axis limits
    if periods_sorted:
        ax_mean.set_xlim(min(periods_sorted) * 0.8, max(periods_sorted) * 1.2)
        ax_std.set_xlim(min(periods_sorted) * 0.8, max(periods_sorted) * 1.2)
    
    # Save plots
    output_file_mean = os.path.join(output_dir, f'response_spectra_vs_periods_Rjb_{target_rjb_km}km.png')
    fig_mean.tight_layout()
    fig_mean.savefig(output_file_mean, dpi=300, bbox_inches='tight')
    plt.close(fig_mean)
    
    output_file_std = os.path.join(output_dir, f'response_spectra_std_vs_periods_Rjb_{target_rjb_km}km.png')
    fig_std.tight_layout()
    fig_std.savefig(output_file_std, dpi=300, bbox_inches='tight')
    plt.close(fig_std)
    
    print(f"Response spectra vs periods plots saved: {output_file_mean}")
    print(f"Response spectra std vs periods plot saved: {output_file_std}")
    
    # Create a summary of the data used
    summary_file = os.path.join(output_dir, f'response_spectra_vs_periods_Rjb_{target_rjb_km}km_summary.txt')
    with open(summary_file, 'w') as f:
        f.write(f"Response Spectra vs Periods Analysis\n")
        f.write(f"Target Rjb Distance: {target_rjb_km} km\n")
        f.write(f"Number of scenarios: {len(scenario_rsa_data)}\n")
        f.write(f"Periods analyzed: {len(periods_sorted)}\n\n")
        
        for scenario_label, scenario_data in scenario_rsa_data.items():
            periods = scenario_data['periods']
            f.write(f"\nScenario: {scenario_label}\n")
            f.write(f"  Actual Rjb: {scenario_data['actual_rjb_km']:.2f} km\n")
            f.write(f"  Available periods: {len(periods)}\n")
            f.write(f"  Period range: {periods.min():.3f} - {periods.max():.3f} s\n")
    
    print(f"Summary saved: {summary_file}")


def plot_response_spectra_bias_vs_periods(scenarios, target_rjb_km, output_dir, magnitude=7.0, vs30=760.0):
    """Manuscript Fig 14 right panel: ln(simulated SA / NGA-West2 avg) vs period
    at a fixed Rjb. Per-simulation dashed, group-mean bold."""
    if not PLOT_GMPE_AVAILABLE:
        print("WARNING: NGA-West2 GMPE unavailable; cannot compute bias.")
        return
    os.makedirs(output_dir, exist_ok=True)

    per_scenario = {}    # label -> {periods, sa_g, actual_rjb_km, code}
    all_periods = set()
    for s in scenarios:
        try:
            data = load_gm_statistics(s.gm_file)
        except Exception as exc:
            print(f"Warning: skipping {s.label}: {exc}")
            continue
        rjb_bins_km = data['rjb_distance_bins'].astype(float) / 1000.0
        actual_rjb_km = float(rjb_bins_km[int(np.argmin(np.abs(rjb_bins_km - target_rjb_km)))])
        extracted = _extract_periods_at_rjb(data, target_rjb_km)
        if extracted is None:
            continue
        periods, sa_g = extracted
        all_periods.update(periods.tolist())
        per_scenario[s.label] = {
            'periods': periods, 'sa_g': sa_g,
            'actual_rjb_km': actual_rjb_km, 'code': _code_of(s.label),
        }

    if not per_scenario:
        print("Bias plot: no scenarios with valid RSA at target Rjb")
        return
    periods_sorted = np.array(sorted(all_periods))

    # Pre-compute GMM predictions at each unique actual bin distance so each
    # scenario's bias is compared against the GMM at the same distance.
    unique_dists = np.unique([sd['actual_rjb_km'] for sd in per_scenario.values()])
    try:
        _gmpe_all = get_nga_west2_gmpe_predictions(
            unique_dists, magnitude, list(periods_sorted), vs30)
        gmpe_nga_avg = {
            float(d): {T: float(_gmpe_all[T]['NGA_AVG']['mean'][i])
                       for T in periods_sorted}
            for i, d in enumerate(unique_dists)
        }
    except Exception as exc:
        print(f"WARNING: GMPE call failed for bias: {exc}")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axhline(0.0, color='black', linewidth=1.5, alpha=0.7, zorder=1, label='0 Bias')

    per_code_bias = {}
    for label, sd in per_scenario.items():
        gmm_avg = np.array([gmpe_nga_avg[sd['actual_rjb_km']][T] for T in periods_sorted])
        gmm_at = np.interp(np.log(sd['periods']),
                           np.log(periods_sorted), np.log(gmm_avg))
        bias = np.log(sd['sa_g']) - gmm_at
        ax.plot(sd['periods'], bias, '--', color='gray',
                alpha=0.35, linewidth=1.0, label='_nolegend_')
        per_code_bias.setdefault(sd['code'], []).append((sd['periods'], bias))

    common_T = np.geomspace(periods_sorted.min(), periods_sorted.max(), 80)
    for code in sorted(per_code_bias):
        x_curve, y_curve = _group_arithmean_xlog(per_code_bias[code], common_T)
        if x_curve is None or len(x_curve) < 2:
            continue
        author_label = _CODE_AUTHOR_LABELS.get(code, code)
        ax.plot(x_curve, y_curve, color=_CODE_COLORS.get(code, 'tab:gray'),
                linewidth=3.0, label=author_label)

    ax.set_xscale('log')
    ax.set_xlabel('Period (s)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Bias (ln)', fontsize=13, fontweight='bold')
    ax.set_title('Spectral Acceleration Bias vs Period', fontsize=13)
    ax.grid(True, which='both', alpha=0.25)
    ax.legend(fontsize=9, loc='upper right', framealpha=0.9)
    ax.tick_params(labelsize=11)

    out_path = os.path.join(output_dir, f'response_spectra_bias_vs_periods_Rjb_{target_rjb_km}km.png')
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Bias plot saved: {out_path}")


def plot_inter_event_std_vs_distance(scenarios, output_dir, periods_target=(3.0, 1.0, 0.333)):
    """Manuscript Fig 17: inter-event tau vs Rjb at three periods.
    For each period: per-code dashed tau (std across that code's per-sim
    binned medians) plus solid 'epistemic' tau (std across the per-code
    group-means). Output: one PNG per period."""
    os.makedirs(output_dir, exist_ok=True)

    for T in periods_target:
        per_code_curves = {}
        for s in scenarios:
            try:
                data = load_gm_statistics(s.gm_file)
            except Exception:
                continue
            mean_key, _ = _match_rsa_period_key(data, T, tol=0.06)
            if mean_key is None:
                continue
            curve = _extract_distance_curve(
                data, mean_key, mean_key.replace('_mean', '_count'),
                convert_to_g=True)
            if curve is None:
                continue
            rjb_km, means, _ = curve
            ok = (rjb_km > 0) & (means > 0) & np.isfinite(means)
            if ok.sum() < 2:
                continue
            per_code_curves.setdefault(_code_of(s.label), []).append(
                (rjb_km[ok], means[ok]))

        if not per_code_curves:
            print(f"Tau-vs-distance: no valid data at T={T}s")
            continue

        rjb_max = max(c[0].max() for code in per_code_curves
                      for c in per_code_curves[code])
        common_rjb = np.geomspace(0.1, rjb_max, 120)

        fig, ax = plt.subplots(figsize=(8, 6))
        # Per-code within-group tau (std across that code's sims)
        per_code_groupmean = []
        for code in sorted(per_code_curves):
            color = _CODE_COLORS.get(code, 'tab:gray')
            x_tau, y_tau = _group_logstd(per_code_curves[code], common_rjb)
            if x_tau is not None and len(x_tau) >= 2:
                ax.plot(x_tau, y_tau, '--', color=color, linewidth=1.6,
                        label=f'{code} τ_within ({len(per_code_curves[code])})')
            x_avg, y_avg = _group_geomean(per_code_curves[code], common_rjb)
            if x_avg is not None and len(x_avg) >= 2:
                # Re-interpolate the group mean back onto common_rjb for the
                # epistemic-tau computation.
                per_code_groupmean.append((x_avg, y_avg))

        # Epistemic tau: std across group-means
        if len(per_code_groupmean) >= 2:
            x_ep, y_ep = _group_logstd(per_code_groupmean, common_rjb, min_n=2)
            if x_ep is not None and len(x_ep) >= 2:
                ax.plot(x_ep, y_ep, '-', color='black', linewidth=3.0,
                        label=f'epistemic τ across {len(per_code_groupmean)} groups')

        ax.set_xlabel('Distance (km)', fontsize=13, fontweight='bold')
        ax.set_ylabel(f'Inter-event τ (ln units), T={T:g}s',
                      fontsize=13, fontweight='bold')
        ax.set_xlim(0, 18.0)
        ax.set_ylim(0, max(0.8, ax.get_ylim()[1]))
        ax.grid(True, which='both', alpha=0.25)
        ax.legend(fontsize=9, loc='upper left', framealpha=0.9)
        ax.tick_params(labelsize=11)
        ax.set_title(f'Inter-event τ vs distance (T={T:g}s)', fontsize=13)

        out_path = os.path.join(output_dir, f'tau_T{T:.3f}s_vs_distance.png')
        fig.tight_layout()
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Tau vs distance saved: {out_path}")


def plot_inter_event_std_vs_periods(scenarios, target_rjb_km, output_dir):
    """Manuscript Fig 18: inter-event tau vs period at fixed Rjb. Per-code
    dashed tau (std across that code's sims) + solid epistemic tau (std
    across the per-code group-means)."""
    os.makedirs(output_dir, exist_ok=True)

    per_code_curves = {}
    all_periods = set()
    for s in scenarios:
        try:
            data = load_gm_statistics(s.gm_file)
        except Exception:
            continue
        extracted = _extract_periods_at_rjb(data, target_rjb_km)
        if extracted is None:
            continue
        periods, sa_g = extracted
        all_periods.update(periods.tolist())
        per_code_curves.setdefault(_code_of(s.label), []).append((periods, sa_g))

    if not per_code_curves:
        print("Tau-vs-period: no valid data")
        return
    common_T = np.geomspace(min(all_periods), max(all_periods), 80)

    fig, ax = plt.subplots(figsize=(8, 6))
    per_code_groupmean = []
    for code in sorted(per_code_curves):
        color = _CODE_COLORS.get(code, 'tab:gray')
        x_tau, y_tau = _group_logstd(per_code_curves[code], common_T)
        if x_tau is not None and len(x_tau) >= 2:
            ax.plot(x_tau, y_tau, '--', color=color, linewidth=1.6,
                    label=f'{code} τ_within ({len(per_code_curves[code])})')
        x_avg, y_avg = _group_geomean(per_code_curves[code], common_T)
        if x_avg is not None and len(x_avg) >= 2:
            per_code_groupmean.append((x_avg, y_avg))

    if len(per_code_groupmean) >= 2:
        x_ep, y_ep = _group_logstd(per_code_groupmean, common_T, min_n=2)
        if x_ep is not None and len(x_ep) >= 2:
            ax.plot(x_ep, y_ep, '-', color='black', linewidth=3.0,
                    label=f'epistemic τ across {len(per_code_groupmean)} groups')

    ax.set_xscale('log')
    ax.set_xlabel('Period (s)', fontsize=13, fontweight='bold')
    ax.set_ylabel(f'Inter-event τ (ln units) at Rjb={target_rjb_km} km',
                  fontsize=13, fontweight='bold')
    ax.set_ylim(0, max(0.8, ax.get_ylim()[1]))
    ax.grid(True, which='both', alpha=0.25)
    ax.legend(fontsize=9, loc='upper left', framealpha=0.9)
    ax.tick_params(labelsize=11)
    ax.set_title(f'Inter-event τ vs period (Rjb={target_rjb_km} km)', fontsize=13)

    out_path = os.path.join(output_dir, f'tau_vs_periods_Rjb_{target_rjb_km}km.png')
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Tau vs period saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description='Plot ground motion statistics - raw values as stored')
    parser.add_argument('scenarios', nargs='*', help='Scenario names or paths')
    parser.add_argument('--scenarios-file', dest='scenario_files', action='append',
                        help='Path to JSON file listing scenario names or paths; repeatable')
    parser.add_argument('--input-dir', default='results', help='Root directory containing scenario results')
    parser.add_argument('--output-dir', default='results', help='Output directory for plots')
    parser.add_argument('--distance-range', nargs=2, type=float, metavar=('MIN', 'MAX'), 
                       help='Distance range to plot in meters (sets x-axis limits only)')
    parser.add_argument('--add-gmpe', action='store_true', default=False,
                        help='Add GMPE comparison curves and GMM φ band on intra-event std plots')
    parser.add_argument('--plot-spectra-vs-periods', type=float, metavar='RJB_KM', default=10.0,
                        help='Generate response spectra vs periods plot at this Rjb (km). Default: 10.')
    parser.add_argument('--no-bias', action='store_true',
                        help='Skip the SA-vs-period bias plot (Fig 14 right panel)')
    parser.add_argument('--no-tau', action='store_true',
                        help='Skip inter-event tau plots (Figs 17/18)')
    parser.add_argument('--tau-periods', nargs='+', type=float,
                        default=[3.0, 1.0, 0.333],
                        help='Periods (s) for tau-vs-distance panels (default: 3 1 0.333)')
    parser.add_argument('--no-pergroup-fig12', action='store_true',
                        help='Skip per-group Fig 12 ensemble panels')
    parser.add_argument('--fig12-period', type=float, default=1.0,
                        help='Spectral period (s) for per-group Fig 12 panels (default: 1.0)')

    args = parser.parse_args()

    combined_scenarios = list(args.scenarios)

    scenario_files = args.scenario_files or []
    for scenario_file in scenario_files:
        try:
            file_scenarios = load_scenarios_from_json(scenario_file)
        except (FileNotFoundError, ValueError) as exc:
            print(f"Error loading scenarios from {scenario_file}: {exc}")
            return
        combined_scenarios.extend(file_scenarios)

    # Deduplicate raw inputs while preserving order
    seen_inputs = set()
    ordered_inputs = []
    for scenario in combined_scenarios:
        if scenario not in seen_inputs:
            seen_inputs.add(scenario)
            ordered_inputs.append(scenario)

    if not ordered_inputs:
        print("Error: No scenarios provided. Supply positional scenario names or --scenarios-file.")
        return
    
    # Parse distance range
    distance_range = None
    if args.distance_range:
        distance_range = (args.distance_range[0], args.distance_range[1])
        print(f"Filtering distance range: {distance_range[0]} - {distance_range[1]} meters")
    
    # Validate scenarios
    valid_scenarios = []
    seen_paths = set()
    for scenario in ordered_inputs:
        try:
            scenario_info = resolve_scenario_entry(scenario, args.input_dir)
        except ValueError as exc:
            print(f"Warning: {exc}")
            continue

        canonical = str(scenario_info.base_path)
        if canonical in seen_paths:
            continue

        if scenario_info.gm_file.exists():
            valid_scenarios.append(scenario_info)
            seen_paths.add(canonical)
        else:
            print(f"Warning: Scenario {scenario_info.base_path} not found or missing gm_statistics.npz")
    
    if not valid_scenarios:
        print("Error: No valid scenarios found")
        return
    
    print(f"Plotting {len(valid_scenarios)} scenarios: {', '.join(info.label for info in valid_scenarios)}")
    print(f"Output directory: {args.output_dir}")
    print("Plotting raw values as stored in npz files")
    
    # Generate plots

    plot_gm_metrics_vs_distance(valid_scenarios, args.output_dir, distance_range, 
                                add_gmpe=args.add_gmpe, 
                                magnitude=7.0, vs30=760.0)
    
    # Always generate response spectra plots  
    plot_response_spectra_vs_distance(valid_scenarios, output_dir=args.output_dir,
                                     distance_range=distance_range,
                                     add_gmpe=args.add_gmpe, magnitude=7.0, 
                                     vs30=760.0)
    
    # Generate response spectra vs periods plot if requested
    if args.plot_spectra_vs_periods:
        print(f"Generating response spectra vs periods plot for Rjb = {args.plot_spectra_vs_periods} km")
        plot_response_spectra_vs_periods(valid_scenarios,
                                        target_rjb_km=args.plot_spectra_vs_periods,
                                        output_dir=args.output_dir,
                                        add_gmpe=args.add_gmpe,
                                        magnitude=7.0,
                                        vs30=760.0)

        if args.add_gmpe and not args.no_bias:
            print(f"Generating Fig 14 bias plot at Rjb = {args.plot_spectra_vs_periods} km")
            plot_response_spectra_bias_vs_periods(valid_scenarios,
                                                  target_rjb_km=args.plot_spectra_vs_periods,
                                                  output_dir=args.output_dir,
                                                  magnitude=7.0, vs30=760.0)

    if not args.no_tau:
        print(f"Generating Fig 17 inter-event tau vs distance for periods: {args.tau_periods}")
        plot_inter_event_std_vs_distance(valid_scenarios,
                                         output_dir=args.output_dir,
                                         periods_target=tuple(args.tau_periods))
        if args.plot_spectra_vs_periods:
            print(f"Generating Fig 18 inter-event tau vs period at Rjb = {args.plot_spectra_vs_periods} km")
            plot_inter_event_std_vs_periods(valid_scenarios,
                                            target_rjb_km=args.plot_spectra_vs_periods,
                                            output_dir=args.output_dir)

    if not args.no_pergroup_fig12:
        try:
            from plot_pergroup_ens_figure12 import plot_one_group, _load_scenario, _code_of as _pg_code_of
            print(f"Generating manuscript Fig 12 panels per code at T={args.fig12_period}s")
            rng = np.random.default_rng(0)
            groups: dict = {}
            for s in valid_scenarios:
                groups.setdefault(_pg_code_of(s.label), []).append(s.label)
            for code, labels in sorted(groups.items()):
                per_code = []
                for label in labels:
                    try:
                        per_code.append(_load_scenario(Path(args.input_dir), label,
                                                       args.fig12_period, rng))
                    except Exception as exc:
                        print(f"Warning: Fig 12 skip {label}: {exc}")
                if not per_code:
                    continue
                try:
                    out = plot_one_group(code, per_code, args.fig12_period, 7.0, 760.0,
                                         Path(args.output_dir), add_gmpe=args.add_gmpe)
                    print(f"  wrote {out}")
                except Exception as exc:
                    print(f"Warning: Fig 12 panel {code} failed: {exc}")
        except ImportError as exc:
            print(f"Warning: plot_pergroup_ens_figure12 unavailable: {exc}")

if __name__ == "__main__":
    main()
