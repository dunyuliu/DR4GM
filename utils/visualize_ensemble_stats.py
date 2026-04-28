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
  --add-gmpe        : Add GMPE comparison curves and reference line at y=0.6 for COV plots
  --magnitude M     : Set magnitude for GMPE predictions (default: 7.0)
  --vs30 VS30       : Set Vs30 for GMPE predictions in m/s (default: 760)
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

# Try to import OpenQuake Engine GMPE functions - requires OpenQuake Engine
# Install with: pip install openquake.engine
try:
    from openquake_engine_gmpe import get_nga_west2_gmpe_predictions
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


def _code_color(label):
    code = label.split('/', 1)[0] if '/' in label else label
    return _CODE_COLORS.get(code, 'tab:gray')


def _valid_bin_mask(counts, min_frac=0.20):
    """Bins with count >= max(1, min_frac * counts.max()) — drops sparse far-edge tail."""
    counts = np.asarray(counts)
    if counts.size == 0:
        return counts.astype(bool)
    threshold = max(1, int(np.ceil(min_frac * counts.max())))
    return counts >= threshold


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
    
    # First pass: collect all data to determine global y-ranges for each metric
    global_ranges = {}
    
    metrics = ['PGA', 'PGV', 'PGD', 'CAV']
    # Collect data within distance range for y-range calculation
    for metric in metrics:
        all_means = []
        for scenario_info in scenarios:
            try:
                data = np.load(scenario_info.gm_file)
                if f'{metric}_mean' not in data:
                    continue
                distances = data['rjb_distance_bins']
                means = data[f'{metric}_mean']
                counts = data[f'{metric}_count']
                
                # Convert acceleration metrics from cm/s² to g
                if metric in ['PGA'] or metric.startswith('RSA_T_'):
                    means = means / 981.0
                
                # Only use valid data
                valid_mask = _valid_bin_mask(counts)
                distances = distances[valid_mask]
                means = means[valid_mask]
                
                # Filter by distance range if specified
                if distance_range:
                    dist_mask = (distances >= distance_range[0]) & (distances <= distance_range[1])
                    means = means[dist_mask]
                
                if len(means) > 0:
                    all_means.extend(means)
            except Exception:
                continue
        
        if all_means:
            y_min = np.min(all_means)
            y_max = np.max(all_means)
            # For log plots, use multiplicative padding (10% in log space)
            log_y_min = np.log10(y_min)
            log_y_max = np.log10(y_max)
            log_range = log_y_max - log_y_min
            log_padding = log_range * 0.1
            global_ranges[metric] = (10**(log_y_min - log_padding), 10**(log_y_max + log_padding))
    
    # Create individual plots for each metric
    for metric in metrics:
        fig_mean, ax_mean = plt.subplots(figsize=(8, 6))
        fig_cov, ax_cov = plt.subplots(figsize=(8, 6))

        # Add reference line at y=0.6 if GMPE is enabled
        if add_gmpe:
            ax_cov.axhline(y=0.6, color='grey', linewidth=4, alpha=0.7,
                          label='Reference (0.6)', zorder=1)

        max_dist_km = 0.0
        for j, scenario_info in enumerate(scenarios):
            try:
                data = load_gm_statistics(scenario_info.gm_file)
                
                # Check if metric exists in this scenario
                if f'{metric}_mean' not in data:
                    print(f"Info: Scenario {scenario_info.label} does not have {metric}, skipping")
                    continue
                
                distances = data['rjb_distance_bins']  # Keep in meters
                means = data[f'{metric}_mean']  # Raw values, no conversion
                stds = data[f'{metric}_std']    # Raw values, no conversion
                
                # Convert acceleration metrics from cm/s² to g (1g = 981 cm/s²)
                if metric in ['PGA'] or metric.startswith('RSA_T_'):
                    means = means / 981.0
                    stds = stds 
                counts = data[f'{metric}_count']  # Get counts to filter empty bins
                
                # Filter out bins with no data (count == 0)
                # Only filter out invalid data
                valid_mask = _valid_bin_mask(counts)
                distances = distances[valid_mask]
                means = means[valid_mask]
                stds = stds[valid_mask]

                if distances.size == 0:
                    print(f"Warning: Scenario {scenario_info.label} has no {metric} data; skipping")
                    continue

                # Clean scenario name for legend
                scenario_name = scenario_info.label
                
                # Sort by distance to avoid connection artifacts
                sort_idx = np.argsort(distances)
                distances_sorted = distances[sort_idx]
                means_sorted = means[sort_idx]
                stds_sorted = stds[sort_idx]

                if distances_sorted.size:
                    max_dist_km = max(max_dist_km, distances_sorted.max() / 1000.0)

                # Plot median values as dashed lines (no markers)
                ax_mean.plot(distances_sorted/1000, means_sorted,
                             label=scenario_name, color=_code_color(scenario_name),
                             linestyle='--', alpha=0.8, linewidth=2)

                # Use intra-event variability directly (stats[:,1] from original DR4GM)
                cov = stds_sorted
                ax_cov.plot(distances_sorted/1000, cov,
                            label=scenario_name, color=_code_color(scenario_name),
                            linestyle='--', alpha=0.8, linewidth=2)

            except Exception as e:
                print(f"Warning: Could not load {scenario_info.source}: {e}")
                continue

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
                    
                    # Convert GMPE distances back to meters for plotting
                    gmpe_distances_m = data['distances'] * 1000.0
                    
                    
                    # Convert GMPE distances to km
                    gmpe_distances_km = gmpe_distances_m / 1000.0
                    
                    # Plot NGA-West2 average 
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

                    print(f"Added GMPE curve for {metric} (M{magnitude}, Vs30={vs30})")
                    
            except Exception as e:
                print(f"Warning: Could not add GMPE curves for {metric}: {e}")
        elif add_gmpe and metric in ['PGV', 'PGD', 'CAV']:
            print(f"Note: {metric} doesn't have direct NGA-West2 GMPE support")
            

        ax_mean.set_xlabel('Distance (km)', fontsize=14, fontweight='bold')
        # Set Y-axis label with appropriate units
        if metric in ['PGA'] or metric.startswith('RSA_T_'):
            ax_mean.set_ylabel(f'{metric} (g)', fontsize=14, fontweight='bold')
        elif metric == 'PGV':
            ax_mean.set_ylabel(f'{metric} (cm/s)', fontsize=14, fontweight='bold')
        elif metric == 'PGD':
            ax_mean.set_ylabel(f'{metric} (cm)', fontsize=14, fontweight='bold')
        elif metric == 'CAV':
            ax_mean.set_ylabel(f'{metric} (g)', fontsize=14, fontweight='bold')
        else:
            ax_mean.set_ylabel(f'{metric}', fontsize=14, fontweight='bold')
        ax_mean.set_yscale('log')
        ax_mean.set_xscale('log')
        ax_mean.grid(True, alpha=0.3)
        ax_mean.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
        ax_mean.tick_params(axis='both', which='major', labelsize=16)

        ax_cov.set_xlabel('Distance (km)', fontsize=14, fontweight='bold')
        ax_cov.set_ylabel(f'{metric} Intra-event Standard Deviation (g)' if metric in ['PGA'] or metric.startswith('RSA_T_') else f'{metric} Intra-event Standard Deviation', fontsize=14, fontweight='bold')
        ax_cov.set_xscale('log')
        ax_cov.set_ylim(0, 1)  # Set COV range from 0 to 1
        ax_cov.grid(True, alpha=0.3)
        ax_cov.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
        ax_cov.tick_params(axis='both', which='major', labelsize=16)

        if distance_range:
            ax_mean.set_xlim((distance_range[0]/1000, distance_range[1]/1000))
            ax_cov.set_xlim((distance_range[0]/1000, distance_range[1]/1000))
        else:
            xmax = max(max_dist_km, 1.0)
            ax_mean.set_xlim((1.0, xmax))
            ax_cov.set_xlim((1.0, xmax))
        
        # Set y-limits based on data range for this metric  
        if metric in global_ranges:
            ax_mean.set_ylim(global_ranges[metric])

        # Adjust layout: plot uses left 75% (6/8), legend on right 25% (2/8)
        fig_mean.subplots_adjust(left=0.15, right=0.75, top=0.9, bottom=0.15)
        fig_mean.savefig(f"{output_dir}/{metric}_vs_distance.png", dpi=300, bbox_inches='tight')
        plt.close(fig_mean)

        # Adjust layout: plot uses left 75% (6/8), legend on right 25% (2/8)
        fig_cov.subplots_adjust(left=0.15, right=0.75, top=0.9, bottom=0.15)
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
    # First pass: collect data within distance range to determine y-ranges for each period
    period_ranges = {}
    for period in periods:
        all_means = []
        for scenario_info in scenarios:
            try:
                data = np.load(scenario_info.gm_file)
                if f'RSA_T_{period}_mean' not in data:
                    continue
                distances = data['rjb_distance_bins']
                means = data[f'RSA_T_{period}_mean'] / 981.0  # Convert to g
                counts = data[f'RSA_T_{period}_count']
                
                # Only use valid data
                valid_mask = _valid_bin_mask(counts)
                distances = distances[valid_mask]
                means = means[valid_mask]
                
                # Filter by distance range if specified
                if distance_range:
                    dist_mask = (distances >= distance_range[0]) & (distances <= distance_range[1])
                    means = means[dist_mask]
                
                if len(means) > 0:
                    all_means.extend(means)
            except Exception:
                continue
        
        if all_means:
            y_min = np.min(all_means)
            y_max = np.max(all_means)
            # For log plots, use multiplicative padding (10% in log space)
            log_y_min = np.log10(y_min)
            log_y_max = np.log10(y_max)
            log_range = log_y_max - log_y_min
            log_padding = log_range * 0.1
            period_ranges[period] = (10**(log_y_min - log_padding), 10**(log_y_max + log_padding))
    
    # Create individual plots for each period
    for period in periods:
        fig_mean, ax_mean = plt.subplots(figsize=(8, 6))
        fig_cov, ax_cov = plt.subplots(figsize=(8, 6))
        period_label = period.replace('_', '.')

        # Add reference line at y=0.6 if GMPE is enabled
        if add_gmpe:
            ax_cov.axhline(y=0.6, color='grey', linewidth=4, alpha=0.7,
                          label='Reference (0.6)', zorder=1)

        max_dist_km = 0.0
        for j, scenario_info in enumerate(scenarios):
            try:
                data = load_gm_statistics(scenario_info.gm_file)
                
                # Check if this period exists in this scenario
                if f'RSA_T_{period}_mean' not in data:
                    print(f"Info: Scenario {scenario_info.label} does not have RSA_T_{period_label}s, skipping")
                    continue
                
                distances = data['rjb_distance_bins']  # Keep in meters
                means = data[f'RSA_T_{period}_mean']  # Raw values, no conversion
                stds = data[f'RSA_T_{period}_std']    # Raw values, no conversion
                
                # Convert acceleration from cm/s² to g (1g = 981 cm/s²)
                means = means / 981.0
                stds = stds 
                counts = data[f'RSA_T_{period}_count']  # Get counts to filter empty bins
                
                # Only filter out invalid data
                valid_mask = _valid_bin_mask(counts)
                distances = distances[valid_mask]
                means = means[valid_mask]
                stds = stds[valid_mask]

                if distances.size == 0:
                    print(f"Warning: Scenario {scenario_info.label} has no RSA_T={period_label}s data; skipping")
                    continue

                # Clean scenario name for legend
                scenario_name = scenario_info.label
                
                # Sort by distance to avoid connection artifacts
                sort_idx = np.argsort(distances)
                distances_sorted = distances[sort_idx]
                means_sorted = means[sort_idx]
                stds_sorted = stds[sort_idx]

                if distances_sorted.size:
                    max_dist_km = max(max_dist_km, distances_sorted.max() / 1000.0)

                # Plot median values as dashed lines (no markers)
                ax_mean.plot(distances_sorted/1000, means_sorted,
                             label=scenario_name, color=_code_color(scenario_name),
                             linestyle='--', alpha=0.8, linewidth=2)

                # Use intra-event variability directly (stats[:,1] from original DR4GM)
                cov = stds_sorted
                ax_cov.plot(distances_sorted/1000, cov,
                            label=scenario_name, color=_code_color(scenario_name),
                            linestyle='--', alpha=0.8, linewidth=2)

            except Exception as e:
                print(f"Warning: Could not load {scenario_info.source} for period {period}: {e}")
                continue
        
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
                    
                    # Convert GMPE distances back to meters for plotting
                    gmpe_distances_m = data['distances'] * 1000.0
                    
                    
                    # Convert GMPE distances to km
                    gmpe_distances_km = gmpe_distances_m / 1000.0
                    
                    # Plot NGA-West2 average 
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
                    
                    print(f"Added GMPE curve for RSA T={period_label}s (M{magnitude}, Vs30={vs30})")
                    
            except Exception as e:
                print(f"Warning: Could not add GMPE curves for RSA T={period_label}s: {e}")
        
        ax_mean.set_xlabel('Distance (km)', fontsize=14, fontweight='bold')
        ax_mean.set_ylabel(f'RSA T={period_label}s (g)', fontsize=14, fontweight='bold')
        ax_mean.set_yscale('log')
        ax_mean.set_xscale('log')
        ax_mean.grid(True, alpha=0.3)
        ax_mean.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
        ax_mean.tick_params(axis='both', which='major', labelsize=16)

        ax_cov.set_xlabel('Distance (km)', fontsize=14, fontweight='bold')
        ax_cov.set_ylabel(f'RSA T={period_label}s Intra-event Standard Deviation (g)', fontsize=14, fontweight='bold')
        ax_cov.set_xscale('log')
        ax_cov.set_ylim(0, 1)  # Set COV range from 0 to 1
        ax_cov.grid(True, alpha=0.3)
        ax_cov.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
        ax_cov.tick_params(axis='both', which='major', labelsize=16)

        if distance_range:
            ax_mean.set_xlim((distance_range[0]/1000, distance_range[1]/1000))
            ax_cov.set_xlim((distance_range[0]/1000, distance_range[1]/1000))
        else:
            xmax = max(max_dist_km, 1.0)
            ax_mean.set_xlim((1.0, xmax))
            ax_cov.set_xlim((1.0, xmax))
        
        # Set y-limits based on data range for this period
        if period in period_ranges:
            ax_mean.set_ylim(period_ranges[period])

        # Adjust layout: plot uses left 75% (6/8), legend on right 25% (2/8)
        fig_mean.subplots_adjust(left=0.15, right=0.75, top=0.9, bottom=0.15)
        fig_mean.savefig(f"{output_dir}/RSA_T{period_label}s_vs_distance.png", dpi=300, bbox_inches='tight')
        plt.close(fig_mean)

        # Adjust layout: plot uses left 75% (6/8), legend on right 25% (2/8)
        fig_cov.subplots_adjust(left=0.15, right=0.75, top=0.9, bottom=0.15)
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
    # Find closest distance bin for each scenario and collect RSA data
    scenario_rsa_data = {}
    all_periods = set()
    
    for scenario_info in scenarios:
        try:
            data = load_gm_statistics(scenario_info.gm_file)
            distances_km = data['rjb_distance_bins'] / 1000.0  # Convert to km
            
            # Find closest distance bin to target
            dist_diff = np.abs(distances_km - target_rjb_km)
            closest_idx = np.argmin(dist_diff)
            actual_rjb_km = distances_km[closest_idx]
            
            print(f"Scenario {scenario_info.label}: target {target_rjb_km}km → actual {actual_rjb_km:.1f}km")
            
            # Extract all RSA periods and values for this distance
            rsa_data = {}
            rsa_keys = [k for k in data.keys() if k.startswith('RSA_T_') and k.endswith('_mean')]
            
            for key in rsa_keys:
                period_str = key.replace('RSA_T_', '').replace('_mean', '')
                period_val = float(period_str.replace('_', '.'))
                
                # Get RSA value at closest distance
                # Check units to determine if conversion is needed
                rsa_units = data.get('RSA_units', 'g')
                if rsa_units == 'cm/s²':
                    rsa_mean = data[key][closest_idx] / 981.0  # Convert cm/s² to g
                else:
                    rsa_mean = data[key][closest_idx]  # Already in g
                rsa_std = data[key.replace('_mean', '_std')][closest_idx]
                rsa_count = data[key.replace('_mean', '_count')][closest_idx]
                
                # Only include if we have valid data
                if rsa_count > 0 and rsa_mean > 0:
                    rsa_data[period_val] = {
                        'mean': rsa_mean,
                        'std': rsa_std,
                        'count': rsa_count
                    }
                    all_periods.add(period_val)
            
            if rsa_data:
                scenario_rsa_data[scenario_info.label] = {
                    'data': rsa_data,
                    'actual_rjb_km': actual_rjb_km,
                    'scenario_info': scenario_info
                }
                
        except Exception as e:
            print(f"Warning: Could not process scenario {scenario_info.label}: {e}")
            continue
    
    if not scenario_rsa_data:
        print("Error: No valid RSA data found for any scenario")
        return
    
    # Sort periods for plotting
    periods_sorted = sorted(all_periods)
    
    # Create two plots: mean and std
    fig_mean, ax_mean = plt.subplots(figsize=(8, 6))
    fig_std, ax_std = plt.subplots(figsize=(8, 6))
    
    for i, (scenario_label, scenario_data) in enumerate(scenario_rsa_data.items()):
        periods_plot = []
        rsa_means_plot = []
        rsa_stds_plot = []
        
        for period in periods_sorted:
            if period in scenario_data['data']:
                periods_plot.append(period)
                rsa_means_plot.append(scenario_data['data'][period]['mean'])
                rsa_stds_plot.append(scenario_data['data'][period]['std'])
        
        if periods_plot:
            periods_plot = np.array(periods_plot)
            rsa_means_plot = np.array(rsa_means_plot)
            rsa_stds_plot = np.array(rsa_stds_plot)
            
            # Plot median line
            ax_mean.loglog(periods_plot, rsa_means_plot, '--', color=_code_color(scenario_label), 
                          label=f"{scenario_label}",
                          linewidth=2)
            
            # Plot standard deviation
            ax_std.loglog(periods_plot, rsa_stds_plot, '--', color=_code_color(scenario_label), 
                         label=f"{scenario_label}",
                         linewidth=2)
            
    
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
            
            # For std plot, get GMPE standard deviations
            gmpe_std_plot = []
            all_gmpe_stds = []
            
            for period in gmpe_periods_list:
                if period in gmpe_results:
                    data = gmpe_results[period]
                    
                    # Get NGA-West2 average std
                    gmpe_std_plot.append(data['NGA_AVG']['std'][0])
                    
                    # Collect all GMPE stds for envelope
                    period_stds = []
                    for gmpe in ['ASK', 'BSSA', 'CB', 'CY']:
                        std_val = data[gmpe]['std'][0]
                        period_stds.append(std_val)
                    all_gmpe_stds.append(period_stds)
            
            gmpe_std_plot = np.array(gmpe_std_plot)
            all_gmpe_stds = np.array(all_gmpe_stds)
            
            # Plot NGA-West2 average std (solid black line) on std plot
            ax_std.loglog(gmpe_periods_plot, gmpe_std_plot, 'black', linewidth=3, 
                         label=f'NGA-West2 Avg (M{magnitude})')
            
            # Black shaded area: range of 4 GMPE stds on std plot
            std_upper = np.max(all_gmpe_stds, axis=1)
            std_lower = np.min(all_gmpe_stds, axis=1)
            ax_std.fill_between(gmpe_periods_plot, std_lower, std_upper, 
                               color='black', alpha=0.2, label='GMPE Std Range')
            
        except Exception as e:
            print(f"Warning: Could not add GMPE curves: {e}")
    
    # Formatting for mean plot
    ax_mean.set_xlabel('Period (s)', fontsize=12, fontweight='bold')
    ax_mean.set_ylabel('Response Spectral Acceleration (g)', fontsize=12, fontweight='bold')
    ax_mean.grid(True, alpha=0.3)
    ax_mean.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Formatting for std plot
    ax_std.set_xlabel('Period (s)', fontsize=12, fontweight='bold')
    ax_std.set_ylabel('RSA Intra-event Standard Deviation (g)', fontsize=12, fontweight='bold')
    ax_std.grid(True, alpha=0.3)
    ax_std.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
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
            f.write(f"\nScenario: {scenario_label}\n")
            f.write(f"  Actual Rjb: {scenario_data['actual_rjb_km']:.2f} km\n")
            f.write(f"  Available periods: {len(scenario_data['data'])}\n")
            f.write(f"  Period range: {min(scenario_data['data'].keys()):.3f} - {max(scenario_data['data'].keys()):.3f} s\n")
    
    print(f"Summary saved: {summary_file}")


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
                        help='Add GMPE comparison curves and reference line at y=0.6 for COV plots')
    parser.add_argument('--plot-spectra-vs-periods', type=float, metavar='RJB_KM', default=10.0,
                        help='Generate response spectra vs periods plot at this Rjb (km). Default: 10.')
    
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

if __name__ == "__main__":
    main()
