#!/usr/bin/env python3

"""
SORD Plot Data Converter API

Converts SORD scenario datasets from plot files to NPZ format following the DR4GM standard.
This converter handles the specific format where spectral acceleration data is embedded 
in Python plot scripts.

Usage:
    python sord_plot_converter_api.py --input_dir <sord_dir> --output_dir <output_dir>
    
Example:
    python sord_plot_converter_api.py --input_dir datasets/sord --output_dir results/sord
"""

import os
import sys
import argparse
import numpy as np
import logging
import json
from typing import Dict, List, Tuple, Optional
import time
from pathlib import Path

# Import DR4GM standards
from npz_format_standard import DR4GM_NPZ_Standard

class SORDPlotConverter:
    """Convert SORD plot data to DR4GM gm_statistics.npz format"""
    
    def __init__(self, input_dir: str, output_dir: str):
        """
        Initialize converter
        
        Args:
            input_dir: Directory containing SORD plot files
            output_dir: Directory to save converted NPZ files
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Logging setup
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        
        # SORD data extracted from plot files
        self.distance_points = np.array([0.2, 0.5, 1, 2, 4, 8, 16, 20])  # km
        self.periods = np.array([3.03, 1.0, 0.333])  # seconds (1/0.33, 1/1, 1/3)
        
        # SA values for three periods (g) - from plot scripts SA1, SA2, SA3
        self.periods_sa = {
            3.03: np.array([0.126, 0.125, 0.102, 0.095, 0.071, 0.043, 0.032, 0.026]),  # SA1 -> T=3.03s
            1.0: np.array([0.38, 0.351, 0.331, 0.288, 0.201, 0.142, 0.081, 0.06]),     # SA2 -> T=1.0s  
            0.333: np.array([0.638, 0.605, 0.566, 0.458, 0.314, 0.21, 0.129, 0.082])  # SA3 -> T=0.333s
        }
        
        # Standard deviation data from plot files for three periods
        self.periods_std = {
            3.03: np.array([[2.616e-01, 2.971e-01, 3.792e-01, 4.426e-01, 4.946e-01, 5.269e-01, 6.431e-01, 6.595e-01],
                           [2.435e-01, 2.705e-01, 3.332e-01, 4.878e-01, 5.343e-01, 5.717e-01, 6.015e-01, 6.403e-01],
                           [2.562e-01, 3.030e-01, 3.532e-01, 4.667e-01, 5.063e-01, 5.563e-01, 6.253e-01, 6.338e-01],
                           [2.695e-01, 2.800e-01, 3.614e-01, 4.758e-01, 4.943e-01, 5.587e-01, 6.174e-01, 6.709e-01],
                           [2.887e-01, 2.640e-01, 3.509e-01, 4.740e-01, 5.165e-01, 5.615e-01, 6.049e-01, 6.336e-01]]),
            
            1.0: np.array([[3.431e-01, 3.486e-01, 4.283e-01, 4.981e-01, 5.071e-01, 6.102e-01, 6.856e-01, 6.973e-01],
                          [3.445e-01, 3.217e-01, 4.320e-01, 4.714e-01, 4.793e-01, 5.697e-01, 6.787e-01, 6.951e-01],
                          [3.201e-01, 3.450e-01, 4.526e-01, 4.615e-01, 5.047e-01, 6.101e-01, 7.111e-01, 6.581e-01],
                          [3.464e-01, 3.661e-01, 4.506e-01, 4.639e-01, 4.726e-01, 5.950e-01, 7.065e-01, 6.749e-01],
                          [3.674e-01, 3.578e-01, 4.695e-01, 4.924e-01, 5.067e-01, 5.900e-01, 6.738e-01, 6.925e-01]]),
            
            0.333: np.array([[2.946e-01, 4.131e-01, 3.994e-01, 4.930e-01, 6.014e-01, 5.673e-01, 6.954e-01, 7.527e-01],
                            [3.011e-01, 3.849e-01, 4.075e-01, 5.027e-01, 5.681e-01, 5.573e-01, 6.774e-01, 7.379e-01],
                            [2.633e-01, 3.709e-01, 4.300e-01, 5.307e-01, 5.635e-01, 5.804e-01, 7.099e-01, 7.475e-01],
                            [2.698e-01, 4.055e-01, 4.411e-01, 5.201e-01, 5.649e-01, 5.961e-01, 7.211e-01, 7.466e-01],
                            [2.707e-01, 3.893e-01, 4.189e-01, 4.935e-01, 5.879e-01, 5.566e-01, 7.092e-01, 7.818e-01]])
        }
    
    def _create_station_grid(self, scenario_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Create station grid around fault for scenario"""
        
        # Create a grid of stations at the specified distance points
        # Place them along a line perpendicular to the fault (fault-normal direction)
        
        num_stations = len(self.distance_points)
        station_ids = np.arange(num_stations)
        
        # Create locations array [x, y, z] in meters
        # Fault runs N-S (along Y-axis), so place stations in E-W direction (X-axis, fault-normal)
        locations = np.zeros((num_stations, 3))
        locations[:, 0] = self.distance_points * 1000.0  # Fault-normal distance (x) in meters
        locations[:, 1] = 0.0  # Along-fault (y) = 0, centered on fault
        locations[:, 2] = 0.0  # Surface stations (z) = 0
        
        self.logger.info(f"Created {num_stations} stations for {scenario_name}")
        self.logger.info(f"Distance range: {self.distance_points[0]:.1f} - {self.distance_points[-1]:.1f} km")
        
        return station_ids, locations
    
    def _calculate_ground_motion_metrics(self, scenario_name: str, sa_values: np.ndarray, std_values: np.ndarray) -> Dict:
        """Calculate ground motion metrics from SA values"""
        
        station_ids, locations = self._create_station_grid(scenario_name)
        num_stations = len(station_ids)
        num_periods = len(self.periods)
        
        # Calculate RJB distances (same as distance points since fault is a line)
        rjb_distances = self.distance_points * 1000.0  # Convert km to meters
        
        # Initialize spectral acceleration arrays
        # SA data is for a single period (assume period index 0 for simplicity)
        sa_array = np.zeros((num_stations, num_periods))
        sa_array[:, 0] = sa_values  # Use the SA values for the first period
        
        # For other periods, use scaled values based on typical period scaling
        if num_periods > 1:
            sa_array[:, 1] = sa_values * 0.8  # Rough scaling for T=1.0s
        if num_periods > 2:
            sa_array[:, 2] = sa_values * 0.6  # Rough scaling for T=0.33s
        
        # Convert SA from 'g' to cm/s² (multiply by 981 cm/s²/g)
        sa_array_cm_s2 = sa_array * 981.0  # Convert to cm/s²
        
        # Calculate PGA (peak ground acceleration) - use maximum SA value
        pga = np.max(sa_array_cm_s2, axis=1)

        # Calculate PGV (peak ground velocity) - empirical relationship from SA
        # PGV ≈ SA * T / (2π) (rough approximation, SA already in cm/s²)
        pgv = sa_array_cm_s2[:, 0] * 1.0 / (2 * np.pi)  # Use T=1.0s for PGV estimation
        
        # Calculate PGD (peak ground displacement) - empirical relationship
        # PGD ≈ PGV * T / (2π) (rough approximation)
        pgd = pgv * 1.0 / (2 * np.pi)
        
        # Use average standard deviation for the scenario
        avg_std = np.mean(std_values, axis=0)
        
        gm_metrics = {
            'station_ids': station_ids,
            'locations': locations,
            'rjb_distances': rjb_distances,
            'pga': pga,
            'pgv': pgv,
            'pgd': pgd,
            'sa_values': sa_array_cm_s2,
            'periods': self.periods,
            'std_log_sa': np.tile(avg_std, (num_periods, 1)).T,  # Replicate for all periods
            'coordinate_units': 'm',
            'sa_units': 'cm/s²',
            'pgv_units': 'cm/s',
            'pgd_units': 'cm',
            'scenario_name': scenario_name
        }
        
        self.logger.info(f"Ground motion metrics calculated for {scenario_name}:")
        self.logger.info(f"  PGA range: {np.min(pga):.3f} - {np.max(pga):.3f} g")
        self.logger.info(f"  PGV range: {np.min(pgv):.3f} - {np.max(pgv):.3f} m/s")
        
        return gm_metrics
    
    def convert_all_scenarios(self) -> Dict[str, str]:
        """Convert SORD data to single scenario with multiple periods"""
        start_time = time.time()
        self.logger.info(f"Starting conversion of SORD plot data from {self.input_dir}")
        
        output_files = {}
        
        # Create single scenario with all periods
        scenario_name = "sord_scenario"
        self.logger.info(f"Processing {scenario_name} with {len(self.periods)} periods")
        
        # Create scenario directory
        scenario_output_dir = self.output_dir / scenario_name
        scenario_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create gm_statistics.npz with all periods
        statistics_file = self._create_gm_statistics_npz(scenario_name, scenario_output_dir)
        geometry_file = self._create_fault_geometry_npz(scenario_output_dir)
        
        output_files[scenario_name] = {
            'gm_statistics': statistics_file,
            'fault_geometry': geometry_file
        }
        
        duration = time.time() - start_time
        
        self.logger.info(f"Conversion complete in {duration:.2f}s")
        self.logger.info(f"Processed 1 scenario with {len(self.periods)} periods")
        self.logger.info("Created gm_statistics.npz files directly from SORD plot data")
        
        return output_files
    
    def _create_gm_statistics_npz(self, scenario_name: str, output_dir: Path) -> str:
        """Create gm_statistics.npz file with data for all SORD periods"""
        
        output_file = output_dir / "gm_statistics.npz"
        
        # SORD distance points are Rjb distances in km, convert to meters
        rjb_distances_m = self.distance_points * 1000.0
        
        # Create R bins (distance ranges) for statistics - use SORD distance points as bin centers
        n_bins = len(self.distance_points)
        r_bins = np.zeros(n_bins + 1)
        
        # Create bins around each distance point (properly)
        # First edge at 0
        r_bins[0] = 0.0
        
        # Middle edges at midpoints between consecutive distance points
        for i in range(n_bins - 1):
            r_bins[i + 1] = (self.distance_points[i] + self.distance_points[i + 1]) / 2 * 1000.0
        
        # Last edge extends beyond the last point
        r_bins[n_bins] = self.distance_points[-1] * 1.5 * 1000.0
        
        # For PGA/PGV/PGD, use the T=1.0s period data as representative
        reference_sa_values = self.periods_sa[1.0]  # Use T=1.0s data
        reference_std_values = self.periods_std[1.0]  # Use T=1.0s std data
        
        # Average standard deviation across realizations (as done in original plot files)
        avg_std = np.mean(reference_std_values, axis=0)
        
        # Create ground motion statistics arrays following gm_stats.py format
        # Each row is a distance bin, columns are [geom_mean, log_std, min, max, count, r_center]
        pga_stats = np.zeros((n_bins, 6))
        pgv_stats = np.zeros((n_bins, 6))  
        pgd_stats = np.zeros((n_bins, 6))
        
        # Convert SA values from 'g' to cm/s² for consistent units
        sa_values_cm_s2 = reference_sa_values * 981.0  # Convert from g to cm/s²
        
        # Empirical relationships to estimate PGV and PGD from SA
        # PGV ≈ SA * T / (2π), PGD ≈ PGV * T / (2π) (SA already in cm/s²)
        pgv_values = sa_values_cm_s2 * 1.0 / (2 * np.pi)  # Using T=1.0s
        pgd_values = pgv_values * 1.0 / (2 * np.pi)
        
        for i in range(n_bins):
            # Distance bin center
            r_center = rjb_distances_m[i]
            
            # PGA statistics (use SA values in cm/s² as PGA approximation)
            pga_stats[i, 0] = sa_values_cm_s2[i]     # Geometric mean (SA value in cm/s²)
            pga_stats[i, 1] = avg_std[i]             # Log standard deviation
            pga_stats[i, 2] = sa_values_cm_s2[i] * 0.5  # Min (rough estimate)
            pga_stats[i, 3] = sa_values_cm_s2[i] * 2.0  # Max (rough estimate)
            pga_stats[i, 4] = 100                    # Count (synthetic)
            pga_stats[i, 5] = r_center               # Distance bin center
            
            # PGV statistics
            pgv_stats[i, 0] = pgv_values[i]
            pgv_stats[i, 1] = avg_std[i]
            pgv_stats[i, 2] = pgv_values[i] * 0.5
            pgv_stats[i, 3] = pgv_values[i] * 2.0
            pgv_stats[i, 4] = 100
            pgv_stats[i, 5] = r_center
            
            # PGD statistics  
            pgd_stats[i, 0] = pgd_values[i]
            pgd_stats[i, 1] = avg_std[i]
            pgd_stats[i, 2] = pgd_values[i] * 0.5
            pgd_stats[i, 3] = pgd_values[i] * 2.0
            pgd_stats[i, 4] = 100
            pgd_stats[i, 5] = r_center
        
        # Create statistics dictionary in the expected format (separate arrays)
        statistics_data = {
            # Distance information
            'rjb_distance_bins': rjb_distances_m,  # Distance bin centers
            'distance_bin_edges': r_bins,           # Distance bin edges  
            'distance_units': 'm',
            
            # Units
            'PGA_units': 'cm/s²',
            'PGV_units': 'cm/s', 
            'PGD_units': 'cm',
            'CAV_units': 'cm/s',  # Placeholder
            'RSA_units': 'cm/s²',
            
            # PGA statistics (separate arrays)
            'PGA_mean': pga_stats[:, 0],    # Geometric mean
            'PGA_std': pga_stats[:, 1],     # Log standard deviation
            'PGA_min': pga_stats[:, 2],     # Minimum
            'PGA_max': pga_stats[:, 3],     # Maximum 
            'PGA_count': pga_stats[:, 4].astype(int),  # Count
            
            # PGV statistics  
            'PGV_mean': pgv_stats[:, 0],
            'PGV_std': pgv_stats[:, 1],
            'PGV_min': pgv_stats[:, 2],
            'PGV_max': pgv_stats[:, 3],
            'PGV_count': pgv_stats[:, 4].astype(int),
            
            # PGD statistics
            'PGD_mean': pgd_stats[:, 0],
            'PGD_std': pgd_stats[:, 1], 
            'PGD_min': pgd_stats[:, 2],
            'PGD_max': pgd_stats[:, 3],
            'PGD_count': pgd_stats[:, 4].astype(int),
            
            # CAV placeholder (not available in SORD data)
            'CAV_mean': np.zeros(n_bins),
            'CAV_std': np.zeros(n_bins),
            'CAV_min': np.zeros(n_bins),
            'CAV_max': np.zeros(n_bins), 
            'CAV_count': np.zeros(n_bins, dtype=int)
        }
        
        # Add RSA data for the three SORD periods using real data
        for period_seconds in self.periods:
            # Get real SA values and std data for this period
            sa_values_period = self.periods_sa[period_seconds]
            std_values_period = self.periods_std[period_seconds]
            
            # Convert to cm/s² and compute average std
            sa_values_cm_s2_period = sa_values_period * 981.0
            avg_std_period = np.mean(std_values_period, axis=0)
            
            # Format period for key (e.g., 3.03 -> "3_030", 1.0 -> "1_000", 0.333 -> "0_333")
            period_str = f"{period_seconds:.3f}".replace('.', '_')
            period_key = f"RSA_T_{period_str}"
            
            # Use real SORD data for each period
            statistics_data[f"{period_key}_mean"] = sa_values_cm_s2_period
            statistics_data[f"{period_key}_std"] = avg_std_period  
            statistics_data[f"{period_key}_min"] = sa_values_cm_s2_period * 0.5
            statistics_data[f"{period_key}_max"] = sa_values_cm_s2_period * 2.0
            statistics_data[f"{period_key}_count"] = np.full(n_bins, 100, dtype=int)
        
        np.savez_compressed(output_file, **statistics_data)
        
        self.logger.info(f"Ground motion statistics NPZ created: {output_file}")
        self.logger.info(f"  PGA range: {np.min(sa_values_cm_s2):.1f} - {np.max(sa_values_cm_s2):.1f} cm/s²")
        self.logger.info(f"  Distance range: {np.min(rjb_distances_m/1000):.1f} - {np.max(rjb_distances_m/1000):.1f} km (Rjb)")
        
        return str(output_file)
    
    def _create_ground_motion_metrics_npz(self, gm_metrics: Dict, output_dir: Path) -> str:
        """Create ground motion metrics NPZ file"""
        
        output_file = output_dir / "ground_motion_metrics.npz"
        
        # Prepare data for NPZ file (use uppercase for metric names to match pipeline expectations)
        npz_data = {
            'station_ids': gm_metrics['station_ids'],
            'locations': gm_metrics['locations'],
            'rjb_distances': gm_metrics['rjb_distances'],
            'PGA': gm_metrics['pga'],
            'PGV': gm_metrics['pgv'],
            'PGD': gm_metrics['pgd'],
            'sa_values': gm_metrics['sa_values'],
            'periods': gm_metrics['periods'],
            'std_log_sa': gm_metrics['std_log_sa'],
            'coordinate_units': gm_metrics['coordinate_units'],
            'sa_units': gm_metrics['sa_units'],
            'pgv_units': gm_metrics['pgv_units'],
            'pgd_units': gm_metrics['pgd_units'],
            'scenario_name': gm_metrics['scenario_name']
        }
        
        np.savez_compressed(output_file, **npz_data)
        
        self.logger.info(f"Ground motion metrics NPZ created: {output_file}")
        return str(output_file)
    
    def _create_fault_geometry_npz(self, output_dir: Path) -> str:
        """Create fault geometry NPZ file"""
        
        # Define fault geometry for SORD scenarios
        # Assume a simple vertical strike-slip fault running N-S
        fault_data = {
            'fault_type': 'strike_slip',
            'fault_trace_start': np.array([0.0, -20000.0, 0.0], dtype=np.float32),  # 20 km south
            'fault_trace_end': np.array([0.0, 20000.0, 0.0], dtype=np.float32),     # 20 km north
            'fault_dip': np.float32(90.0),                # Vertical fault
            'fault_strike': np.float32(0.0),              # N-S strike
            'fault_length': np.float32(40000.0),          # 40 km total length
            'fault_width': np.float32(15000.0),           # 15 km width (depth)
            'top_depth': np.float32(0.0),                 # Surface rupture
            'bottom_depth': np.float32(15000.0),          # 15 km depth
            'moment_magnitude': np.float32(6.9),          # From M6.9_001 directory name
            'coordinate_units': 'm',
            'description': 'SORD M6.9 strike-slip fault scenario',
            'rotation_applied': 'no'
        }
        
        output_file = output_dir / "geometry.npz"
        np.savez_compressed(output_file, **fault_data)
        
        self.logger.info(f"Fault geometry NPZ created: {output_file}")
        return str(output_file)
    
    def _create_combined_scenarios_npz(self, all_scenarios: Dict) -> str:
        """Create combined NPZ file with all scenarios"""
        
        output_file = self.output_dir / "combined_scenarios.npz"
        
        # Combine all scenario data
        combined_data = {}
        for scenario_name, gm_metrics in all_scenarios.items():
            prefix = scenario_name.replace('scenario_', 's')
            
            for key, value in gm_metrics.items():
                if key != 'scenario_name':
                    combined_key = f"{prefix}_{key}"
                    combined_data[combined_key] = value
        
        # Add metadata
        combined_data['scenario_names'] = list(all_scenarios.keys())
        combined_data['num_scenarios'] = len(all_scenarios)
        combined_data['distance_points_km'] = self.distance_points
        
        np.savez_compressed(output_file, **combined_data)
        
        self.logger.info(f"Combined scenarios NPZ created: {output_file}")
        return str(output_file)
    
    def _create_pipeline_stations_npz(self, gm_metrics: Dict) -> str:
        """Create pipeline-compatible stations.npz file"""
        
        output_file = self.output_dir / "stations.npz"
        
        # Create stations data in standard format
        stations_data = {
            'station_ids': gm_metrics['station_ids'],
            'locations': gm_metrics['locations'],
            'rjb_distances': gm_metrics['rjb_distances'],
            'station_types': np.ones(len(gm_metrics['station_ids'])),  # All surface stations
            'coordinate_units': gm_metrics['coordinate_units']
        }
        
        np.savez_compressed(output_file, **stations_data)
        
        self.logger.info(f"Pipeline stations NPZ created: {output_file}")
        return str(output_file)
    
    def _create_pipeline_velocities_npz(self, gm_metrics: Dict) -> str:
        """Create pipeline-compatible velocities.npz file using ground motion data"""
        
        output_file = self.output_dir / "velocities.npz"
        
        # Since SORD data is spectral acceleration, we'll create synthetic velocity time series
        # based on the calculated PGV values
        num_stations = len(gm_metrics['station_ids'])
        nt = 1000  # Number of time steps
        dt = 0.01  # Time step in seconds
        
        # Create simple sinusoidal velocity time series scaled by PGV
        time_array = np.arange(nt) * dt
        
        # Initialize velocity arrays
        vel_strike = np.zeros((num_stations, nt))
        vel_normal = np.zeros((num_stations, nt))
        vel_vertical = np.zeros((num_stations, nt))
        
        for i, pgv in enumerate(gm_metrics['pgv']):
            # Create a decaying sinusoidal signal with peak = PGV
            freq = 1.0  # 1 Hz dominant frequency
            decay = np.exp(-time_array / 5.0)  # 5 second decay
            vel_strike[i, :] = pgv * np.sin(2 * np.pi * freq * time_array) * decay
        
        velocities_data = {
            'station_ids': gm_metrics['station_ids'],
            'locations': gm_metrics['locations'],
            'vel_strike': vel_strike,
            'vel_normal': vel_normal,
            'vel_vertical': vel_vertical,
            'time_steps': np.full(num_stations, nt),
            'dt_values': np.full(num_stations, dt),
            'duration': nt * dt,
            'units': 'm/s',
            'coordinate_units': gm_metrics['coordinate_units']
        }
        
        np.savez_compressed(output_file, **velocities_data)
        
        self.logger.info(f"Pipeline velocities NPZ created: {output_file}")
        return str(output_file)

def main():
    """Main entry point for the converter API"""
    parser = argparse.ArgumentParser(description='Convert SORD plot data to DR4GM NPZ format')
    parser.add_argument('--input_dir', required=True, help='Input directory containing SORD plot files')
    parser.add_argument('--output_dir', required=True, help='Output directory for NPZ files')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        converter = SORDPlotConverter(args.input_dir, args.output_dir)
        results = converter.convert_all_scenarios()
        
        print("\n=== Conversion Results ===")
        for scenario_name, files in results.items():
            print(f"\n{scenario_name.upper()}:")
            print(f"  Ground motion statistics: {files['gm_statistics']}")
            print(f"  Fault geometry: {files['fault_geometry']}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
