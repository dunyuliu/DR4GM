#!/usr/bin/env python3

"""
Ground Motion Statistics Script for DR4GM

Computes statistics of simulation ground motion results as a function of Rjb distance.
Calculates mean, standard deviation, min, max, and variability for ground motion metrics
binned by Joyner-Boore distance.

Usage:
    python gm_stats.py --gm_data <ground_motion_metrics.npz> --output_dir <output_dir> 
    
Example:
    python gm_stats.py --gm_data eqdyna.0001.A.100m/ground_motion_metrics.npz --output_dir ./gm_statistics
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import pandas as pd

class GMStatistics:
    """Compute ground motion statistics as a function of Rjb distance"""
    
    def __init__(self, gm_data_file: str, output_dir: str, distance_range: Tuple[float, float] = (0, 30000), distance_bin_size: float = 1000):
        """
        Initialize GM statistics computation
        
        Args:
            gm_data_file: Path to ground motion metrics NPZ file
            output_dir: Directory to save statistics results
            distance_range: Distance range in meters (default: 0-30km)
            distance_bin_size: Distance bin size in meters (default: 1km)
        """
        self.gm_data_file = Path(gm_data_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if not self.gm_data_file.exists():
            raise FileNotFoundError(f"Ground motion data file not found: {gm_data_file}")
        
        # Distance binning parameters
        self.distance_range = distance_range
        self.distance_bin_size = distance_bin_size
        
        # Load ground motion data
        self.logger = self._setup_logging()
        self.logger.info(f"Loading ground motion data from {gm_data_file}")
        
        self.gm_data = np.load(self.gm_data_file)
        self.station_ids = self.gm_data['station_ids']
        self.locations = self.gm_data['locations']
        
        # Calculate Rjb distances from fault (in meters, not km)
        self.rjb_distances = self._calculate_rjb_distances()
        
        # Define earthquake parameters (adjust based on your scenario)
        self.earthquake_params = {
            'magnitude': 7.0,  # Moment magnitude
            'vs30': 760,       # Average shear wave velocity (m/s)
            'style_of_faulting': 'strike-slip',  # strike-slip, normal, reverse
            'depth_to_top': 0.0,  # Depth to top of rupture (km)
            'dip': 90.0,       # Fault dip (degrees)
            'width': 15.0,     # Fault width (km)
        }
        
        # Available periods for SA comparison  
        available_periods = [float(key.split('_')[-1]) for key in self.gm_data.keys() 
                           if key.startswith('RSA_T_')]
        self.periods = np.array(sorted(available_periods)) if available_periods else np.array([0.1, 0.2, 0.5, 1.0, 2.0, 5.0])
        
        self.logger.info(f"Loaded {len(self.station_ids)} stations")
        self.logger.info(f"Rjb distance range: {self.rjb_distances.min():.0f} - {self.rjb_distances.max():.0f} m")
        self.logger.info(f"Available GM metrics: {list(self.gm_data.keys())}")
    
    def _setup_logging(self):
        """Setup logging"""
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        return logging.getLogger(__name__)
    
    def _load_fault_geometry(self) -> Dict:
        """Load fault geometry from geometry.npz file"""
        # Look for geometry.npz in the same directory as the ground motion data file
        geometry_file = self.gm_data_file.parent / "geometry.npz"
        
        if not geometry_file.exists():
            raise FileNotFoundError(f"Fault geometry file not found: {geometry_file}")
        
        try:
            geometry_data = np.load(geometry_file)
            fault_info = {
                'fault_type': str(geometry_data['fault_type']),
                'fault_trace_start': geometry_data['fault_trace_start'],
                'fault_trace_end': geometry_data['fault_trace_end'],
                'fault_dip': float(geometry_data['fault_dip']),
                'fault_strike': float(geometry_data['fault_strike']),
                'fault_length': float(geometry_data['fault_length']),
                'rotation_applied': str(geometry_data.get('rotation_applied', 'none')),
                'description': str(geometry_data['description'])
            }
            self.logger.info(f"Loaded fault geometry: {fault_info['description']}")
            return fault_info
        except Exception as e:
            self.logger.error(f"Failed to load fault geometry: {e}")
            raise
    
    def _point_to_line_distance(self, point: np.ndarray, line_start: np.ndarray, line_end: np.ndarray) -> float:
        """Calculate shortest distance from a point to a line segment"""
        # Vector from line start to end
        line_vec = line_end - line_start
        # Vector from line start to point
        point_vec = point - line_start
        
        # Length of line segment
        line_length_sq = np.dot(line_vec, line_vec)
        
        if line_length_sq == 0:
            # Line is a point, return distance to that point
            return np.linalg.norm(point - line_start)
        
        # Project point onto line
        t = max(0, min(1, np.dot(point_vec, line_vec) / line_length_sq))
        
        # Find closest point on line segment
        projection = line_start + t * line_vec
        
        # Return distance from point to closest point on line
        return np.linalg.norm(point - projection)
    
    def _calculate_rjb_distances(self) -> np.ndarray:
        """
        Calculate Joyner-Boore (Rjb) distances from fault line segment for each station
        Following gmFuncLib.py calcRjb logic but adapted to use geometry loaded from geometry.npz
        """
        # Load fault geometry
        fault_info = self._load_fault_geometry()
        
        # Extract fault trace endpoints (use only X,Y coordinates for 2D distance)
        fault_start = fault_info['fault_trace_start'][:2]  # [x, y]
        fault_end = fault_info['fault_trace_end'][:2]      # [x, y]
        
        self.logger.info(f"Computing Rjb distances following gmFuncLib.py logic")
        self.logger.info(f"Fault line: ({fault_start[0]:.0f}, {fault_start[1]:.0f}) to ({fault_end[0]:.0f}, {fault_end[1]:.0f})")
        
        # Determine fault orientation to apply appropriate calcRjb logic
        # Check if fault is more horizontal (constant Y) or vertical (constant X)
        dx = abs(fault_end[0] - fault_start[0])
        dy = abs(fault_end[1] - fault_start[1])
        
        # Calculate Rjb for each station following gmFuncLib.py calcRjb logic
        rjb_distances = np.zeros(len(self.locations))
        
        if dx > dy:
            # Horizontal fault (fault runs along X-axis, like original gmFuncLib with strike-slip)
            fault_xmin = min(fault_start[0], fault_end[0])
            fault_xmax = max(fault_start[0], fault_end[0])
            
            self.logger.info(f"Horizontal fault detected: X from {fault_xmin:.0f} to {fault_xmax:.0f}")
            
            for i, location in enumerate(self.locations):
                x, y = location[0], location[1]
                
                # Follow gmFuncLib.py calcRjb logic exactly
                if x <= fault_xmin:
                    # Station is west of fault start - distance to start point
                    rjb_distances[i] = ((x - fault_xmin)**2 + (y - fault_start[1])**2)**0.5
                elif x >= fault_xmax:
                    # Station is east of fault end - distance to end point  
                    rjb_distances[i] = ((x - fault_xmax)**2 + (y - fault_end[1])**2)**0.5
                else:
                    # Station is between fault endpoints - perpendicular distance to fault line
                    rjb_distances[i] = abs(y - fault_start[1])  # Distance to fault line at constant Y
        else:
            # Vertical fault (fault runs along Y-axis)
            fault_ymin = min(fault_start[1], fault_end[1])
            fault_ymax = max(fault_start[1], fault_end[1])
            
            self.logger.info(f"Vertical fault detected: Y from {fault_ymin:.0f} to {fault_ymax:.0f}")
            
            for i, location in enumerate(self.locations):
                x, y = location[0], location[1]
                
                # Adapt gmFuncLib.py logic for vertical fault (Y-direction)
                if y <= fault_ymin:
                    # Station is south of fault start - distance to start point
                    rjb_distances[i] = ((x - fault_start[0])**2 + (y - fault_ymin)**2)**0.5
                elif y >= fault_ymax:
                    # Station is north of fault end - distance to end point
                    rjb_distances[i] = ((x - fault_end[0])**2 + (y - fault_ymax)**2)**0.5
                else:
                    # Station is between fault endpoints - perpendicular distance to fault line
                    rjb_distances[i] = abs(x - fault_start[0])  # Distance to fault line at constant X
        
        # Add small minimum distance to avoid zero distances
        rjb_distances = np.maximum(rjb_distances, 100.0)  # Minimum 100 m
        
        self.logger.info(f"Rjb distances calculated: {rjb_distances.min():.0f} to {rjb_distances.max():.0f} m")
        
        return rjb_distances
    
    
    def calc_gm_stats_vs_r(self, values: np.ndarray, rjb_distances: np.ndarray, 
                          r_bins: np.ndarray) -> np.ndarray:
        """
        Calculate ground motion statistics vs distance bins (adapted from gmGetSimuAndGMPEScaling)
        
        Args:
            values: 1D array of GM values for all stations
            rjb_distances: 1D array of distances for all stations (in meters)
            r_bins: 1D array of distance bin edges (in meters)
            
        Returns:
            stats array (len(r_bins)-1, 6): [geometric_mean, log_std, min, max, count, unused]
        """
        n_bins = len(r_bins) - 1
        gm_metrics_vs_r = np.zeros((n_bins, 50000))  # Pre-allocate like original
        log_gm_metrics = np.zeros((n_bins, 50000))
        gm_metrics_stats = np.zeros((n_bins, 6))
        num_of_gm_metrics_per_r = np.zeros(n_bins, dtype=int)

        # Bin the data (adapted from original nested loop structure)
        for i_st in range(len(values)):
            for i_r in range(n_bins):
                if (rjb_distances[i_st] >= r_bins[i_r] and 
                    rjb_distances[i_st] < r_bins[i_r + 1] and 
                    abs(values[i_st]) > 0.):
                    gm_metrics_vs_r[i_r, num_of_gm_metrics_per_r[i_r]] = values[i_st]
                    num_of_gm_metrics_per_r[i_r] += 1

        # Calculate statistics (exact copy from original)
        for i_r in range(n_bins):
            sum_log_x = 0
            sum_sq_log_x = 0
            for i_s in range(num_of_gm_metrics_per_r[i_r]):
                log_gm_metrics[i_r, i_s] = np.log(gm_metrics_vs_r[i_r, i_s])
                sum_log_x = sum_log_x + log_gm_metrics[i_r, i_s]
                sum_sq_log_x = sum_sq_log_x + (log_gm_metrics[i_r, i_s])**2

            if num_of_gm_metrics_per_r[i_r] > 1:
                mean_log_x = sum_log_x / num_of_gm_metrics_per_r[i_r]
                mean_sq_log_x = sum_sq_log_x / (num_of_gm_metrics_per_r[i_r] - 1)
                std_log_x = (mean_sq_log_x - mean_log_x**2 * num_of_gm_metrics_per_r[i_r] / 
                           (num_of_gm_metrics_per_r[i_r] - 1))**0.5
                min_x = np.min(gm_metrics_vs_r[i_r, :num_of_gm_metrics_per_r[i_r]])
                max_x = np.max(gm_metrics_vs_r[i_r, :num_of_gm_metrics_per_r[i_r]])
            else:
                mean_log_x = 0.
                std_log_x = 0.
                min_x = 0.
                max_x = 0.

            gm_metrics_stats[i_r, 0] = np.exp(mean_log_x)  # Geometric mean
            gm_metrics_stats[i_r, 1] = std_log_x           # Log standard deviation 
            gm_metrics_stats[i_r, 2] = min_x               # Minimum
            gm_metrics_stats[i_r, 3] = max_x               # Maximum
            gm_metrics_stats[i_r, 4] = num_of_gm_metrics_per_r[i_r]  # Count

        return gm_metrics_stats
    
    def compute_all_gm_statistics(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Compute statistics for all available ground motion metrics as a function of Rjb distance
        
        Returns:
            Dictionary with GM metric names as keys, each containing statistics arrays
        """
        self.logger.info("Computing ground motion statistics for all metrics")
        
        # Define distance bins
        r_bin_range = self.distance_range
        r_bin_size = self.distance_bin_size
        n_bins = int((r_bin_range[1] - r_bin_range[0]) / r_bin_size)
        r_bins = np.linspace(r_bin_range[0], r_bin_range[1], n_bins + 1)
        r_bin_centers = r_bins[:-1] + r_bin_size / 2.0
        
        # Available GM metrics to process
        gm_metrics = ['PGA', 'PGV', 'PGD', 'CAV']
        
        # Add spectral acceleration periods
        sa_keys = [key for key in self.gm_data.keys() if key.startswith('RSA_T_')]
        gm_metrics.extend(sa_keys)
        
        all_statistics = {}
        
        for metric in gm_metrics:
            if metric not in self.gm_data:
                self.logger.warning(f"Metric {metric} not found in data, skipping")
                continue
                
            self.logger.info(f"Processing {metric}")
            
            # Get the data for this metric
            metric_data = self.gm_data[metric]
            
            # No unit conversion needed - ground_motion_metrics.npz already has correct units
            # PGA/RSA are already in cm/s², PGV in cm/s, PGD in cm from npz_gm_processor.py
            # Removing incorrect double conversion that was multiplying by 100
            
            # Calculate statistics using the existing function
            stats = self.calc_gm_stats_vs_r(metric_data, self.rjb_distances, r_bins)
            
            # Store results
            all_statistics[metric] = {
                'rjb_distance': r_bin_centers,
                'geometric_mean': stats[:, 0],
                'log_std': stats[:, 1],
                'min': stats[:, 2],
                'max': stats[:, 3],
                'count': stats[:, 4].astype(int),
                'bins': r_bins
            }
        
        return all_statistics
    
    def save_statistics_npz(self, statistics: Dict[str, Dict[str, np.ndarray]], filename: str = 'gm_statistics.npz') -> None:
        """
        Save all statistics to a single NPZ file with unit information
        
        Args:
            statistics: Dictionary of statistics from compute_all_gm_statistics
            filename: Output filename
        """
        save_dict = {}
        
        # Add distance bins (same for all metrics)
        first_metric = next(iter(statistics.values()))
        save_dict['rjb_distance_bins'] = first_metric['rjb_distance']
        save_dict['distance_bin_edges'] = first_metric['bins']
        
        # Add units information
        save_dict['distance_units'] = 'm'  # distances in meters
        save_dict['PGA_units'] = 'cm/s²'
        save_dict['PGV_units'] = 'cm/s'
        save_dict['PGD_units'] = 'cm'
        save_dict['CAV_units'] = 'cm·s'
        save_dict['RSA_units'] = 'cm/s²'
        
        # Add statistics for each metric
        for metric, stats in statistics.items():
            prefix = metric.replace('.', '_')  # Replace dots for valid keys
            save_dict[f'{prefix}_mean'] = stats['geometric_mean']
            save_dict[f'{prefix}_std'] = stats['log_std']
            save_dict[f'{prefix}_min'] = stats['min']
            save_dict[f'{prefix}_max'] = stats['max']
            save_dict[f'{prefix}_count'] = stats['count']
        
        # Save to NPZ file
        output_path = self.output_dir / filename
        np.savez_compressed(output_path, **save_dict)
        self.logger.info(f"Statistics saved to: {output_path}")
    
    def calc_inter_event_variability(self, observed_means: np.ndarray, predicted_means: np.ndarray) -> float:
        """
        Calculate inter-event variability (between-event standard deviation)
        This represents epistemic uncertainty between the simulation and GMPE
        """
        # Remove invalid data points
        valid_mask = (observed_means > 0) & (predicted_means > 0) & np.isfinite(observed_means) & np.isfinite(predicted_means)
        
        if np.sum(valid_mask) < 2:
            return 0.0
        
        obs_valid = observed_means[valid_mask]
        pred_valid = predicted_means[valid_mask]
        
        # Calculate residuals in log space
        residuals = np.log(obs_valid / pred_valid)
        
        # Inter-event variability is the standard deviation of residuals
        inter_event_std = np.std(residuals, ddof=1)
        
        return inter_event_std
    
    
    def calculate_residuals(self) -> Dict[str, np.ndarray]:
        """Calculate residuals between simulation and GMPE predictions"""
        self.logger.info("Calculating residuals")
        
        # Get simulation PGA data
        sim_pga = self.gm_data['PGA'] * 100 / 981.0  # Convert to g units
        
        # Calculate GMPE predictions for each station
        eq_params = self.earthquake_params
        bssa_pga = self.boore_stewart_seyhan_atkinson_2014_pga(
            self.rjb_distances, eq_params['magnitude'], eq_params['vs30'])
        
        # Calculate residuals (natural log)
        residuals = np.log(sim_pga / bssa_pga)
        
        # Remove invalid residuals
        valid_mask = (sim_pga > 0) & (bssa_pga > 0) & np.isfinite(residuals)
        residuals = residuals[valid_mask]
        distances_valid = self.rjb_distances[valid_mask]
        
        # Create residual plots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Residuals vs distance
        ax1.scatter(distances_valid, residuals, alpha=0.6, s=1)
        ax1.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax1.set_xlabel('Distance (km)', fontsize=12)
        ax1.set_ylabel('Residual (ln(Obs/Pred))', fontsize=12)
        ax1.set_title('PGA Residuals vs Distance', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        # Residual histogram
        ax2.hist(residuals, bins=50, alpha=0.7, density=True, edgecolor='black')
        ax2.axvline(x=0, color='r', linestyle='--', linewidth=2)
        ax2.axvline(x=np.mean(residuals), color='g', linestyle='-', linewidth=2, 
                   label=f'Mean = {np.mean(residuals):.3f}')
        ax2.set_xlabel('Residual (ln(Obs/Pred))', fontsize=12)
        ax2.set_ylabel('Density', fontsize=12)
        ax2.set_title('PGA Residual Distribution', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'pga_residuals.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Calculate statistics
        residual_stats = {
            'mean': np.mean(residuals),
            'std': np.std(residuals),
            'median': np.median(residuals),
            'percentile_16': np.percentile(residuals, 16),
            'percentile_84': np.percentile(residuals, 84),
            'count': len(residuals)
        }
        
        self.logger.info(f"Residual statistics: Mean={residual_stats['mean']:.3f}, "
                        f"Std={residual_stats['std']:.3f}")
        
        return residual_stats
    
    def generate_statistics_report(self) -> None:
        """Generate ground motion statistics and save results"""
        self.logger.info("Generating ground motion statistics report")
        
        # Compute statistics for all metrics
        all_statistics = self.compute_all_gm_statistics()
        
        # Save statistics to NPZ file
        self.save_statistics_npz(all_statistics)
        
        # Write summary report
        report_file = self.output_dir / 'gm_statistics_summary.txt'
        with open(report_file, 'w') as f:
            f.write("Ground Motion Statistics Summary\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("Dataset Information:\n")
            f.write(f"  Number of stations: {len(self.station_ids)}\n")
            f.write(f"  Distance range: {self.rjb_distances.min():.1f} - {self.rjb_distances.max():.1f} m\n")
            f.write(f"  Distance bin range: {self.distance_range[0]:.0f} - {self.distance_range[1]:.0f} m\n")
            f.write(f"  Distance bin size: {self.distance_bin_size:.0f} m\n")
            f.write(f"  Number of distance bins: {int((self.distance_range[1] - self.distance_range[0]) / self.distance_bin_size)}\n")
            f.write("\n")
            
            f.write("Available GM metrics:\n")
            for metric in all_statistics.keys():
                stats = all_statistics[metric]
                valid_bins = np.sum(stats['count'] > 0)
                f.write(f"  {metric}: {valid_bins} bins with data\n")
            f.write("\n")
            
            f.write("Generated Files:\n")
            f.write("  - gm_statistics.npz (main statistics file)\n")
            f.write("  - gm_statistics_summary.txt (this file)\n")
        
        self.logger.info(f"Statistics report saved to: {report_file}")
        self.logger.info("All statistics processing completed successfully!")

def main():
    """Main entry point for ground motion statistics computation"""
    parser = argparse.ArgumentParser(description='Compute ground motion statistics as a function of Rjb distance')
    parser.add_argument('--gm_data', required=True, 
                       help='Path to ground motion metrics NPZ file')
    parser.add_argument('--output_dir', required=True, 
                       help='Output directory for statistics results')
    parser.add_argument('--distance_range', nargs=2, type=float, default=[0, 30000],
                       help='Distance range in meters (default: 0 30000)')
    parser.add_argument('--distance_bin_size', type=float, default=1000,
                       help='Distance bin size in meters (default: 1000)')
    parser.add_argument('--verbose', action='store_true', 
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate distance range
    if args.distance_range[0] < 0:
        print("Error: Minimum distance must be >= 0")
        sys.exit(1)
    if args.distance_range[1] <= args.distance_range[0]:
        print("Error: Maximum distance must be greater than minimum distance")
        sys.exit(1)
    if args.distance_bin_size <= 0:
        print("Error: Distance bin size must be > 0")
        sys.exit(1)
    
    try:
        # Create GM statistics instance
        gm_stats = GMStatistics(
            args.gm_data, 
            args.output_dir,
            distance_range=tuple(args.distance_range),
            distance_bin_size=args.distance_bin_size
        )
        
        # Generate statistics report
        gm_stats.generate_statistics_report()
        
        print(f"\nGround motion statistics computation completed successfully!")
        print(f"Results saved in: {args.output_dir}")
        print(f"Main statistics file: {args.output_dir}/gm_statistics.npz")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()