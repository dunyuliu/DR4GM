#!/usr/bin/env python3
"""
SPECFEM3D Ground Motion Converter API

Converts SPECFEM3D processed ground motion data (CSV format) directly to DR4GM 
gm_statistics.npz format, bypassing time series processing since GM metrics 
are already computed.

The SPECFEM3D CSV files contain:
- Station coordinates (x, y, z)
- Distance metrics (R_rup - closest distance to fault)
- Ground motion metrics (GMROT for T=0.5s, 1s, 3s; CAV)
- Moment magnitude (Mw)

Usage:
    python specfem3d_converter_api.py --input_file <csv_file> --output_dir <out_dir>

The converter writes gm_statistics.npz directly by binning the station data
by distance and computing statistics for each distance bin.
"""

from __future__ import annotations

import argparse
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple

class SPECFEM3DConverter:
    """Convert SPECFEM3D CSV ground motion data to DR4GM gm_statistics.npz format."""

    def __init__(
        self,
        input_file: Path,
        output_dir: Path,
        distance_range: Tuple[float, float] = (0, 100000),  # 0-100km in meters
        distance_bin_size: float = 1000,  # 1km bins in meters
        log_level: int = logging.INFO,
    ) -> None:
        self.input_file = Path(input_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if not self.input_file.exists():
            raise FileNotFoundError(f"Input CSV file not found: {input_file}")
        
        # Distance binning parameters
        self.distance_range = distance_range
        self.distance_bin_size = distance_bin_size
        
        # Setup logging
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger(__name__)

    def load_csv_data(self) -> pd.DataFrame:
        """Load SPECFEM3D CSV data."""
        self.logger.info(f"Loading SPECFEM3D data from {self.input_file.name}")
        
        # Read CSV with tab delimiter
        df = pd.read_csv(self.input_file, sep='\t')
        
        # Verify required columns
        required_cols = ['x', 'y', 'z', 'R_rup', 'gm50_T0p5', 'gm50_T1', 'gm50_T3', 'CAV_geometric_mean', 'Mw']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        self.logger.info(f"Loaded {len(df)} stations")
        self.logger.info(f"Distance range: {df['R_rup'].min():.1f} - {df['R_rup'].max():.1f} km")
        self.logger.info(f"Moment magnitude: {df['Mw'].iloc[0]}")
        
        return df

    def calc_gm_stats_vs_r(self, values: np.ndarray, distances: np.ndarray, 
                          r_bins: np.ndarray) -> np.ndarray:
        """
        Calculate ground motion statistics vs distance bins.
        
        Returns:
            stats array (len(r_bins)-1, 6): [geometric_mean, log_std, min, max, count, unused]
        """
        n_bins = len(r_bins) - 1
        gm_metrics_vs_r = np.zeros((n_bins, 50000))  # Pre-allocate
        log_gm_metrics = np.zeros((n_bins, 50000))
        gm_metrics_stats = np.zeros((n_bins, 6))
        num_of_gm_metrics_per_r = np.zeros(n_bins, dtype=int)

        # Bin the data
        for i_st in range(len(values)):
            for i_r in range(n_bins):
                if (distances[i_st] >= r_bins[i_r] and 
                    distances[i_st] < r_bins[i_r + 1] and 
                    abs(values[i_st]) > 0.):
                    gm_metrics_vs_r[i_r, num_of_gm_metrics_per_r[i_r]] = values[i_st]
                    num_of_gm_metrics_per_r[i_r] += 1

        # Calculate statistics
        for i_r in range(n_bins):
            if num_of_gm_metrics_per_r[i_r] > 1:
                # Calculate geometric mean and log standard deviation
                valid_data = gm_metrics_vs_r[i_r, :num_of_gm_metrics_per_r[i_r]]
                log_data = np.log(valid_data)
                
                mean_log_x = np.mean(log_data)
                std_log_x = np.std(log_data, ddof=1)
                min_x = np.min(valid_data)
                max_x = np.max(valid_data)
                
                gm_metrics_stats[i_r, 0] = np.exp(mean_log_x)  # Geometric mean
                gm_metrics_stats[i_r, 1] = std_log_x           # Log standard deviation
                gm_metrics_stats[i_r, 2] = min_x               # Minimum
                gm_metrics_stats[i_r, 3] = max_x               # Maximum
                gm_metrics_stats[i_r, 4] = num_of_gm_metrics_per_r[i_r]  # Count
            elif num_of_gm_metrics_per_r[i_r] == 1:
                # Single data point
                gm_metrics_stats[i_r, 0] = gm_metrics_vs_r[i_r, 0]
                gm_metrics_stats[i_r, 1] = 0.0
                gm_metrics_stats[i_r, 2] = gm_metrics_vs_r[i_r, 0]
                gm_metrics_stats[i_r, 3] = gm_metrics_vs_r[i_r, 0]
                gm_metrics_stats[i_r, 4] = 1
            else:
                # No data - use placeholder values
                gm_metrics_stats[i_r, 0] = 1.0  # Placeholder (will be filtered out)
                gm_metrics_stats[i_r, 1] = 0.0
                gm_metrics_stats[i_r, 2] = 0.0
                gm_metrics_stats[i_r, 3] = 0.0
                gm_metrics_stats[i_r, 4] = 0

        return gm_metrics_stats

    def compute_statistics(self, df: pd.DataFrame) -> Dict[str, Dict[str, np.ndarray]]:
        """Compute statistics for all GM metrics as a function of distance."""
        self.logger.info("Computing ground motion statistics vs distance")
        
        # Convert distances from km to meters for consistency with other scenarios
        distances_m = df['R_rup'].values * 1000.0
        
        # Define distance bins in meters
        r_bin_range = self.distance_range
        r_bin_size = self.distance_bin_size
        n_bins = int((r_bin_range[1] - r_bin_range[0]) / r_bin_size)
        r_bins = np.linspace(r_bin_range[0], r_bin_range[1], n_bins + 1)
        r_bin_centers = r_bins[:-1] + r_bin_size / 2.0
        
        # Map SPECFEM3D metrics to DR4GM format
        # Note: SPECFEM3D GMROT values appear to be in g units, convert to cm/s²
        metric_mapping = {
            'PGA': ('gm50_T0p5', 981.0),          # Use T=0.5s as PGA approximation
            'RSA_T_0_500': ('gm50_T0p5', 981.0),  # Convert g to cm/s² (T=0.5s)
            'RSA_T_1_000': ('gm50_T1', 981.0),    # Convert g to cm/s² (T=1.0s)
            'RSA_T_3_000': ('gm50_T3', 981.0),    # Convert g to cm/s² (T=3.0s)
            'CAV': ('CAV_geometric_mean', 100.0), # Convert from m/s to cm/s
        }
        
        all_statistics = {}
        
        for metric, (csv_col, conversion_factor) in metric_mapping.items():
            self.logger.info(f"Processing {metric} from {csv_col}")
            
            # Get the data and apply unit conversion
            metric_data = df[csv_col].values * conversion_factor
            
            # Calculate statistics
            stats = self.calc_gm_stats_vs_r(metric_data, distances_m, r_bins)
            
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

    def save_statistics_npz(self, statistics: Dict[str, Dict[str, np.ndarray]], 
                           df: pd.DataFrame, filename: str = 'gm_statistics.npz') -> None:
        """Save statistics to NPZ file compatible with DR4GM format."""
        save_dict = {}
        
        # Add distance bins (same for all metrics)
        first_metric = next(iter(statistics.values()))
        save_dict['rjb_distance_bins'] = first_metric['rjb_distance']
        save_dict['distance_bin_edges'] = first_metric['bins']
        
        # Add units information
        save_dict['distance_units'] = 'm'  # distances in meters
        save_dict['RSA_units'] = 'cm/s²'
        save_dict['CAV_units'] = 'cm·s'
        
        # Add metadata
        save_dict['moment_magnitude'] = df['Mw'].iloc[0]
        save_dict['total_stations'] = len(df)
        save_dict['source_file'] = self.input_file.name
        
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

    def load_fault_geometry_json(self) -> Dict:
        """Load fault geometry from JSON file in input directory"""
        # For SPECFEM3D, we need to look for the JSON file in the parent directory of the CSV file
        input_dir = self.input_file.parent
        fault_json_file = input_dir / "fault_geometry.json"
        
        if not fault_json_file.exists():
            self.logger.error(f"ERROR: fault_geometry.json not found in {input_dir}")
            self.logger.error("Each dataset directory must contain a fault_geometry.json file")
            raise FileNotFoundError(f"Required file not found: {fault_json_file}")
        
        try:
            with open(fault_json_file, 'r') as f:
                fault_geometry = json.load(f)
            
            self.logger.info(f"Loaded fault geometry from: {fault_json_file}")
            return fault_geometry
            
        except Exception as e:
            self.logger.error(f"Error reading fault_geometry.json: {e}")
            raise

    def create_geometry_npz(self, df: pd.DataFrame) -> str:
        """Create fault geometry NPZ file using fault_geometry.json"""
        self.logger.info("Creating fault geometry NPZ file")
        
        # Load fault geometry from JSON file
        fault_json = self.load_fault_geometry_json()
        
        # Extract data from JSON
        fault_start = np.array(fault_json['fault_trace_start'], dtype=np.float32)
        fault_end = np.array(fault_json['fault_trace_end'], dtype=np.float32)
        
        # Apply rotation if specified (SPECFEM3D typically doesn't need rotation)
        fault_start_rotated = fault_start.copy()
        fault_end_rotated = fault_end.copy()
        
        if fault_json.get('rotation', 'no').lower() == 'yes':
            self.logger.info("Applying 90° counterclockwise rotation to fault geometry")
            # Apply 90° counterclockwise rotation: (x,y) -> (-y,x)
            fault_start_rotated = np.array([-fault_start[1], fault_start[0], fault_start[2]], dtype=np.float32)
            fault_end_rotated = np.array([-fault_end[1], fault_end[0], fault_end[2]], dtype=np.float32)
        
        geometry_payload = {
            'fault_type': fault_json['fault_type'],
            'fault_trace_start': fault_start_rotated,
            'fault_trace_end': fault_end_rotated,
            'fault_dip': np.float32(fault_json['fault_dip']),
            'fault_strike': np.float32(fault_json['fault_strike']),
            'fault_length': np.float32(fault_json['fault_length']),
            'fault_width': np.float32(fault_json['fault_width']),
            'top_depth': np.float32(fault_json['top_depth']),
            'bottom_depth': np.float32(fault_json['bottom_depth']),
            'coordinate_units': fault_json['coordinate_units'],
            'description': fault_json['description'],
            'rotation_applied': 'yes' if fault_json.get('rotation', 'no').lower() == 'yes' else 'no',
            'moment_magnitude': np.float32(df['Mw'].iloc[0]),
        }
        
        # Add moment magnitude from JSON if present (prioritize over CSV)
        if 'moment_magnitude' in fault_json:
            geometry_payload['moment_magnitude'] = np.float32(fault_json['moment_magnitude'])
        
        output_path = self.output_dir / 'geometry.npz'
        np.savez_compressed(output_path, **geometry_payload)
        self.logger.info(f"Geometry saved to: {output_path}")
        self.logger.info(f"Fault trace: {fault_start_rotated} to {fault_end_rotated}")
        return str(output_path)

    def convert(self) -> Dict[str, str]:
        """Execute the conversion pipeline."""
        # Load CSV data
        df = self.load_csv_data()
        
        # Compute statistics
        statistics = self.compute_statistics(df)
        
        # Save statistics
        self.save_statistics_npz(statistics, df)
        
        # Create geometry file
        geometry_file = self.create_geometry_npz(df)
        
        # Write summary report
        report_file = self.output_dir / 'conversion_summary.txt'
        with open(report_file, 'w') as f:
            f.write("SPECFEM3D Conversion Summary\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Source file: {self.input_file.name}\n")
            f.write(f"Total stations: {len(df)}\n")
            f.write(f"Moment magnitude: {df['Mw'].iloc[0]}\n")
            f.write(f"Distance range: {df['R_rup'].min():.1f} - {df['R_rup'].max():.1f} km\n")
            f.write(f"Distance bin size: {self.distance_bin_size/1000:.1f} km\n")
            f.write("\nGenerated files:\n")
            f.write("  - gm_statistics.npz (main statistics file)\n")
            f.write("  - geometry.npz (fault geometry)\n")
            f.write("  - conversion_summary.txt (this file)\n")
        
        self.logger.info(f"Conversion completed successfully!")
        self.logger.info(f"Results saved in: {self.output_dir}")
        
        return {
            "statistics": str(self.output_dir / 'gm_statistics.npz'),
            "geometry": geometry_file,
            "summary": str(report_file)
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert SPECFEM3D CSV ground motion data to DR4GM NPZ format",
    )
    parser.add_argument(
        "--input_file",
        required=True,
        help="Path to SPECFEM3D CSV file (e.g., Model1_GMROT_with_CAV.csv)",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory where NPZ files will be written",
    )
    parser.add_argument(
        "--distance_range",
        nargs=2,
        type=float,
        default=[0, 100000],
        help="Distance range in meters (default: 0 100000)",
    )
    parser.add_argument(
        "--distance_bin_size",
        type=float,
        default=1000,
        help="Distance bin size in meters (default: 1000)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_level = logging.DEBUG if args.verbose else logging.INFO
    
    converter = SPECFEM3DConverter(
        input_file=Path(args.input_file),
        output_dir=Path(args.output_dir),
        distance_range=tuple(args.distance_range),
        distance_bin_size=args.distance_bin_size,
        log_level=log_level,
    )
    converter.convert()


if __name__ == "__main__":
    main()