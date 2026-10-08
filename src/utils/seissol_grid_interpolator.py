#!/usr/bin/env python3

"""
SeisSol Grid Interpolator

Post-processing utility to interpolate irregular SeisSol station data to regular grids.
This is a separate utility that operates on already-converted SeisSol NPZ files.

Usage:
    python seissol_grid_interpolator.py --input_npz <seissol_velocities.npz> --output_npz <interpolated.npz> --grid_resolution 1000
"""

import os
import sys
import argparse
import numpy as np
import logging
from typing import Dict, Tuple
import time
from pathlib import Path

# Import DR4GM standards
from npz_format_standard import DR4GM_NPZ_Standard

class SeisSolGridInterpolator:
    """Interpolate irregular SeisSol data to regular grids"""
    
    def __init__(self, grid_resolution: float):
        """
        Initialize interpolator
        
        Args:
            grid_resolution: Grid resolution in meters for interpolation
        """
        self.grid_resolution = grid_resolution
        
        # Logging setup
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
    
    def interpolate_velocities_fast(self, locations: np.ndarray, velocities: np.ndarray, grid_points: np.ndarray) -> np.ndarray:
        """
        Fast interpolation using nearest neighbor for velocity time series
        
        Args:
            locations: Original irregular station locations [N, 2]
            velocities: Velocity time series [N, T]
            grid_points: Target grid points [M, 2]
            
        Returns:
            Interpolated velocities [M, T]
        """
        try:
            from scipy.spatial import cKDTree
        except ImportError:
            raise ImportError("scipy is required for interpolation")
        
        # Build spatial index for fast nearest neighbor lookup
        tree = cKDTree(locations)
        
        # Find nearest neighbors for each grid point
        distances, indices = tree.query(grid_points, k=1)
        
        # Simply copy values from nearest neighbors
        interpolated_velocities = velocities[indices]
        
        return interpolated_velocities
    
    def interpolate_to_regular_grid(self, input_npz: str, output_npz: str) -> Dict:
        """
        Interpolate irregular SeisSol NPZ data to regular grid
        
        Args:
            input_npz: Path to input SeisSol velocities NPZ file
            output_npz: Path to output interpolated NPZ file
            
        Returns:
            Dictionary with interpolation results
        """
        start_time = time.time()
        self.logger.info(f"Starting grid interpolation from {input_npz}")
        self.logger.info(f"Target grid resolution: {self.grid_resolution}m")
        
        # Load original data
        data = np.load(input_npz)
        
        # Extract required fields
        orig_locations = data['locations']
        orig_vel_strike = data['vel_strike']
        orig_vel_normal = data['vel_normal']
        orig_vel_vertical = data['vel_vertical']
        
        self.logger.info(f"Loaded {len(orig_locations)} irregular stations")
        self.logger.info(f"Time series shape: {orig_vel_strike.shape}")
        
        # Define regular grid bounds
        x_min, x_max = orig_locations[:, 0].min(), orig_locations[:, 0].max()
        y_min, y_max = orig_locations[:, 1].min(), orig_locations[:, 1].max()
        
        # Create regular grid
        x_grid = np.arange(x_min, x_max + self.grid_resolution, self.grid_resolution)
        y_grid = np.arange(y_min, y_max + self.grid_resolution, self.grid_resolution)
        
        # Create meshgrid
        X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
        grid_points_2d = np.column_stack([X_grid.ravel(), Y_grid.ravel()])
        
        # Add Z=0 for all grid points (surface stations)
        grid_locations = np.column_stack([grid_points_2d, np.zeros(len(grid_points_2d))])
        
        n_grid_stations = len(grid_locations)
        
        self.logger.info(f"Target regular grid: {len(x_grid)} x {len(y_grid)} = {n_grid_stations} stations")
        
        # Fast interpolation using nearest neighbors only (much faster than linear)
        self.logger.info("Performing fast nearest-neighbor interpolation...")
        
        orig_points_2d = orig_locations[:, :2]
        
        # Interpolate each velocity component
        grid_vel_strike = self.interpolate_velocities_fast(orig_points_2d, orig_vel_strike, grid_points_2d)
        grid_vel_normal = self.interpolate_velocities_fast(orig_points_2d, orig_vel_normal, grid_points_2d)
        grid_vel_vertical = self.interpolate_velocities_fast(orig_points_2d, orig_vel_vertical, grid_points_2d)
        
        # Generate new station IDs for grid points
        grid_station_ids = np.arange(n_grid_stations)
        
        # Create interpolated dataset
        interpolated_data = {
            'station_ids': grid_station_ids,
            'locations': grid_locations,
            'vel_strike': grid_vel_strike,
            'vel_normal': grid_vel_normal,
            'vel_vertical': grid_vel_vertical,
            'time_steps': np.full(n_grid_stations, data['time_steps'][0]),
            'dt_values': np.full(n_grid_stations, data['dt_values'][0]),
            'duration': data['duration'],
            'units': data.get('units', 'm/s'),
            'coordinate_units': data.get('coordinate_units', 'm'),
            'time_array': data.get('time_array', None)
        }
        
        # Save interpolated data
        self.logger.info(f"Saving interpolated data to {output_npz}")
        np.savez_compressed(output_npz, **interpolated_data)
        
        # Validate against DR4GM standard
        is_valid = DR4GM_NPZ_Standard.validate_npz(output_npz, 'layer3_velocities')
        
        duration = time.time() - start_time
        
        self.logger.info(f"Grid interpolation completed in {duration:.2f}s")
        self.logger.info(f"Original irregular stations: {len(orig_locations)}")
        self.logger.info(f"Interpolated regular grid: {n_grid_stations}")
        self.logger.info(f"Output file valid: {is_valid}")
        
        return {
            'input_file': input_npz,
            'output_file': output_npz,
            'original_stations': len(orig_locations),
            'grid_stations': n_grid_stations,
            'grid_resolution': self.grid_resolution,
            'interpolation_time': duration,
            'is_valid': is_valid
        }

def main():
    """Main entry point for the grid interpolator"""
    parser = argparse.ArgumentParser(description='Interpolate irregular SeisSol data to regular grid')
    parser.add_argument('--input_npz', required=True, help='Input SeisSol velocities NPZ file')
    parser.add_argument('--output_npz', required=True, help='Output interpolated NPZ file')
    parser.add_argument('--grid_resolution', type=float, required=True, help='Grid resolution in meters')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if not os.path.exists(args.input_npz):
        print(f"Error: Input file {args.input_npz} does not exist")
        sys.exit(1)
    
    try:
        interpolator = SeisSolGridInterpolator(args.grid_resolution)
        results = interpolator.interpolate_to_regular_grid(args.input_npz, args.output_npz)
        
        print("\n=== Grid Interpolation Results ===")
        print(f"Input file: {results['input_file']}")
        print(f"Output file: {results['output_file']}")
        print(f"Original stations: {results['original_stations']:,}")
        print(f"Grid stations: {results['grid_stations']:,}")
        print(f"Grid resolution: {results['grid_resolution']}m")
        print(f"Interpolation time: {results['interpolation_time']:.2f}s")
        print(f"Output file valid: {results['is_valid']}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()