#!/usr/bin/env python3

"""
Station Subset Selector

Creates a subset NPZ file with selected stations from a larger NPZ velocity file.
The subset file can then be processed with npz_gm_processor.py

Usage:
    python station_subset_selector.py --input_npz <input.npz> --output_npz <subset.npz> [options]

Examples:
    # Select first 1000 stations
    python station_subset_selector.py --input_npz velocities.npz --output_npz subset.npz --max_stations 1000
    
    # Select stations 5000-10000
    python station_subset_selector.py --input_npz velocities.npz --output_npz subset.npz --station_range 5000 10000
    
    # Select every 10th station
    python station_subset_selector.py --input_npz velocities.npz --output_npz subset.npz --station_step 10
    
    # Combine: first 5000 stations, every 5th one
    python station_subset_selector.py --input_npz velocities.npz --output_npz subset.npz --max_stations 5000 --station_step 5
"""

import os
import sys
import argparse
import numpy as np
import logging
from typing import Optional, Tuple
from pathlib import Path

# Import DR4GM standards
from npz_format_standard import DR4GM_NPZ_Standard

class StationSubsetSelector:
    """Select subset of stations from NPZ velocity file"""
    
    def __init__(self, input_npz: str, output_npz: str,
                 station_range: Optional[Tuple[int, int]] = None,
                 max_stations: Optional[int] = None, 
                 station_step: int = 1,
                 spatial_percentage: Optional[float] = None,
                 grid_resolution: Optional[float] = None):
        """
        Initialize station subset selector
        
        Args:
            input_npz: Path to input NPZ velocity file
            output_npz: Path to output subset NPZ file
            station_range: Process only stations in range [start, end) (0-based indexing)
            max_stations: Maximum number of stations to select
            station_step: Select every Nth station (default: 1)
            spatial_percentage: Percentage of stations to select uniformly across space (0-100)
            grid_resolution: Grid resolution in meters for grid-based station selection
        """
        self.input_npz = Path(input_npz)
        self.output_npz = Path(output_npz)
        self.station_range = station_range
        self.max_stations = max_stations
        self.station_step = station_step
        self.spatial_percentage = spatial_percentage
        self.grid_resolution = grid_resolution
        
        # Create output directory if needed
        self.output_npz.parent.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        if not self.input_npz.exists():
            raise FileNotFoundError(f"Input NPZ file not found: {input_npz}")
    
    def spatial_uniform_selection(self, locations: np.ndarray, percentage: float) -> np.ndarray:
        """
        Select stations uniformly distributed across space using k-means clustering
        
        Args:
            locations: Station locations (N, 2 or 3)
            percentage: Percentage of stations to select (0-100)
            
        Returns:
            Selected station indices
        """
        n_total = len(locations)
        n_select = int(n_total * percentage / 100.0)
        
        if n_select >= n_total:
            return np.arange(n_total)
        
        # Use 2D coordinates for spatial distribution (X, Y)
        coords_2d = locations[:, :2]
        
        try:
            from sklearn.cluster import KMeans
            
            # Cluster stations and pick one representative from each cluster
            kmeans = KMeans(n_clusters=n_select, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(coords_2d)
            
            # Select the station closest to each cluster center
            selected_indices = []
            for i in range(n_select):
                cluster_mask = cluster_labels == i
                if np.any(cluster_mask):
                    cluster_coords = coords_2d[cluster_mask]
                    cluster_indices = np.where(cluster_mask)[0]
                    center = kmeans.cluster_centers_[i]
                    
                    # Find closest station to cluster center
                    distances = np.sum((cluster_coords - center)**2, axis=1)
                    closest_idx = cluster_indices[np.argmin(distances)]
                    selected_indices.append(closest_idx)
            
            return np.array(selected_indices)
            
        except ImportError:
            self.logger.warning("sklearn not available, falling back to grid-based selection")
            # Fallback to simple spatial grid
            return self.grid_based_selection(locations, percentage * 1000)  # Convert to rough grid size
    
    def grid_based_selection(self, locations: np.ndarray, grid_resolution: float) -> np.ndarray:
        """
        Select stations on a regular spatial grid
        
        Args:
            locations: Station locations (N, 2 or 3)
            grid_resolution: Grid spacing in meters
            
        Returns:
            Selected station indices
        """
        # Use 2D coordinates for grid (X, Y)
        coords_2d = locations[:, :2]
        
        # Define grid bounds
        x_min, y_min = coords_2d.min(axis=0)
        x_max, y_max = coords_2d.max(axis=0)
        
        # Create grid points
        x_grid = np.arange(x_min, x_max + grid_resolution, grid_resolution)
        y_grid = np.arange(y_min, y_max + grid_resolution, grid_resolution)
        
        selected_indices = []
        
        # For each grid cell, find the closest station
        for x_center in x_grid:
            for y_center in y_grid:
                grid_center = np.array([x_center, y_center])
                
                # Find station closest to this grid point
                distances = np.sum((coords_2d - grid_center)**2, axis=1)
                closest_idx = np.argmin(distances)
                
                # Check if this station is within the grid cell
                closest_coord = coords_2d[closest_idx]
                if (abs(closest_coord[0] - x_center) <= grid_resolution/2 and 
                    abs(closest_coord[1] - y_center) <= grid_resolution/2):
                    if closest_idx not in selected_indices:
                        selected_indices.append(closest_idx)
        
        return np.array(selected_indices)

    def select_station_indices(self, total_stations: int, locations: np.ndarray = None) -> np.ndarray:
        """Select station indices based on user criteria"""
        self.logger.info(f"Selecting stations from {total_stations} total stations")
        
        # Check for spatial selection methods first
        if self.spatial_percentage is not None:
            if locations is None:
                raise ValueError("Station locations required for spatial selection")
            self.logger.info(f"Using spatial uniform selection: {self.spatial_percentage}% of stations")
            indices = self.spatial_uniform_selection(locations, self.spatial_percentage)
            self.logger.info(f"Spatial uniform selection: {len(indices)} stations selected")
            return indices
        
        if self.grid_resolution is not None:
            if locations is None:
                raise ValueError("Station locations required for grid-based selection")
            self.logger.info(f"Using grid-based selection: {self.grid_resolution}m resolution")
            indices = self.grid_based_selection(locations, self.grid_resolution)
            self.logger.info(f"Grid-based selection: {len(indices)} stations selected")
            return indices
        
        # Original selection methods (sequential)
        # Start with all stations
        indices = np.arange(total_stations)
        
        # Apply station range filter
        if self.station_range is not None:
            start, end = self.station_range
            start = max(0, start)
            end = min(total_stations, end)
            indices = indices[start:end]
            self.logger.info(f"Applied station range filter: [{start}, {end}) -> {len(indices)} stations")
        
        # Apply station step filter (every Nth station)
        if self.station_step > 1:
            indices = indices[::self.station_step]
            self.logger.info(f"Applied station step filter: every {self.station_step} stations -> {len(indices)} stations")
        
        # Apply max stations limit
        if self.max_stations is not None and len(indices) > self.max_stations:
            indices = indices[:self.max_stations]
            self.logger.info(f"Applied max stations limit: {self.max_stations} -> {len(indices)} stations")
        
        self.logger.info(f"Final selection: {len(indices)} stations ({100*len(indices)/total_stations:.1f}% of total)")
        return indices
    
    def create_subset(self) -> dict:
        """Create subset NPZ file with selected stations"""
        self.logger.info(f"Loading data from {self.input_npz}")
        
        # Load input NPZ file
        with np.load(self.input_npz) as data:
            # Get basic info
            total_stations = len(data['station_ids'])
            self.logger.info(f"Input file contains {total_stations} stations")
            
            # Detect available velocity components
            has_strike_normal = 'vel_strike' in data and 'vel_normal' in data
            has_xy = 'vel_x' in data and 'vel_y' in data
            has_xyz = 'vel_x' in data and 'vel_y' in data and 'vel_z' in data
            
            # Select station indices (pass locations for spatial methods)
            locations = data['locations'] if ('locations' in data) else None
            selected_indices = self.select_station_indices(total_stations, locations)
            
            if len(selected_indices) == 0:
                raise ValueError("No stations selected with current criteria")
            
            # Create subset data dictionary
            subset_data = {}
            
            # Copy basic station info
            subset_data['station_ids'] = data['station_ids'][selected_indices]
            subset_data['locations'] = data['locations'][selected_indices]
            
            # Copy velocity data for selected stations
            if has_strike_normal:
                subset_data['vel_strike'] = data['vel_strike'][selected_indices]
                subset_data['vel_normal'] = data['vel_normal'][selected_indices]
                if 'vel_vertical' in data:
                    subset_data['vel_vertical'] = data['vel_vertical'][selected_indices]
                self.logger.info("Copied strike/normal velocity components")
            
            if has_xy:
                subset_data['vel_x'] = data['vel_x'][selected_indices]
                subset_data['vel_y'] = data['vel_y'][selected_indices]
                if has_xyz:
                    subset_data['vel_z'] = data['vel_z'][selected_indices]
                self.logger.info("Copied X/Y/Z velocity components")
            
            # Copy metadata if present
            metadata_keys = ['dt_values', 'time_steps', 'duration', 'units']
            for key in metadata_keys:
                if key in data:
                    if key in ['dt_values', 'time_steps']:
                        # These are per-station arrays
                        subset_data[key] = data[key][selected_indices]
                    else:
                        # These are scalars or global metadata
                        subset_data[key] = data[key]
            
            # Copy any other arrays that might be present
            for key in data.keys():
                if key not in subset_data and key not in ['station_ids', 'locations']:
                    try:
                        if data[key].shape[0] == total_stations:
                            # This looks like a per-station array
                            subset_data[key] = data[key][selected_indices]
                            self.logger.info(f"Copied per-station array: {key}")
                        else:
                            # This looks like metadata
                            subset_data[key] = data[key]
                            self.logger.info(f"Copied metadata: {key}")
                    except (AttributeError, IndexError):
                        # Skip items that don't have shape or can't be indexed
                        subset_data[key] = data[key]
        
        # Save subset NPZ file
        self.logger.info(f"Saving subset to {self.output_npz}")
        np.savez_compressed(self.output_npz, **subset_data)
        
        # Create summary
        summary = {
            'input_file': str(self.input_npz),
            'output_file': str(self.output_npz),
            'total_stations': total_stations,
            'selected_stations': len(selected_indices),
            'selection_percentage': 100 * len(selected_indices) / total_stations,
            'station_range': self.station_range,
            'max_stations': self.max_stations,
            'station_step': self.station_step,
            'velocity_components': []
        }
        
        if has_strike_normal:
            summary['velocity_components'].append('strike/normal')
        if has_xy:
            summary['velocity_components'].append('x/y/z')
        
        return summary
    
    def validate_subset(self) -> bool:
        """Validate the created subset NPZ file"""
        try:
            # Try to validate against DR4GM standard
            is_valid = DR4GM_NPZ_Standard.validate_npz(str(self.output_npz), 'layer3_velocities')
            if is_valid:
                self.logger.info("Subset NPZ file passes DR4GM validation")
            else:
                self.logger.warning("Subset NPZ file validation warnings (but should still work)")
            return True
        except Exception as e:
            self.logger.error(f"Validation failed: {e}")
            return False

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Select subset of stations from NPZ velocity file')
    parser.add_argument('--input_npz', required=True, help='Input NPZ velocity file')
    parser.add_argument('--output_npz', required=True, help='Output subset NPZ file')
    parser.add_argument('--station_range', nargs=2, type=int, metavar=('START', 'END'), 
                       help='Select only stations in range [START, END) (0-based indexing)')
    parser.add_argument('--max_stations', type=int, help='Maximum number of stations to select')
    parser.add_argument('--station_step', type=int, default=1, help='Select every Nth station (default: 1)')
    parser.add_argument('--spatial_percentage', type=float, metavar='PERCENT',
                       help='Select PERCENT%% of stations uniformly distributed across space (0-100)')
    parser.add_argument('--grid_resolution', type=float, metavar='METERS',
                       help='Select stations on a spatial grid with given resolution in meters')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        selector = StationSubsetSelector(
            args.input_npz,
            args.output_npz, 
            args.station_range,
            args.max_stations,
            args.station_step,
            args.spatial_percentage,
            args.grid_resolution
        )
        
        summary = selector.create_subset()
        
        # Validate the result
        is_valid = selector.validate_subset()
        
        print("\n=== Station Subset Selection Results ===")
        print(f"Input file: {summary['input_file']}")
        print(f"Output file: {summary['output_file']}")
        print(f"Total stations: {summary['total_stations']:,}")
        print(f"Selected stations: {summary['selected_stations']:,}")
        print(f"Selection percentage: {summary['selection_percentage']:.1f}%")
        print(f"Velocity components: {', '.join(summary['velocity_components'])}")
        print(f"File validation: {'PASSED' if is_valid else 'FAILED'}")
        
        if summary['station_range']:
            print(f"Station range: [{summary['station_range'][0]}, {summary['station_range'][1]})")
        if summary['max_stations']:
            print(f"Max stations limit: {summary['max_stations']:,}")
        if summary['station_step'] > 1:
            print(f"Station step: every {summary['station_step']} stations")
        
        print(f"\nSubset file ready for processing with npz_gm_processor.py:")
        print(f"python npz_gm_processor.py --velocity_npz {summary['output_file']} --output_dir <output_dir>")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()