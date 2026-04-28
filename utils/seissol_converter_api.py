#!/usr/bin/env python3

"""
SeisSol Dataset Converter API

Converts SeisSol scenario datasets to parsable station lists and velocity time history databases
in NPZ format following the DR4GM standard.

SeisSol provides HDF5 files containing velocity time series on many surface stations.
Each .h5 file contains:
- time: time array
- v1, v2, v3: velocity components (fault-parallel, fault-perpendicular, vertical)
- xyz: station coordinates

Usage:
    python seissol_converter_api.py --input_dir <seissol_dir> --output_dir <output_dir>
    
Example:
    python seissol_converter_api.py --input_dir ../../datasets/seissol --output_dir ./converted_data
"""

import os
import sys
import argparse
import numpy as np
import h5py
import logging
from typing import Dict, List, Tuple, Optional
import time
import json
from pathlib import Path

# Import DR4GM standards
from npz_format_standard import DR4GM_NPZ_Standard

class SeisSolConverter:
    """Convert SeisSol datasets to DR4GM NPZ format"""
    
    def __init__(self, input_dir: str, output_dir: str):
        """
        Initialize converter
        
        Args:
            input_dir: Directory containing seissol .h5 files
            output_dir: Directory to save converted NPZ files
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Station and time series data
        self.global_station_id = 0
        
        # Logging setup
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
    
    def discover_h5_files(self) -> List[Path]:
        """Discover all .h5 files in the input directory"""
        h5_files = list(self.input_dir.glob("*.h5"))
        h5_files.sort()
        
        self.logger.info(f"Discovered {len(h5_files)} .h5 files: {[f.name for f in h5_files]}")
        return h5_files
    
    def load_seissol_data(self, h5_file: Path) -> Dict:
        """
        Load data from a single SeisSol .h5 file
        
        Returns:
            Dictionary containing time series data and metadata
        """
        self.logger.info(f"Loading SeisSol data from {h5_file.name}")
        
        try:
            with h5py.File(h5_file, 'r') as f:
                # Load datasets
                time_array = f['time'][:]
                locations = f['xyz'][:]  # [n_stations, 3] - already in correct format
                vel_v1 = f['v1'][:]  # [n_stations, n_time_steps] - fault-parallel (strike)
                vel_v2 = f['v2'][:]  # [n_stations, n_time_steps] - fault-perpendicular (normal)  
                vel_v3 = f['v3'][:]  # [n_stations, n_time_steps] - vertical
                
                # Validate data shapes
                n_stations, n_time_steps = vel_v1.shape
                assert vel_v2.shape == (n_stations, n_time_steps), "v2 shape mismatch"
                assert vel_v3.shape == (n_stations, n_time_steps), "v3 shape mismatch"
                assert locations.shape == (n_stations, 3), "xyz shape mismatch"
                assert len(time_array) == n_time_steps, "time array length mismatch"
                
                # Calculate time step
                if n_time_steps > 1:
                    dt = time_array[1] - time_array[0]
                else:
                    dt = 0.01  # Default fallback
                
                self.logger.info(f"Loaded {n_stations} stations with {n_time_steps} time steps")
                self.logger.info(f"Time range: {time_array[0]:.3f} to {time_array[-1]:.3f} s (dt={dt:.6f})")
                self.logger.info(f"Original coordinate range: X=[{locations[:,0].min():.1f}, {locations[:,0].max():.1f}], "
                               f"Y=[{locations[:,1].min():.1f}, {locations[:,1].max():.1f}], "
                               f"Z=[{locations[:,2].min():.1f}, {locations[:,2].max():.1f}]")
                
                # Check if rotation should be applied based on fault_geometry.json
                fault_geometry = self.load_fault_geometry_json()
                if fault_geometry.get('rotation', 'no').lower() == 'yes':
                    # Apply counterclockwise 90-degree rotation: (x, y) -> (-y, x)
                    x_orig = locations[:, 0].copy()
                    y_orig = locations[:, 1].copy()
                    locations[:, 0] = -y_orig  # new_x = -old_y
                    locations[:, 1] = x_orig   # new_y = old_x
                    # Z coordinate remains unchanged
                    self.logger.info(f"Applied counterclockwise 90° rotation: (x,y) -> (-y,x)")
                else:
                    self.logger.info(f"No rotation applied as per fault_geometry.json")
                self.logger.info(f"Rotated coordinate range: X=[{locations[:,0].min():.1f}, {locations[:,0].max():.1f}], "
                               f"Y=[{locations[:,1].min():.1f}, {locations[:,1].max():.1f}], "
                               f"Z=[{locations[:,2].min():.1f}, {locations[:,2].max():.1f}]")
                
                return {
                    'time': time_array,
                    'locations': locations,
                    'vel_strike': vel_v1,      # v1 = fault-parallel (strike)
                    'vel_normal': vel_v2,      # v2 = fault-perpendicular (normal)
                    'vel_vertical': vel_v3,    # v3 = vertical
                    'n_stations': n_stations,
                    'n_time_steps': n_time_steps,
                    'dt': dt,
                    'file_name': h5_file.name
                }
                
        except Exception as e:
            self.logger.error(f"Error loading {h5_file}: {e}")
            raise e
    
    def convert_dataset(self, h5_file: Path) -> Dict:
        """Convert a single SeisSol .h5 file to station data"""
        self.logger.info(f"Converting {h5_file.name}")
        
        # Load SeisSol data
        seissol_data = self.load_seissol_data(h5_file)
        
        # Generate station IDs
        n_stations = seissol_data['n_stations']
        station_ids = np.arange(self.global_station_id, self.global_station_id + n_stations)
        self.global_station_id += n_stations
        
        # Prepare conversion data
        conversion_data = {
            'station_ids': station_ids,
            'locations': seissol_data['locations'],
            'vel_strike': seissol_data['vel_strike'],
            'vel_normal': seissol_data['vel_normal'],
            'vel_vertical': seissol_data['vel_vertical'],
            'time_steps': seissol_data['n_time_steps'],
            'dt': seissol_data['dt'],
            'time_array': seissol_data['time'],
            'file_name': seissol_data['file_name']
        }
        
        self.logger.info(f"Converted {n_stations} stations from {h5_file.name}")
        return conversion_data
    
    def merge_multiple_files(self, all_conversion_data: List[Dict]) -> Dict:
        """Merge data from multiple .h5 files into a single dataset"""
        if len(all_conversion_data) == 1:
            return all_conversion_data[0]
        
        self.logger.info(f"Merging {len(all_conversion_data)} datasets")
        
        # Verify all files have same time parameters
        reference_dt = all_conversion_data[0]['dt']
        reference_time_steps = all_conversion_data[0]['time_steps']
        
        for i, data in enumerate(all_conversion_data[1:], 1):
            if abs(data['dt'] - reference_dt) > 1e-6:
                self.logger.warning(f"Time step mismatch in file {i}: {data['dt']} vs {reference_dt}")
            if data['time_steps'] != reference_time_steps:
                self.logger.warning(f"Time steps mismatch in file {i}: {data['time_steps']} vs {reference_time_steps}")
        
        # Combine all data
        combined_data = {
            'station_ids': np.concatenate([data['station_ids'] for data in all_conversion_data]),
            'locations': np.vstack([data['locations'] for data in all_conversion_data]),
            'vel_strike': np.vstack([data['vel_strike'] for data in all_conversion_data]),
            'vel_normal': np.vstack([data['vel_normal'] for data in all_conversion_data]),
            'vel_vertical': np.vstack([data['vel_vertical'] for data in all_conversion_data]),
            'time_steps': reference_time_steps,
            'dt': reference_dt,
            'time_array': all_conversion_data[0]['time_array'],  # Use first file's time array
            'file_names': [data['file_name'] for data in all_conversion_data]
        }
        
        total_stations = len(combined_data['station_ids'])
        self.logger.info(f"Merged {total_stations} stations from {len(all_conversion_data)} files")
        
        return combined_data
    
    def create_station_list_npz(self, conversion_data: Dict) -> str:
        """Create station list NPZ file following DR4GM standard"""
        self.logger.info("Creating station list NPZ file")
        
        station_data = {
            'station_ids': conversion_data['station_ids'],
            'locations': conversion_data['locations'],
            'chunk_ids': np.zeros(len(conversion_data['station_ids'])),  # All from single chunk
            'local_ids': np.arange(len(conversion_data['station_ids'])),
            'station_types': np.ones(len(conversion_data['station_ids'])),  # All surface stations
            'coordinate_units': 'm'
        }
        
        output_file = self.output_dir / "stations.npz"
        np.savez_compressed(output_file, **station_data)
        
        # Validate against DR4GM standard
        is_valid = DR4GM_NPZ_Standard.validate_npz(str(output_file), 'layer2_stations')
        self.logger.info(f"Station list NPZ created: {output_file} (Valid: {is_valid})")
        
        return str(output_file)
    
    def create_velocity_database_npz(self, conversion_data: Dict) -> str:
        """Create velocity time history database NPZ file"""
        self.logger.info("Creating velocity database NPZ file")
        
        n_stations = len(conversion_data['station_ids'])
        time_steps = conversion_data['time_steps']
        dt = conversion_data['dt']
        
        velocity_data = {
            'station_ids': conversion_data['station_ids'],
            'locations': conversion_data['locations'],
            'vel_strike': conversion_data['vel_strike'],
            'vel_normal': conversion_data['vel_normal'],
            'vel_vertical': conversion_data['vel_vertical'],
            'time_steps': np.full(n_stations, time_steps),
            'dt_values': np.full(n_stations, dt),
            'duration': time_steps * dt,
            'units': 'm/s',
            'coordinate_units': 'm',
            'time_array': conversion_data['time_array']
        }
        
        output_file = self.output_dir / "velocities.npz"
        np.savez_compressed(output_file, **velocity_data)
        
        # Validate against DR4GM standard
        is_valid = DR4GM_NPZ_Standard.validate_npz(str(output_file), 'layer3_velocities')
        self.logger.info(f"Velocity database NPZ created: {output_file} (Valid: {is_valid})")
        
        return str(output_file)
    
    def load_fault_geometry_json(self) -> Dict:
        """Load fault geometry from JSON file in input directory"""
        fault_json_file = self.input_dir / "fault_geometry.json"
        
        if not fault_json_file.exists():
            self.logger.error(f"ERROR: fault_geometry.json not found in {self.input_dir}")
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
    
    def create_fault_geometry_npz(self, conversion_data: Dict) -> str:
        """Create fault geometry NPZ file from JSON configuration"""
        self.logger.info("Creating fault geometry NPZ file")
        
        # Load original geometry from JSON
        fault_json = self.load_fault_geometry_json()
        
        # Extract original coordinates
        fault_start = np.array(fault_json['fault_trace_start'])
        fault_end = np.array(fault_json['fault_trace_end'])
        
        # Apply rotation if specified
        if fault_json.get('rotation', 'no').lower() == 'yes':
            self.logger.info("Applying counterclockwise 90° rotation to fault geometry")
            # Apply rotation: (x,y) -> (-y,x)
            fault_start_rotated = np.array([-fault_start[1], fault_start[0], fault_start[2]])
            fault_end_rotated = np.array([-fault_end[1], fault_end[0], fault_end[2]])
            rotation_applied = 'counterclockwise_90deg'
            strike_rotated = 0.0  # N-S after rotation
        else:
            fault_start_rotated = fault_start
            fault_end_rotated = fault_end
            rotation_applied = 'none'
            strike_rotated = fault_json['fault_strike']
        
        # Analyze station distribution for domain extent
        locations = conversion_data['locations']
        x_coords = locations[:, 0]
        y_coords = locations[:, 1] 
        z_coords = locations[:, 2]
        
        fault_data = {
            'fault_type': fault_json['fault_type'],
            'fault_trace_start': fault_start_rotated,
            'fault_trace_end': fault_end_rotated,
            'fault_dip': fault_json['fault_dip'],
            'fault_strike': strike_rotated,
            'coordinate_units': fault_json['coordinate_units'],
            'fault_length': fault_json['fault_length'],
            'rotation_applied': rotation_applied,
            'domain_extent': {
                'x_range': [float(x_coords.min()), float(x_coords.max())],
                'y_range': [float(y_coords.min()), float(y_coords.max())],
                'z_range': [float(z_coords.min()), float(z_coords.max())]
            },
            'description': fault_json['description']
        }
        
        output_file = self.output_dir / "geometry.npz"
        np.savez_compressed(output_file, **fault_data)
        
        self.logger.info(f"Fault geometry NPZ created: {output_file}")
        self.logger.info(f"Final fault trace: ({fault_start_rotated[0]:.0f}, {fault_start_rotated[1]:.0f}) to ({fault_end_rotated[0]:.0f}, {fault_end_rotated[1]:.0f})")
        self.logger.info(f"Rotation applied: {rotation_applied}")
        return str(output_file)
    
    def convert_all_datasets(self) -> Dict[str, str]:
        """Convert all SeisSol datasets found in input directory"""
        start_time = time.time()
        self.logger.info(f"Starting conversion of {self.input_dir}")
        
        # Discover .h5 files
        h5_files = self.discover_h5_files()
        
        if not h5_files:
            raise ValueError("No .h5 files found in input directory")
        
        # Convert each .h5 file
        all_conversion_data = []
        for h5_file in h5_files:
            try:
                conversion_data = self.convert_dataset(h5_file)
                all_conversion_data.append(conversion_data)
            except Exception as e:
                self.logger.error(f"Failed to convert {h5_file}: {e}")
                continue
        
        if not all_conversion_data:
            raise ValueError("No datasets were successfully converted")
        
        # Merge multiple files if necessary
        merged_data = self.merge_multiple_files(all_conversion_data)
        
        # Create NPZ files
        station_file = self.create_station_list_npz(merged_data)
        velocity_file = self.create_velocity_database_npz(merged_data)
        geometry_file = self.create_fault_geometry_npz(merged_data)
        
        # Summary
        total_stations = len(merged_data['station_ids'])
        duration = time.time() - start_time
        
        self.logger.info(f"Conversion complete in {duration:.2f}s")
        self.logger.info(f"Total stations: {total_stations}")
        self.logger.info(f"Files processed: {len(h5_files)}")
        self.logger.info(f"Files created: {station_file}, {velocity_file}, {geometry_file}")
        
        return {
            'station_list': station_file,
            'velocity_database': velocity_file,
            'fault_geometry': geometry_file,
            'total_stations': total_stations,
            'files_processed': len(h5_files),
            'conversion_time': duration
        }

def main():
    """Main entry point for the converter API"""
    parser = argparse.ArgumentParser(description='Convert SeisSol datasets to DR4GM NPZ format')
    parser.add_argument('--input_dir', required=True, help='Input directory containing SeisSol .h5 files')
    parser.add_argument('--output_dir', required=True, help='Output directory for NPZ files')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        converter = SeisSolConverter(args.input_dir, args.output_dir)
        results = converter.convert_all_datasets()
        
        print("\n=== Conversion Results ===")
        print(f"Station list: {results['station_list']}")
        print(f"Velocity database: {results['velocity_database']}")
        print(f"Fault geometry: {results['fault_geometry']}")
        print(f"Total stations: {results['total_stations']}")
        print(f"Files processed: {results['files_processed']}")
        print(f"Conversion time: {results['conversion_time']:.2f}s")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()