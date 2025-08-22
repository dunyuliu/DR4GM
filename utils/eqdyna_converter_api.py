#!/usr/bin/env python3

"""
EQDyna Dataset Converter API

Converts EQDyna scenario datasets to parsable station lists and velocity time history databases
in NPZ format following the DR4GM standard.

Usage:
    python eqdyna_converter_api.py --input_dir <eqdyna_dir> --output_dir <output_dir>
    
Example:
    python eqdyna_converter_api.py --input_dir ../../datasets/eqdyna/eqdyna.0001.A.100m --output_dir ./converted_data
"""

import os
import sys
import argparse
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
import time
from pathlib import Path

# Import DR4GM standards
from npz_format_standard import DR4GM_NPZ_Standard

class EQDynaConverter:
    """Convert EQDyna datasets to DR4GM NPZ format"""
    
    def __init__(self, input_dir: str, output_dir: str, dt: float = None):
        """
        Initialize converter
        
        Args:
            input_dir: Directory containing eqdyna data files
            output_dir: Directory to save converted NPZ files
            dt: Time step (if None, will try to read from user_defined_params.py)
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Data type for binary files
        self.dtype = np.float64
        self.value_size = np.dtype(self.dtype).itemsize
        
        # Try to get dt from parameters file
        self.dt = dt
        if self.dt is None:
            self.dt = self._get_dt_from_params()
        
        # Station and time series data
        self.stations_data = {}
        self.chunk_mapping = {}
        self.global_station_id = 0
        
        # Logging setup
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
    
    def _get_dt_from_params(self) -> float:
        """Try to get dt from user_defined_params.py and multiply by 10 for GM sampling"""
        try:
            params_file = self.input_dir / 'user_defined_params.py'
            if params_file.exists():
                sys.path.insert(0, str(self.input_dir))
                from user_defined_params import par
                # Multiply by 10 for GM sampling rate as per eqdyna convention
                return par.dt * 10
        except:
            self.logger.warning("Could not read dt from user_defined_params.py, using default 0.05")
        
        return 0.05  # Default dt value (0.005 * 10)
    
    def discover_chunks(self) -> List[int]:
        """Discover all available chunk IDs from surface coordinate files"""
        chunk_ids = []
        
        for file_path in self.input_dir.glob("surface_coor.txt*"):
            chunk_str = file_path.name.replace("surface_coor.txt", "")
            if chunk_str.isdigit():
                chunk_ids.append(int(chunk_str))
            elif chunk_str == "":  # surface_coor.txt without number
                chunk_ids.append(0)
        
        chunk_ids.sort()
        self.logger.info(f"Discovered {len(chunk_ids)} chunks: {chunk_ids}")
        return chunk_ids
    
    def load_station_locations(self, chunk_id: int) -> np.ndarray:
        """Load station locations from surface_coor.txt file"""
        if chunk_id == 0:
            filename = "surface_coor.txt"
        else:
            filename = f"surface_coor.txt{chunk_id}"
        
        filepath = self.input_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Station coordinate file not found: {filepath}")
        
        # Load coordinates (X, Y, Z)
        coords = np.loadtxt(filepath)
        
        # Ensure 3D coordinates
        if coords.ndim == 1:
            coords = coords.reshape(1, -1)
        
        if coords.shape[1] < 3:
            # Add Z=0 if missing
            z_coords = np.zeros((coords.shape[0], 1))
            coords = np.column_stack([coords, z_coords])
        
        # Apply counterclockwise 90-degree rotation: (x, y) -> (-y, x)
        # This aligns the coordinate system with standard conventions
        x_orig = coords[:, 0].copy()
        y_orig = coords[:, 1].copy()
        coords[:, 0] = -y_orig  # new_x = -old_y
        coords[:, 1] = x_orig   # new_y = old_x
        # Z coordinate remains unchanged
        
        self.logger.info(f"Loaded {coords.shape[0]} stations from chunk {chunk_id}")
        self.logger.info(f"Applied counterclockwise 90° rotation: (x,y) -> (-y,x)")
        return coords
    
    def get_all_velocity_timeseries(self, chunk_id: int, coords: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract velocity time series for all stations in chunk using vectorized operations"""
        if chunk_id == 0:
            gm_filename = "gm"
        else:
            gm_filename = f"gm{chunk_id}"
        
        gm_filepath = self.input_dir / gm_filename
        
        if not gm_filepath.exists():
            raise FileNotFoundError(f"GM binary file not found: {gm_filepath}")
        
        # Calculate file structure
        file_size = gm_filepath.stat().st_size
        num_data_points = file_size // self.value_size
        num_stations = coords.shape[0]
        components_per_station = 3  # strike, normal, vertical
        num_time_steps = num_data_points // (num_stations * components_per_station)
        
        self.logger.info(f"Loading binary data: {num_stations} stations, {num_time_steps} time steps")
        
        # Load entire binary file at once
        all_data = np.fromfile(gm_filepath, dtype=self.dtype)
        
        # Verify data size
        expected_size = num_stations * components_per_station * num_time_steps
        if len(all_data) < expected_size:
            self.logger.warning(f"File size mismatch: got {len(all_data)}, expected {expected_size}")
            all_data = np.pad(all_data, (0, expected_size - len(all_data)), mode='constant')
        elif len(all_data) > expected_size:
            all_data = all_data[:expected_size]
        
        try:
            # Reshape to [num_time_steps, num_stations, 3] using vectorized operations
            # Data format: [t0_s0_c0, t0_s0_c1, t0_s0_c2, t0_s1_c0, t0_s1_c1, t0_s1_c2, ...]
            reshaped_data = all_data.reshape(num_time_steps, num_stations, components_per_station)
            
            # Extract components and transpose to [num_stations, num_time_steps]
            vel_strike = reshaped_data[:, :, 0].T    # Strike component
            vel_normal = reshaped_data[:, :, 1].T    # Normal component  
            vel_vertical = reshaped_data[:, :, 2].T  # Vertical component
            
            self.logger.info("Completed vectorized time series extraction")
            
        except Exception as e:
            self.logger.error(f"Vectorized reshape failed: {e}, falling back to loop method")
            
            # Fallback to loop method if vectorization fails
            vel_strike = np.zeros((num_stations, num_time_steps))
            vel_normal = np.zeros((num_stations, num_time_steps))
            vel_vertical = np.zeros((num_stations, num_time_steps))
            
            for station_idx in range(num_stations):
                if station_idx % 1000 == 0:
                    self.logger.info(f"  Processing station {station_idx}/{num_stations} ({100*station_idx/num_stations:.1f}%)")
                
                # Calculate indices for this station
                indices_strike = np.arange(num_time_steps) * num_stations * 3 + station_idx * 3
                
                for i, index in enumerate(indices_strike):
                    if index + 2 < len(all_data):
                        vel_strike[station_idx, i] = all_data[index]
                        vel_normal[station_idx, i] = all_data[index + 1]
                        vel_vertical[station_idx, i] = all_data[index + 2]
            
            self.logger.info("Completed fallback loop-based extraction")
        
        return {
            'strike': vel_strike,
            'normal': vel_normal, 
            'vertical': vel_vertical,
            'time_steps': num_time_steps,
            'dt': self.dt
        }
    
    def convert_chunk(self, chunk_id: int) -> Dict:
        """Convert a single chunk to station data using vectorized operations"""
        self.logger.info(f"Converting chunk {chunk_id}")
        
        # Load station coordinates
        coords = self.load_station_locations(chunk_id)
        num_stations = coords.shape[0]
        
        # Extract velocity time series for all stations at once using vectorized operations
        try:
            velocity_data = self.get_all_velocity_timeseries(chunk_id, coords)
            
            # Generate station IDs
            station_ids = np.arange(self.global_station_id, self.global_station_id + num_stations)
            self.global_station_id += num_stations
            
            # Prepare chunk data
            chunk_data = {
                'station_ids': station_ids,
                'locations': coords,
                'chunk_ids': np.full(num_stations, chunk_id),
                'local_ids': np.arange(num_stations),
                'vel_strike': velocity_data['strike'],
                'vel_normal': velocity_data['normal'],
                'vel_vertical': velocity_data['vertical'],
                'time_steps': velocity_data['time_steps'],
                'dt': self.dt
            }
            
            self.logger.info(f"Chunk {chunk_id}: Converted {num_stations} stations using vectorized operations")
            
        except Exception as e:
            self.logger.error(f"Error processing chunk {chunk_id}: {e}")
            # Return empty chunk data on failure
            chunk_data = {
                'station_ids': np.array([]),
                'locations': np.array([]).reshape(0, 3),
                'chunk_ids': np.array([]),
                'local_ids': np.array([]),
                'vel_strike': np.array([]).reshape(0, 0),
                'vel_normal': np.array([]).reshape(0, 0),
                'vel_vertical': np.array([]).reshape(0, 0),
                'time_steps': 0,
                'dt': self.dt
            }
        
        return chunk_data
    
    def create_station_list_npz(self, all_chunks_data: List[Dict]) -> str:
        """Create station list NPZ file following DR4GM standard"""
        self.logger.info("Creating station list NPZ file")
        
        # Combine all chunk data
        combined_data = {
            'station_ids': np.concatenate([chunk['station_ids'] for chunk in all_chunks_data]),
            'locations': np.vstack([chunk['locations'] for chunk in all_chunks_data]),
            'chunk_ids': np.concatenate([chunk['chunk_ids'] for chunk in all_chunks_data]),
            'local_ids': np.concatenate([chunk['local_ids'] for chunk in all_chunks_data]),
            'station_types': np.ones(sum(len(chunk['station_ids']) for chunk in all_chunks_data)),  # All surface stations
            'coordinate_units': 'm'
        }
        
        output_file = self.output_dir / "stations.npz"
        np.savez_compressed(output_file, **combined_data)
        
        # Validate against DR4GM standard
        is_valid = DR4GM_NPZ_Standard.validate_npz(str(output_file), 'layer2_stations')
        self.logger.info(f"Station list NPZ created: {output_file} (Valid: {is_valid})")
        
        return str(output_file)
    
    def create_velocity_database_npz(self, all_chunks_data: List[Dict]) -> str:
        """Create velocity time history database NPZ file"""
        self.logger.info("Creating velocity database NPZ file")
        
        # Get common time parameters
        time_steps = all_chunks_data[0]['time_steps']
        dt = all_chunks_data[0]['dt']
        
        # Combine velocity data
        all_vel_strike = np.vstack([chunk['vel_strike'] for chunk in all_chunks_data])
        all_vel_normal = np.vstack([chunk['vel_normal'] for chunk in all_chunks_data])
        all_vel_vertical = np.vstack([chunk['vel_vertical'] for chunk in all_chunks_data])
        
        velocity_data = {
            'station_ids': np.concatenate([chunk['station_ids'] for chunk in all_chunks_data]),
            'locations': np.vstack([chunk['locations'] for chunk in all_chunks_data]),
            'vel_strike': all_vel_strike,
            'vel_normal': all_vel_normal,
            'vel_vertical': all_vel_vertical,
            'time_steps': np.full(len(all_vel_strike), time_steps),
            'dt_values': np.full(len(all_vel_strike), dt),
            'duration': time_steps * dt,
            'units': 'm/s',
            'coordinate_units': 'm'
        }
        
        output_file = self.output_dir / "velocities.npz"
        np.savez_compressed(output_file, **velocity_data)
        
        # Validate against DR4GM standard
        is_valid = DR4GM_NPZ_Standard.validate_npz(str(output_file), 'layer3_velocities')
        self.logger.info(f"Velocity database NPZ created: {output_file} (Valid: {is_valid})")
        
        return str(output_file)
    
    def create_fault_geometry_npz(self) -> str:
        """Create fault geometry NPZ file for eqdyna scenario"""
        self.logger.info("Creating fault geometry NPZ file")
        
        # Original EQDyna fault geometry: strike-slip fault from -20km to +20km along X-axis at Y=0
        # After counterclockwise 90° rotation: (x,y) -> (-y,x)
        # Original: (-20000, 0) to (20000, 0) along X-axis
        # Rotated: (0, -20000) to (0, 20000) along Y-axis
        
        fault_data = {
            'fault_type': 'strike-slip',
            'fault_trace_start': np.array([0.0, -20000.0, 0.0]),  # Start point after rotation (m)
            'fault_trace_end': np.array([0.0, 20000.0, 0.0]),     # End point after rotation (m)
            'fault_dip': 90.0,  # Vertical fault
            'fault_strike': 90.0,  # Along Y-axis after rotation (North-South)
            'coordinate_units': 'm',
            'fault_length': 40000.0,  # Total fault length in meters
            'rotation_applied': 'counterclockwise_90deg',
            'description': 'EQDyna strike-slip fault after 90° CCW rotation: Y=-20km to Y=+20km along Y-axis, vertical fault at X=0'
        }
        
        output_file = self.output_dir / "geometry.npz"
        np.savez_compressed(output_file, **fault_data)
        
        self.logger.info(f"Fault geometry NPZ created: {output_file}")
        self.logger.info(f"Fault trace: ({fault_data['fault_trace_start'][0]:.0f}, {fault_data['fault_trace_start'][1]:.0f}) to ({fault_data['fault_trace_end'][0]:.0f}, {fault_data['fault_trace_end'][1]:.0f})")
        return str(output_file)
    
    def convert_dataset(self) -> Dict[str, str]:
        """Convert complete EQDyna dataset to NPZ format"""
        start_time = time.time()
        self.logger.info(f"Starting conversion of {self.input_dir}")
        
        # Discover available chunks
        chunk_ids = self.discover_chunks()
        
        if not chunk_ids:
            raise ValueError("No valid chunks found in input directory")
        
        # Convert each chunk
        all_chunks_data = []
        for chunk_id in chunk_ids:
            try:
                chunk_data = self.convert_chunk(chunk_id)
                all_chunks_data.append(chunk_data)
            except Exception as e:
                self.logger.error(f"Failed to convert chunk {chunk_id}: {e}")
                continue
        
        if not all_chunks_data:
            raise ValueError("No chunks were successfully converted")
        
        # Create NPZ files
        station_file = self.create_station_list_npz(all_chunks_data)
        velocity_file = self.create_velocity_database_npz(all_chunks_data)
        geometry_file = self.create_fault_geometry_npz()
        
        # Summary
        total_stations = sum(len(chunk['station_ids']) for chunk in all_chunks_data)
        duration = time.time() - start_time
        
        self.logger.info(f"Conversion complete in {duration:.2f}s")
        self.logger.info(f"Total stations: {total_stations}")
        self.logger.info(f"Files created: {station_file}, {velocity_file}, {geometry_file}")
        
        return {
            'station_list': station_file,
            'velocity_database': velocity_file,
            'fault_geometry': geometry_file,
            'total_stations': total_stations,
            'chunks_processed': len(all_chunks_data),
            'conversion_time': duration
        }

def main():
    """Main entry point for the converter API"""
    parser = argparse.ArgumentParser(description='Convert EQDyna datasets to DR4GM NPZ format')
    parser.add_argument('--input_dir', required=True, help='Input directory containing eqdyna files')
    parser.add_argument('--output_dir', required=True, help='Output directory for NPZ files')
    parser.add_argument('--dt', type=float, help='Time step (default: read from user_defined_params.py)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        converter = EQDynaConverter(args.input_dir, args.output_dir, args.dt)
        results = converter.convert_dataset()
        
        print("\n=== Conversion Results ===")
        print(f"Station list: {results['station_list']}")
        print(f"Velocity database: {results['velocity_database']}")
        print(f"Total stations: {results['total_stations']}")
        print(f"Chunks processed: {results['chunks_processed']}")
        print(f"Conversion time: {results['conversion_time']:.2f}s")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()