#!/usr/bin/env python3

"""
Waveqlab3d Dataset Converter API

Converts Waveqlab3d scenario datasets to parsable station lists and velocity time history databases
in NPZ format following the DR4GM standard.

Based exactly on extract_new_saway.py logic for station generation.

Usage:
    python waveqlab3d_converter_api.py --input_dir <waveqlab3d_dir> --output_dir <output_dir>
"""

import os
import sys
import argparse
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
import time
from pathlib import Path
import glob

# Import DR4GM standards
from npz_format_standard import DR4GM_NPZ_Standard

class Waveqlab3dConverter:
    """Convert Waveqlab3d datasets to DR4GM NPZ format"""
    
    def __init__(self, input_dir: str, output_dir: str, dt: float = None):
        """
        Initialize converter
        
        Args:
            input_dir: Directory containing waveqlab3d data files
            output_dir: Directory to save converted NPZ files
            dt: Time step (default: 0.314775373058478677E-02*16 from extract_new_saway.py)
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Data type for binary files (float32 as per extract_new_saway.py)
        self.dtype = np.float32
        
        # Time step from extract_new_saway.py
        self.dt = dt if dt is not None else 0.314775373058478677E-02 * 16
        
        # Grid parameters from extract_new_saway.py (lines 55-58)
        self.lsl = 401  # left side length
        self.rsl = 401  # right side length  
        self.stl = 1601  # strike length
        
        # Station data
        self.global_station_id = 0
        
        # Logging setup
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
    
    def discover_hslice_files(self) -> Dict[str, List[Path]]:
        """Discover all Hslice files in input directory"""
        hslice_patterns = [
            "*.Hslice1seisx",
            "*.Hslice1seisy", 
            "*.Hslice2seisx",
            "*.Hslice2seisy"
        ]
        
        file_sets = {}
        
        for pattern in hslice_patterns:
            files = list(self.input_dir.glob(pattern))
            if files:
                # Extract base filename (everything before .Hslice)
                base_name = files[0].name.split('.Hslice')[0]
                if base_name not in file_sets:
                    file_sets[base_name] = {}
                
                component = pattern.replace("*.", "").replace("Hslice", "")
                file_sets[base_name][component] = files[0]
        
        self.logger.info(f"Discovered {len(file_sets)} file sets: {list(file_sets.keys())}")
        return file_sets
    
    def generate_station_coordinates(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate station coordinates exactly as extract_new_saway.py lines 59-72
        
        Returns:
            left_coords: Coordinates for left side of fault [N_left, 3]
            right_coords: Coordinates for right side of fault [N_right, 3]
        """
        self.logger.info("Generating station coordinates using extract_new_saway.py logic")
        
        # Initialize arrays exactly as extract_new_saway.py lines 59-62
        lsx = np.zeros(self.lsl * self.stl)
        lsy = np.zeros(self.lsl * self.stl)
        rsx = np.zeros(self.rsl * self.stl)
        rsy = np.zeros(self.rsl * self.stl)
        
        # Y coordinate array exactly as extract_new_saway.py line 64
        templ = np.linspace(0, 79.9, self.stl)
        
        # Generate coordinates exactly as extract_new_saway.py lines 66-72
        # With vertical fault at X=20 constant
        fault_x = 20.0  # Vertical fault at X=20km
        
        for i in range(self.stl):
            # Left side X: from 0 to fault location (line 68) - convert km to m
            lsx[self.lsl*i:self.lsl*(i+1)] = np.linspace(0, fault_x, self.lsl) * 1000.0
            # Left side Y: constant Y value (line 69) - convert km to m
            lsy[self.lsl*i:self.lsl*(i+1)] = np.ones(self.lsl) * templ[i] * 1000.0
            
            # Right side X: from fault location to 40 km (line 71) - convert km to m
            rsx[self.rsl*i:self.rsl*(i+1)] = np.linspace(fault_x, 40, self.rsl) * 1000.0
            # Right side Y: constant Y value (line 72) - convert km to m
            rsy[self.rsl*i:self.rsl*(i+1)] = np.ones(self.rsl) * templ[i] * 1000.0
        
        # Convert to 3D coordinate arrays
        left_coords = np.column_stack([lsx, lsy, np.zeros(len(lsx))])
        right_coords = np.column_stack([rsx, rsy, np.zeros(len(rsx))])
        
        self.logger.info(f"Generated {len(left_coords)} left-side and {len(right_coords)} right-side stations")
        self.logger.info(f"Left side X range: {lsx.min():.1f} to {lsx.max():.1f} m")
        self.logger.info(f"Right side X range: {rsx.min():.1f} to {rsx.max():.1f} m") 
        self.logger.info(f"Y range: {lsy.min():.1f} to {lsy.max():.1f} m")
        
        return left_coords, right_coords
    
    def load_hslice_data(self, hslice_file: Path) -> np.ndarray:
        """Load time series data from Hslice binary file"""
        if not hslice_file.exists():
            raise FileNotFoundError(f"Hslice file not found: {hslice_file}")
        
        # Load as float32 binary data exactly as extract_new_saway.py
        data = np.fromfile(hslice_file, dtype=self.dtype)
        self.logger.info(f"Loaded {len(data)} data points from {hslice_file.name}")
        return data
    
    def extract_station_timeseries(self, data: np.ndarray, side: str) -> np.ndarray:
        """
        Extract individual station time series from waveqlab3d data
        Based on extract_new_saway.py extraction logic
        
        Args:
            data: Raw time series data from Hslice file
            side: 'left' or 'right' side of fault
            
        Returns:
            Array of shape [n_stations, n_time_steps]
        """
        if side == 'left':
            n_stations = self.lsl
        else:
            n_stations = self.rsl
        
        # Calculate number of time steps
        total_points = len(data)
        expected_points = n_stations * self.stl
        
        if total_points % expected_points != 0:
            self.logger.warning(f"Data size {total_points} not evenly divisible by expected {expected_points}")
        
        n_time_steps = total_points // expected_points
        total_stations = n_stations * self.stl
        self.logger.info(f"Extracting time series for {side} side: {total_stations} stations, {n_time_steps} time steps")
        
        # Use vectorized reshaping for performance
        try:
            # Trim data to exact expected size
            data_trimmed = data[:n_time_steps * total_stations]
            
            # Reshape: data is stored as [st0_t0, st1_t0, ..., stN_t0, st0_t1, st1_t1, ...]
            timeseries = data_trimmed.reshape(n_time_steps, total_stations).T
            
            self.logger.info(f"Completed time series extraction for {side} side using vectorized reshaping")
            return timeseries
            
        except Exception as e:
            self.logger.error(f"Vectorized reshape failed: {e}")
            raise e
    
    def convert_dataset(self, file_set_name: str, file_paths: Dict[str, Path]) -> Dict:
        """Convert a single waveqlab3d file set"""
        self.logger.info(f"Converting file set: {file_set_name}")
        
        # Generate station coordinates
        left_coords, right_coords = self.generate_station_coordinates()
        
        # Load Hslice data
        slice1_x = self.load_hslice_data(file_paths['1seisx'])  # Left side X
        slice1_y = self.load_hslice_data(file_paths['1seisy'])  # Left side Y
        slice2_x = self.load_hslice_data(file_paths['2seisx'])  # Right side X
        slice2_y = self.load_hslice_data(file_paths['2seisy'])  # Right side Y
        
        # Extract time series for each side
        left_vel_x = self.extract_station_timeseries(slice1_x, 'left')
        left_vel_y = self.extract_station_timeseries(slice1_y, 'left')
        right_vel_x = self.extract_station_timeseries(slice2_x, 'right')
        right_vel_y = self.extract_station_timeseries(slice2_y, 'right')
        
        # Combine left and right sides
        all_coords = np.vstack([left_coords, right_coords])
        all_vel_x = np.vstack([left_vel_x, right_vel_x])
        all_vel_y = np.vstack([left_vel_y, right_vel_y])
        
        # Generate vertical component (zeros for now)
        all_vel_z = np.zeros_like(all_vel_x)
        
        # Generate station IDs
        n_total_stations = len(all_coords)
        station_ids = np.arange(self.global_station_id, self.global_station_id + n_total_stations)
        self.global_station_id += n_total_stations
        
        # Store conversion data
        conversion_data = {
            'station_ids': station_ids,
            'locations': all_coords,
            'vel_strike': all_vel_x,  # Fault parallel (X)
            'vel_normal': all_vel_y,  # Fault perpendicular (Y)
            'vel_vertical': all_vel_z,  # Vertical (Z)
            'time_steps': all_vel_x.shape[1],
            'dt': self.dt,
            'file_set_name': file_set_name
        }
        
        self.logger.info(f"Converted {n_total_stations} stations with {all_vel_x.shape[1]} time steps")
        return conversion_data
    
    def create_station_list_npz(self, conversion_data: Dict) -> str:
        """Create station list NPZ file following DR4GM standard"""
        self.logger.info("Creating station list NPZ file")
        
        station_data = {
            'station_ids': conversion_data['station_ids'],
            'locations': conversion_data['locations'],
            'chunk_ids': np.zeros(len(conversion_data['station_ids'])),
            'local_ids': np.arange(len(conversion_data['station_ids'])),
            'station_types': np.ones(len(conversion_data['station_ids'])),
            'coordinate_units': 'm'
        }
        
        output_file = self.output_dir / "stations.npz"
        np.savez_compressed(output_file, **station_data)
        
        # Validate against DR4GM standard
        is_valid = DR4GM_NPZ_Standard.validate_npz(str(output_file), 'layer2_stations')
        self.logger.info(f"Station list NPZ created: {output_file} (Valid: {is_valid})")
        
        # Create station distribution plot
        self._plot_station_distribution(conversion_data)
        
        return str(output_file)
    
    def _plot_station_distribution(self, conversion_data: Dict) -> None:
        """Create station distribution plot and save to output directory"""
        import matplotlib.pyplot as plt
        
        locations = conversion_data['locations']
        x_coords = locations[:, 0]
        y_coords = locations[:, 1]
        
        self.logger.info("Creating station distribution plot")
        
        # Create figure
        plt.figure(figsize=(12, 10))
        
        # Plot stations (sample if too many for performance, but preserve grid structure)
        if len(x_coords) > 10000:
            # Sample by taking every Nth row and every Nth column to preserve grid structure
            # Total stations = 401*1601*2, we want ~10000 stations
            # So sample every ~10 rows and every ~10 columns
            y_step = max(1, self.stl // 100)  # Sample every 16th Y row  
            x_step = max(1, self.lsl // 40)   # Sample every 10th X column
            
            # Create sampling indices that preserve grid structure
            sample_indices = []
            for side in ['left', 'right']:
                side_start = 0 if side == 'left' else len(x_coords) // 2
                for i in range(0, self.stl, y_step):  # Y sampling
                    for j in range(0, self.lsl, x_step):  # X sampling
                        idx = side_start + i * self.lsl + j
                        if idx < len(x_coords):
                            sample_indices.append(idx)
            
            x_plot = x_coords[sample_indices]
            y_plot = y_coords[sample_indices]
            title_suffix = f'(grid sampled: every {y_step} Y rows, every {x_step} X cols)'
        else:
            x_plot = x_coords
            y_plot = y_coords
            title_suffix = '(all stations)'
        
        plt.scatter(x_plot, y_plot, s=0.5, alpha=0.6, c='blue')
        plt.xlabel('X coordinate (km)', fontsize=12)
        plt.ylabel('Y coordinate (km)', fontsize=12)
        plt.title(f'Waveqlab3d Station Distribution {title_suffix}', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        # Add statistics text
        unique_x = len(np.unique(x_coords))
        unique_y = len(np.unique(y_coords))
        plt.text(0.02, 0.98, f'Total stations: {len(x_coords):,}\\n' +
                            f'X range: {x_coords.min():.1f} - {x_coords.max():.1f} m\\n' +
                            f'Y range: {y_coords.min():.1f} - {y_coords.max():.1f} m\\n' +
                            f'Unique X: {unique_x:,}, Unique Y: {unique_y:,}',
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        # Save plot
        plot_file = self.output_dir / 'waveqlab3d_station_distribution.png'
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Station distribution plot saved: {plot_file}")
    
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
            'coordinate_units': 'm'
        }
        
        output_file = self.output_dir / "velocities.npz"
        np.savez_compressed(output_file, **velocity_data)
        
        # Validate against DR4GM standard
        is_valid = DR4GM_NPZ_Standard.validate_npz(str(output_file), 'layer3_velocities')
        self.logger.info(f"Velocity database NPZ created: {output_file} (Valid: {is_valid})")
        
        return str(output_file)
    
    def create_fault_geometry_npz(self) -> str:
        """Create fault geometry NPZ file for waveqlab3d scenario"""
        self.logger.info("Creating fault geometry NPZ file")
        
        # Waveqlab3d fault geometry: vertical fault at X=20km
        # No rotation needed - coordinates already in meters
        # Fault is 60km long, centered in the domain (Y=0 to 79.9km domain)
        
        fault_length_km = 60.0  # Waveqlab3d fault length
        domain_length_km = 79.9  # Y domain extends from 0 to 79.9km
        
        # Center the 60km fault in the 79.9km domain
        fault_start_km = (domain_length_km - fault_length_km) / 2  # 9.95 km
        fault_end_km = fault_start_km + fault_length_km            # 69.95 km
        
        fault_data = {
            'fault_type': 'strike-slip',
            'fault_trace_start': np.array([20000.0, fault_start_km * 1000.0, 0.0]),  # Start point (m)
            'fault_trace_end': np.array([20000.0, fault_end_km * 1000.0, 0.0]),     # End point (m)
            'fault_dip': 90.0,  # Vertical fault
            'fault_strike': 90.0,  # Along Y-axis (North-South)
            'coordinate_units': 'm',
            'fault_length': fault_length_km * 1000.0,  # Total fault length in meters
            'rotation_applied': 'none',
            'description': f'Waveqlab3d vertical fault at X=20km, Y={fault_start_km:.1f} to {fault_end_km:.1f}km, 60km long'
        }
        
        output_file = self.output_dir / "geometry.npz"
        np.savez_compressed(output_file, **fault_data)
        
        self.logger.info(f"Fault geometry NPZ created: {output_file}")
        self.logger.info(f"Fault trace: ({fault_data['fault_trace_start'][0]:.0f}, {fault_data['fault_trace_start'][1]:.0f}) to ({fault_data['fault_trace_end'][0]:.0f}, {fault_data['fault_trace_end'][1]:.0f})")
        return str(output_file)
    
    def convert_all_datasets(self) -> Dict[str, str]:
        """Convert all waveqlab3d datasets found in input directory"""
        start_time = time.time()
        self.logger.info(f"Starting conversion of {self.input_dir}")
        
        # Discover Hslice files
        file_sets = self.discover_hslice_files()
        
        if not file_sets:
            raise ValueError("No Hslice files found in input directory")
        
        # Convert the first file set found
        file_set_name = list(file_sets.keys())[0]
        file_paths = file_sets[file_set_name]
        
        # Check that we have all 4 required files
        required_components = ['1seisx', '1seisy', '2seisx', '2seisy']
        missing_components = [comp for comp in required_components if comp not in file_paths]
        
        if missing_components:
            raise ValueError(f"Missing required Hslice files: {missing_components}")
        
        # Convert the dataset
        conversion_data = self.convert_dataset(file_set_name, file_paths)
        
        # Create NPZ files
        station_file = self.create_station_list_npz(conversion_data)
        velocity_file = self.create_velocity_database_npz(conversion_data)
        geometry_file = self.create_fault_geometry_npz()
        
        # Summary
        total_stations = len(conversion_data['station_ids'])
        duration = time.time() - start_time
        
        self.logger.info(f"Conversion complete in {duration:.2f}s")
        self.logger.info(f"Total stations: {total_stations}")
        self.logger.info(f"Files created: {station_file}, {velocity_file}, {geometry_file}")
        
        return {
            'station_list': station_file,
            'velocity_database': velocity_file,
            'fault_geometry': geometry_file,
            'total_stations': total_stations,
            'file_set_name': file_set_name,
            'conversion_time': duration
        }

def main():
    """Main entry point for the converter API"""
    parser = argparse.ArgumentParser(description='Convert Waveqlab3d datasets to DR4GM NPZ format')
    parser.add_argument('--input_dir', required=True, help='Input directory containing waveqlab3d Hslice files')
    parser.add_argument('--output_dir', required=True, help='Output directory for NPZ files')
    parser.add_argument('--dt', type=float, help='Time step (default: 0.314775373058478677E-02*16)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        converter = Waveqlab3dConverter(args.input_dir, args.output_dir, args.dt)
        results = converter.convert_all_datasets()
        
        print("\n=== Conversion Results ===")
        print(f"File set: {results['file_set_name']}")
        print(f"Station list: {results['station_list']}")
        print(f"Velocity database: {results['velocity_database']}")
        print(f"Total stations: {results['total_stations']}")
        print(f"Conversion time: {results['conversion_time']:.2f}s")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()