#!/usr/bin/env python3
"""
MAFE Surface Velocity Dataset Converter API

Converts the "3realizations_M7" MATLAB datasets provided by the MAFE campaign
into the standard DR4GM NPZ station and velocity files so they can flow through
the unified processing pipeline.

Each realization MAT file stores surface velocity time histories sampled on a
regular strike-parallel/fault-normal grid with 1 km spacing and 0.007 s time
step. The variables of interest are:
    - vx: fault-normal component at the surface (km grid, meters per second)
    - vy: fault-parallel component at the surface (m/s)
    - vz: vertical component at the surface (m/s)
    - Mw: scalar moment magnitude for the realization

Usage:
    python mafe_converter_api.py --input_dir <dir> --output_dir <out_dir>

The converter writes three NPZ files into the output directory:
    - stations.npz  : grid of surface stations and metadata (layer2)
    - velocities.npz: velocity time histories aligned with station ids (layer3)
    - geometry.npz  : simplified fault geometry for distance calculations

The converter automatically searches the input directory (recursively) for a
single ``*.mat`` file to process. If no files are found, or more than one is
present, the conversion stops with an error so that each run handles one
realization at a time.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from scipy.io import loadmat

from npz_format_standard import DR4GM_NPZ_Standard


class MafeConverter:
    """Convert MAFE MATLAB realizations into DR4GM NPZ files."""
    
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

    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        dt: float = 0.007,
        x_start_km: float = -10.0,
        y_start_km: float = -60.0,
        spacing_km: float = 1.0,
        log_level: int = logging.INFO,
    ) -> None:
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.mat_path = self._resolve_mat_file()
        self.dt = dt
        self.x_start_km = x_start_km
        self.y_start_km = y_start_km
        self.spacing_km = spacing_km

        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------
    def _resolve_mat_file(self) -> Path:
        """Pick the MAT file to convert based on directory contents."""
        mat_files = sorted(self.input_dir.rglob("*.mat"))
        if not mat_files:
            raise FileNotFoundError(f"No MAT files found in {self.input_dir}")
        if len(mat_files) > 1:
            choices = ", ".join(str(path.relative_to(self.input_dir)) for path in mat_files)
            raise ValueError(
                "Multiple MAT files found under {}: {}. Specify a directory containing only one realization.".format(
                    self.input_dir, choices
                )
            )
        return mat_files[0]

    # ------------------------------------------------------------------
    def _load_realization(self) -> Dict[str, np.ndarray]:
        """Load MATLAB data arrays for the chosen realization."""
        self.logger.info("Loading realization file: %s", self.mat_path.name)
        data = loadmat(self.mat_path)

        required_vars = ["vx", "vy", "vz"]
        missing = [var for var in required_vars if var not in data]
        if missing:
            raise KeyError(
                f"MAT file {self.mat_path.name} missing required variables: {missing}"
            )

        return data

    # ------------------------------------------------------------------
    def _derive_grid_coordinates(self, nx: int, ny: int) -> Tuple[np.ndarray, np.ndarray]:
        """Build strike (y) and normal (x) coordinate arrays in meters."""
        x_coords_km = self.x_start_km + self.spacing_km * np.arange(nx)
        y_coords_km = self.y_start_km + self.spacing_km * np.arange(ny)

        x_coords_m = x_coords_km * 1000.0
        y_coords_m = y_coords_km * 1000.0

        self.logger.info(
            "Generated coordinate grid: %d x points from %.1f to %.1f km, %d y points from %.1f to %.1f km",
            nx,
            x_coords_km[0],
            x_coords_km[-1],
            ny,
            y_coords_km[0],
            y_coords_km[-1],
        )

        return x_coords_m, y_coords_m

    # ------------------------------------------------------------------
    def convert(self) -> Dict[str, str]:
        """Execute the conversion pipeline and return generated file paths."""
        mat_data = self._load_realization()

        vx = np.asarray(mat_data["vx"], dtype=np.float32)
        vy = np.asarray(mat_data["vy"], dtype=np.float32)
        vz = np.asarray(mat_data["vz"], dtype=np.float32)

        if vx.shape != vy.shape or vx.shape != vz.shape:
            raise ValueError(
                f"Velocity components have inconsistent shapes: vx{vx.shape}, vy{vy.shape}, vz{vz.shape}"
            )

        nx, ny, nt = vx.shape
        x_coords_m, y_coords_m = self._derive_grid_coordinates(nx, ny)

        # Flatten 2D station grid into station-major arrays.
        X, Y = np.meshgrid(x_coords_m, y_coords_m, indexing="ij")
        station_coords = np.column_stack(
            [X.reshape(-1), Y.reshape(-1), np.zeros(nx * ny, dtype=np.float32)]
        )
        station_ids = np.arange(nx * ny)

        # Reshape velocity cubes into [stations, time] tables.
        vel_normal = vx.reshape(nx * ny, nt)
        vel_strike = vy.reshape(nx * ny, nt)
        vel_vertical = vz.reshape(nx * ny, nt)

        moment_magnitude = float(np.squeeze(mat_data.get("Mw", np.array([[np.nan]]))))

        stations_npz = self._write_stations_npz(
            station_ids,
            station_coords,
            moment_magnitude,
        )
        velocities_npz = self._write_velocities_npz(
            station_ids,
            station_coords,
            vel_strike,
            vel_normal,
            vel_vertical,
            nt,
        )
        geometry_npz = self._write_geometry_npz(ny, moment_magnitude)

        self.logger.info(
            "Finished conversion for %s -> %s, %s, %s",
            self.mat_path.name,
            stations_npz,
            velocities_npz,
            geometry_npz,
        )

        return {
            "stations": stations_npz,
            "velocities": velocities_npz,
            "geometry": geometry_npz,
        }

    # ------------------------------------------------------------------
    def _write_stations_npz(
        self,
        station_ids: np.ndarray,
        locations: np.ndarray,
        moment_magnitude: float,
    ) -> str:
        station_payload = {
            "station_ids": station_ids,
            "locations": locations,
            "chunk_ids": np.zeros_like(station_ids),
            "local_ids": np.arange(len(station_ids)),
            "station_types": np.ones_like(station_ids),
            "coordinate_units": "m",
            "metadata_Mw": np.array([moment_magnitude], dtype=np.float32),
            "source_file": np.array([self.mat_path.name]),
        }

        output_path = self.output_dir / "stations.npz"
        np.savez_compressed(output_path, **station_payload)

        DR4GM_NPZ_Standard.validate_npz(str(output_path), "layer2_stations")
        return str(output_path)

    # ------------------------------------------------------------------
    def _write_velocities_npz(
        self,
        station_ids: np.ndarray,
        locations: np.ndarray,
        vel_strike: np.ndarray,
        vel_normal: np.ndarray,
        vel_vertical: np.ndarray,
        time_steps: int,
    ) -> str:
        velocity_payload = {
            "station_ids": station_ids,
            "locations": locations,
            "vel_strike": vel_strike,
            "vel_normal": vel_normal,
            "vel_vertical": vel_vertical,
            "time_steps": np.full(len(station_ids), time_steps),
            "dt_values": np.full(len(station_ids), self.dt, dtype=np.float32),
            "duration": time_steps * self.dt,
            "units": "m/s",
            "coordinate_units": "m",
        }

        output_path = self.output_dir / "velocities.npz"
        np.savez_compressed(output_path, **velocity_payload)

        DR4GM_NPZ_Standard.validate_npz(str(output_path), "layer3_velocities")
        return str(output_path)

    # ------------------------------------------------------------------
    def _write_geometry_npz(self, ny: int, moment_magnitude: float) -> str:
        """Create fault geometry NPZ file using fault_geometry.json"""
        self.logger.info("Creating fault geometry NPZ file")
        
        # Load fault geometry from JSON file
        fault_json = self.load_fault_geometry_json()
        
        # Extract data from JSON
        fault_start = np.array(fault_json['fault_trace_start'], dtype=np.float32)
        fault_end = np.array(fault_json['fault_trace_end'], dtype=np.float32)
        
        # Apply rotation if specified (MAFE typically doesn't need rotation)
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
            'moment_magnitude': np.float32(moment_magnitude),
        }
        
        # Add moment magnitude from JSON if present (prioritize over parameter)
        if 'moment_magnitude' in fault_json:
            geometry_payload['moment_magnitude'] = np.float32(fault_json['moment_magnitude'])
        
        output_path = self.output_dir / "geometry.npz"
        np.savez_compressed(output_path, **geometry_payload)
        
        self.logger.info(f"Fault geometry NPZ created: {output_path}")
        self.logger.info(f"Fault trace: {fault_start_rotated} to {fault_end_rotated}")
        return str(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert MAFE MATLAB realizations to DR4GM NPZ format",
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Directory containing realizationX.mat files",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory where NPZ files will be written",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.007,
        help="Time step in seconds (default: 0.007 s)",
    )
    parser.add_argument(
        "--x_start_km",
        type=float,
        default=-10.0,
        help="Starting fault-normal coordinate in km (default: -10)",
    )
    parser.add_argument(
        "--y_start_km",
        type=float,
        default=-60.0,
        help="Starting fault-parallel coordinate in km (default: -60)",
    )
    parser.add_argument(
        "--spacing_km",
        type=float,
        default=1.0,
        help="Grid spacing in km for both directions (default: 1 km)",
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
    converter = MafeConverter(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        dt=args.dt,
        x_start_km=args.x_start_km,
        y_start_km=args.y_start_km,
        spacing_km=args.spacing_km,
        log_level=log_level,
    )
    converter.convert()


if __name__ == "__main__":
    main()
