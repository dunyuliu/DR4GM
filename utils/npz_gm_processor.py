#!/usr/bin/env python3

"""
Fast NPZ Ground Motion Processor

Vectorized ground motion processor for large NPZ datasets.
Uses batch processing and vectorized operations for maximum speed.

Usage:
    python npz_gm_processor_fast.py --velocity_npz <velocity_file.npz> --output_dir <output_dir>
"""

import os
import sys
import argparse
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
import time
from pathlib import Path
# Removed multiprocessing imports - sequential processing is faster for GM calculations

# Import DR4GM standards
from npz_format_standard import DR4GM_NPZ_Standard

def process_single_chunk_worker(chunk_data):
    """
    Worker function for processing a single chunk in parallel
    This function needs to be at module level for multiprocessing
    """
    chunk_id, acc_h1_chunk, acc_h2_chunk, dt, periods, periods_keys, total_gm_metrics_keys, gm_metrics_keys = chunk_data
    
    try:
        # Add paths for gmpe-smtk library
        import sys
        from pathlib import Path
        utils_dir = Path(__file__).parent
        gmpe_smtk_dir = utils_dir.parent / 'gmpe-smtk'
        if gmpe_smtk_dir.exists():
            sys.path.insert(0, str(gmpe_smtk_dir))
        
        from ComputeGroundMotionParametersFromSurfaceOutput_Hybrid_Lite import gmrotdpp_withPG
        
        n_stations = acc_h1_chunk.shape[0]
        results = {key: np.zeros(n_stations) for key in total_gm_metrics_keys}
        
        # Process each station in the chunk
        for i in range(n_stations):
            # Input acceleration is already in cm/s² from vectorized_vel_to_acc
            acc_h1_cm = acc_h1_chunk[i]  # Already in cm/s²
            acc_h2_cm = acc_h2_chunk[i]  # Already in cm/s²
            
            try:
                result = gmrotdpp_withPG(
                    acc_h1_cm, dt,
                    acc_h2_cm, dt,
                    periods,
                    percentile=50,
                    damping=0.05,
                    units='cm/s/s',
                    method='Nigam-Jennings'
                )
                
                # Extract basic metrics
                for key in gm_metrics_keys:
                    results[key][i] = result.get(key, 0.0)
                
                # Extract ALL spectral accelerations
                if 'Acceleration' in result and len(result['Acceleration']) >= len(periods):
                    for j, period_key in enumerate(periods_keys):
                        if j < len(result['Acceleration']):
                            results[period_key][i] = result['Acceleration'][j]
                            
            except Exception as e:
                # Leave as zeros for failed stations
                continue
        
        return chunk_id, results
        
    except ImportError as e:
        print(f"Worker {chunk_id}: gmrotdpp_withPG not available: {e}")
        # Return basic metrics only
        n_stations = acc_h1_chunk.shape[0]
        results = {key: np.zeros(n_stations) for key in total_gm_metrics_keys}
        
        # Basic PGA calculation (acceleration already in cm/s²)
        pga_h1 = np.max(np.abs(acc_h1_chunk), axis=1)
        pga_h2 = np.max(np.abs(acc_h2_chunk), axis=1)
        results['PGA'] = np.maximum(pga_h1, pga_h2)  # cm/s²
        
        return chunk_id, results

class FastNPZGroundMotionProcessor:
    """Fast vectorized ground motion processor for NPZ velocity time series data"""
    
    def __init__(self, velocity_npz: str, output_dir: str, dt: Optional[float] = None, 
                 chunk_size: int = 10000):
        """
        Initialize fast GM processor
        
        Args:
            velocity_npz: Path to NPZ file containing velocity time series
            output_dir: Directory to save ground motion results
            dt: Time step override (if None, read from NPZ file)
            chunk_size: Number of stations to process in each chunk
        """
        self.velocity_npz = Path(velocity_npz)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_size = chunk_size
        
        if not self.velocity_npz.exists():
            raise FileNotFoundError(f"Velocity NPZ file not found: {velocity_npz}")
        
        # Load velocity data
        self.logger = self._setup_logging()
        self.logger.info(f"Loading velocity data from {velocity_npz}")
        
        self.velocity_data = np.load(self.velocity_npz)
        self.station_ids = self.velocity_data['station_ids']
        self.locations = self.velocity_data['locations']
        
        # Read units from NPZ file
        if 'units' in self.velocity_data:
            self.velocity_units = str(self.velocity_data['units'])
            self.logger.info(f"Velocity units from NPZ: {self.velocity_units}")
        else:
            self.velocity_units = 'm/s'  # Default assumption
            self.logger.warning("No units field found in NPZ file, assuming m/s")
        
        # Determine unit conversion factor for velocity (m/s -> cm/s)
        if self.velocity_units == 'm/s':
            self.velocity_to_cm_factor = 100.0  # m/s to cm/s
        elif self.velocity_units == 'cm/s':
            self.velocity_to_cm_factor = 1.0    # already in cm/s
        else:
            self.logger.warning(f"Unknown velocity units: {self.velocity_units}, assuming m/s")
            self.velocity_to_cm_factor = 100.0
        
        # Determine time step
        if dt is not None:
            self.dt = dt
        elif 'dt_values' in self.velocity_data:
            self.dt = float(self.velocity_data['dt_values'][0])
        else:
            self.dt = 0.05
            
        # Ground motion parameters
        self.periods = np.array([0.100, 0.125, 0.25, 0.4, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 5])
        self.gm_metrics_keys = ['PGA', 'PGV', 'PGD', 'CAV']
        self.periods_keys = [f'RSA_T_{period:.3f}' for period in self.periods]
        self.total_gm_metrics_keys = self.gm_metrics_keys + self.periods_keys
        
        # Detect velocity components
        self.has_strike_normal = 'vel_strike' in self.velocity_data and 'vel_normal' in self.velocity_data
        self.has_xy = 'vel_x' in self.velocity_data and 'vel_y' in self.velocity_data
        
        self.logger.info(f"Loaded {len(self.station_ids)} stations")
        self.logger.info(f"Time step: {self.dt:.6f} seconds")
        self.logger.info(f"Chunk size: {self.chunk_size}")
        self.logger.info(f"Components: Strike/Normal={self.has_strike_normal}, X/Y={self.has_xy}")
    
    def _setup_logging(self):
        """Setup logging"""
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        return logging.getLogger(__name__)
    
    def vectorized_vel_to_acc(self, velocities: np.ndarray) -> np.ndarray:
        """
        Vectorized velocity to acceleration conversion with unit handling
        velocities: shape (n_stations, n_time_steps) in original units
        Returns: accelerations in cm/s²
        """
        n_stations, n_time = velocities.shape
        accelerations = np.zeros_like(velocities)
        
        # Convert velocities to cm/s first
        velocities_cm = velocities * self.velocity_to_cm_factor
        
        # Simple finite difference: acc[i] = (vel[i] - vel[i-1]) / dt
        # This gives cm/s²
        accelerations[:, 1:] = np.diff(velocities_cm, axis=1) / self.dt
        
        return accelerations
    
    def calc_basic_gm_metrics_vectorized(self, acc_h1: np.ndarray, acc_h2: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Vectorized calculation of basic ground motion metrics
        acc_h1, acc_h2: shape (n_stations, n_time_steps) in cm/s²
        Returns: All metrics in cm units (cm/s², cm/s, cm, cm/s)
        """
        n_stations = acc_h1.shape[0]
        
        # PGA - Peak Ground Acceleration (cm/s²)
        pga_h1 = np.max(np.abs(acc_h1), axis=1)
        pga_h2 = np.max(np.abs(acc_h2), axis=1)
        pga = np.maximum(pga_h1, pga_h2)
        
        # Integrate to get velocity and displacement
        # vel = integral(acc * dt) -> cm/s
        vel_h1 = np.cumsum(acc_h1, axis=1) * self.dt
        vel_h2 = np.cumsum(acc_h2, axis=1) * self.dt
        
        # PGV - Peak Ground Velocity (cm/s)
        pgv_h1 = np.max(np.abs(vel_h1), axis=1)
        pgv_h2 = np.max(np.abs(vel_h2), axis=1)
        pgv = np.maximum(pgv_h1, pgv_h2)
        
        # PGD - Peak Ground Displacement (cm)
        # disp = integral(vel * dt) -> cm
        disp_h1 = np.cumsum(vel_h1, axis=1) * self.dt
        disp_h2 = np.cumsum(vel_h2, axis=1) * self.dt
        pgd_h1 = np.max(np.abs(disp_h1), axis=1)
        pgd_h2 = np.max(np.abs(disp_h2), axis=1)
        pgd = np.maximum(pgd_h1, pgd_h2)
        
        # CAV - Cumulative Absolute Velocity (cm/s)
        # CAV = sum(|acc| * dt) -> cm/s
        cav = np.sum(np.abs(acc_h1), axis=1) * self.dt + np.sum(np.abs(acc_h2), axis=1) * self.dt
        
        return {
            'PGA': pga,    # cm/s²
            'PGV': pgv,    # cm/s
            'PGD': pgd,    # cm
            'CAV': cav     # cm/s
        }
    
    def process_chunk_with_gmrotd(self, acc_h1_chunk: np.ndarray, acc_h2_chunk: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Process a chunk using gmrotdpp_withPG for accurate SA calculations
        """
        try:
            # Add paths for gmpe-smtk library
            utils_dir = Path(__file__).parent
            gmpe_smtk_dir = utils_dir.parent / 'gmpe-smtk'
            if gmpe_smtk_dir.exists():
                sys.path.insert(0, str(gmpe_smtk_dir))
            
            from ComputeGroundMotionParametersFromSurfaceOutput_Hybrid_Lite import gmrotdpp_withPG
            
            n_stations = acc_h1_chunk.shape[0]
            results = {key: np.zeros(n_stations) for key in self.total_gm_metrics_keys}
            
            # Process each station in the chunk (no excessive logging for speed)
            for i in range(n_stations):
                
                # Input acceleration is already in cm/s² from vectorized_vel_to_acc
                acc_h1_cm = acc_h1_chunk[i]  # Already in cm/s²
                acc_h2_cm = acc_h2_chunk[i]  # Already in cm/s²
                
                try:
                    result = gmrotdpp_withPG(
                        acc_h1_cm, self.dt,
                        acc_h2_cm, self.dt,
                        self.periods,
                        percentile=50,
                        damping=0.05,
                        units='cm/s/s',
                        method='Nigam-Jennings'
                    )
                    
                    # Extract basic metrics
                    for key in self.gm_metrics_keys:
                        results[key][i] = result.get(key, 0.0)
                    
                    # Extract ALL spectral accelerations
                    if 'Acceleration' in result and len(result['Acceleration']) >= len(self.periods):
                        for j, period_key in enumerate(self.periods_keys):
                            if j < len(result['Acceleration']):
                                results[period_key][i] = result['Acceleration'][j]
                    # Skip incomplete data warnings for speed
                        
                except Exception as e:
                    # Leave as zeros for failed stations (no logging for speed)
                    continue
            
            return results
            
        except ImportError as e:
            self.logger.warning(f"gmrotdpp_withPG not available: {e}, using basic metrics only")
            basic_results = self.calc_basic_gm_metrics_vectorized(acc_h1_chunk, acc_h2_chunk)
            
            # Add zero SA values for all periods
            for period_key in self.periods_keys:
                basic_results[period_key] = np.zeros(acc_h1_chunk.shape[0])
            
            return basic_results
    
    def process_all_stations_parallel(self) -> Dict[str, np.ndarray]:
        """Parallel processing of all stations using multiprocessing"""
        start_time = time.time()
        num_stations = len(self.station_ids)
        
        self.logger.info(f"Processing {num_stations} stations with {self.n_processes} parallel processes")
        self.logger.info(f"Chunk size: {self.chunk_size}")
        
        # Initialize results
        results = {key: np.zeros(num_stations) for key in self.total_gm_metrics_keys}
        
        # Prepare chunks for parallel processing
        chunks_data = []
        chunk_ranges = []
        
        for chunk_id, chunk_start in enumerate(range(0, num_stations, self.chunk_size)):
            chunk_end = min(chunk_start + self.chunk_size, num_stations)
            chunk_ranges.append((chunk_start, chunk_end))
            
            # Load velocity data for this chunk
            if self.has_strike_normal:
                vel_h1_chunk = self.velocity_data['vel_strike'][chunk_start:chunk_end]
                vel_h2_chunk = self.velocity_data['vel_normal'][chunk_start:chunk_end]
            elif self.has_xy:
                vel_h1_chunk = self.velocity_data['vel_x'][chunk_start:chunk_end]
                vel_h2_chunk = self.velocity_data['vel_y'][chunk_start:chunk_end]
            else:
                raise ValueError("No recognized velocity components")
            
            # Convert to acceleration (vectorized)
            acc_h1_chunk = self.vectorized_vel_to_acc(vel_h1_chunk)
            acc_h2_chunk = self.vectorized_vel_to_acc(vel_h2_chunk)
            
            # Prepare data for worker
            chunk_data = (
                chunk_id,
                acc_h1_chunk,
                acc_h2_chunk,
                self.dt,
                self.periods,
                self.periods_keys,
                self.total_gm_metrics_keys,
                self.gm_metrics_keys
            )
            chunks_data.append(chunk_data)
        
        self.logger.info(f"Created {len(chunks_data)} chunks for parallel processing")
        
        # Process chunks in parallel with progress tracking
        if self.n_processes > 1:
            self.logger.info(f"Starting parallel processing with {self.n_processes} processes...")
            with Pool(processes=self.n_processes) as pool:
                # Use imap for progress tracking
                chunk_results = []
                total_chunks = len(chunks_data)
                
                for i, result in enumerate(pool.imap(process_single_chunk_worker, chunks_data)):
                    chunk_results.append(result)
                    
                    # Progress update
                    completed = i + 1
                    percent = (completed / total_chunks) * 100
                    elapsed = time.time() - start_time
                    rate = completed / elapsed
                    eta = (total_chunks - completed) / rate if rate > 0 else 0
                    
                    # Calculate stations processed
                    stations_completed = completed * self.chunk_size if completed < total_chunks else num_stations
                    station_rate = stations_completed / elapsed
                    
                    self.logger.info(f"Progress: {completed}/{total_chunks} chunks ({percent:.1f}%) - "
                                   f"{stations_completed:,}/{num_stations:,} stations - "
                                   f"{station_rate:.1f} st/s - ETA: {eta/60:.1f}min")
        else:
            self.logger.info("Processing sequentially (n_processes=1)...")
            chunk_results = []
            total_chunks = len(chunks_data)
            
            for i, chunk_data in enumerate(chunks_data):
                result = process_single_chunk_worker(chunk_data)
                chunk_results.append(result)
                
                # Progress update for sequential processing
                completed = i + 1
                percent = (completed / total_chunks) * 100
                elapsed = time.time() - start_time
                rate = completed / elapsed
                eta = (total_chunks - completed) / rate if rate > 0 else 0
                
                stations_completed = completed * self.chunk_size if completed < total_chunks else num_stations
                station_rate = stations_completed / elapsed
                
                self.logger.info(f"Progress: {completed}/{total_chunks} chunks ({percent:.1f}%) - "
                               f"{stations_completed:,}/{num_stations:,} stations - "
                               f"{station_rate:.1f} st/s - ETA: {eta/60:.1f}min")
        
        # Combine results from all chunks
        self.logger.info("Combining results from parallel chunks...")
        for chunk_id, chunk_result in chunk_results:
            chunk_start, chunk_end = chunk_ranges[chunk_id]
            
            for key in self.total_gm_metrics_keys:
                results[key][chunk_start:chunk_end] = chunk_result[key]
        
        duration = time.time() - start_time
        self.logger.info(f"Parallel processing completed in {duration:.2f}s ({num_stations/duration:.1f} stations/s)")
        
        # Verify results
        for key in self.total_gm_metrics_keys:
            non_zero = np.count_nonzero(results[key])
            self.logger.info(f"  {key}: {non_zero}/{num_stations} non-zero values")
        
        return results
    
    def process_all_stations_sequential(self) -> Dict[str, np.ndarray]:
        """Sequential processing with good progress tracking"""
        start_time = time.time()
        num_stations = len(self.station_ids)
        
        self.logger.info(f"Processing {num_stations} stations sequentially")
        self.logger.info(f"Chunk size: {self.chunk_size}")
        
        # Initialize results
        results = {key: np.zeros(num_stations) for key in self.total_gm_metrics_keys}
        
        # Process in chunks sequentially
        total_chunks = (num_stations + self.chunk_size - 1) // self.chunk_size
        
        for chunk_id, chunk_start in enumerate(range(0, num_stations, self.chunk_size)):
            chunk_end = min(chunk_start + self.chunk_size, num_stations)
            chunk_size_actual = chunk_end - chunk_start
            
            chunk_start_time = time.time()
            
            # Load velocity data for this chunk
            if self.has_strike_normal:
                vel_h1_chunk = self.velocity_data['vel_strike'][chunk_start:chunk_end]
                vel_h2_chunk = self.velocity_data['vel_normal'][chunk_start:chunk_end]
            elif self.has_xy:
                vel_h1_chunk = self.velocity_data['vel_x'][chunk_start:chunk_end]
                vel_h2_chunk = self.velocity_data['vel_y'][chunk_start:chunk_end]
            else:
                raise ValueError("No recognized velocity components")
            
            # Convert to acceleration (vectorized)
            acc_h1_chunk = self.vectorized_vel_to_acc(vel_h1_chunk)
            acc_h2_chunk = self.vectorized_vel_to_acc(vel_h2_chunk)
            
            # Process this chunk directly (faster than worker function)
            chunk_results = self.process_chunk_with_gmrotd(acc_h1_chunk, acc_h2_chunk)
            
            # Store results
            for key in self.total_gm_metrics_keys:
                results[key][chunk_start:chunk_end] = chunk_results[key]
            
            chunk_duration = time.time() - chunk_start_time
            chunk_rate = chunk_size_actual / chunk_duration
            
            # Progress update
            completed_chunks = chunk_id + 1
            percent = (completed_chunks / total_chunks) * 100
            elapsed = time.time() - start_time
            overall_rate = chunk_end / elapsed
            eta = (num_stations - chunk_end) / overall_rate if overall_rate > 0 else 0
            
            self.logger.info(f"Progress: {completed_chunks}/{total_chunks} chunks ({percent:.1f}%) - "
                           f"{chunk_end:,}/{num_stations:,} stations - "
                           f"Chunk: {chunk_rate:.1f} st/s, Overall: {overall_rate:.1f} st/s - "
                           f"ETA: {eta/60:.1f}min")
            
            # Force garbage collection
            import gc
            gc.collect()
        
        duration = time.time() - start_time
        self.logger.info(f"Sequential processing completed in {duration:.2f}s ({num_stations/duration:.1f} stations/s)")
        
        return results
    
    def save_results(self, gm_results: Dict[str, np.ndarray]) -> Dict[str, str]:
        """Save results to NPZ and summary files"""
        self.logger.info("Saving results")
        
        # Save NPZ file with ALL metrics
        layer4_data = {
            'station_ids': self.station_ids,
            'locations': self.locations,
            'PGA': gm_results['PGA'],
            'PGV': gm_results['PGV'],
            'PGD': gm_results['PGD'],
            'CAV': gm_results['CAV']
        }
        
        # Add spectral accelerations as both SA array AND individual keys
        sa_data = np.column_stack([gm_results[key] for key in self.periods_keys])
        layer4_data['SA'] = sa_data
        layer4_data['periods'] = self.periods
        layer4_data['percentile'] = 50
        
        # ALSO add individual RSA keys for visualization script
        for period_key in self.periods_keys:
            layer4_data[period_key] = gm_results[period_key]
        
        self.logger.info(f"Saving {len(self.periods)} spectral acceleration periods:")
        for i, (period, key) in enumerate(zip(self.periods, self.periods_keys)):
            non_zero = np.count_nonzero(gm_results[key])
            self.logger.info(f"  {key} (T={period:.3f}s): {non_zero}/{len(self.station_ids)} non-zero values")
        
        gm_file = self.output_dir / "ground_motion_metrics.npz"
        np.savez_compressed(gm_file, **layer4_data)
        
        # Save summary
        summary_file = self.output_dir / "gm_summary_statistics.txt"
        with open(summary_file, 'w') as f:
            f.write("Ground Motion Summary Statistics\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Source file: {self.velocity_npz}\n")
            f.write(f"Number of stations: {len(self.station_ids)}\n")
            f.write(f"Time step: {self.dt:.6f} seconds\n\n")
            
            for key in self.total_gm_metrics_keys:
                values = gm_results[key]
                f.write(f"{key:15s}: min={np.min(values):8.3e}, max={np.max(values):8.3e}, mean={np.mean(values):8.3e}, std={np.std(values):8.3e}\n")
        
        self.logger.info(f"Results saved: {gm_file}, {summary_file}")
        
        return {
            'ground_motion_npz': str(gm_file),
            'summary_statistics': str(summary_file)
        }
    
    def process_dataset(self) -> Dict[str, str]:
        """Process complete dataset"""
        start_time = time.time()
        
        # Process all stations sequentially (fastest for GM calculations)
        gm_results = self.process_all_stations_sequential()
        
        # Save results
        file_paths = self.save_results(gm_results)
        
        duration = time.time() - start_time
        file_paths.update({
            'total_stations': len(self.station_ids),
            'processing_time': duration
        })
        
        return file_paths

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Fast ground motion processing from NPZ velocity data')
    parser.add_argument('--velocity_npz', required=True, help='Input NPZ file containing velocity time series')
    parser.add_argument('--output_dir', required=True, help='Output directory for ground motion results')
    parser.add_argument('--dt', type=float, help='Time step override (if not in NPZ file)')
    parser.add_argument('--chunk_size', type=int, default=10000, help='Chunk size for processing (default: 10000)')
# Removed n_processes parameter - sequential processing is optimal
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        processor = FastNPZGroundMotionProcessor(
            args.velocity_npz, 
            args.output_dir, 
            args.dt,
            chunk_size=args.chunk_size
        )
        
        results = processor.process_dataset()
        
        print("\n=== Fast Ground Motion Processing Results ===")
        print(f"Ground motion NPZ: {results['ground_motion_npz']}")
        print(f"Summary statistics: {results['summary_statistics']}")
        print(f"Total stations: {results['total_stations']}")
        print(f"Processing time: {results['processing_time']:.2f}s")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()